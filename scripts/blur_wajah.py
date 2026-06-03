"""
blur_wajah.py
-------------
Blur wajah otomatis menggunakan OpenCV DNN (lebih akurat) + Haar Cascade (fallback).
Mendukung input: gambar (jpg/png) atau video (mp4).
Opsional: crop persegi, blur berbentuk bulat/elips, dan exclude tangan dari area wajah.

Cara pakai:
    # Satu gambar
    python blur_wajah.py --input foto.jpg

    # Semua gambar/video dalam folder
    python blur_wajah.py --input folder_gambar/

    # Ubah kekuatan blur (default 51, harus ganjil)
    python blur_wajah.py --input foto.jpg --blur 99

    # Simpan ke folder tertentu
    python blur_wajah.py --input foto.jpg --output hasil/

    # Crop persegi terbesar (center crop) sebelum blur
    python blur_wajah.py --input foto.jpg --crop-square

    # Blur berbentuk bulat/elips (bukan kotak)
    python blur_wajah.py --input foto.jpg --circular

    # Blur bulat + tangan tidak ikut di-blur
    python blur_wajah.py --input foto.jpg --circular --exclude-hands
"""

import cv2
import numpy as np
import argparse
import sys
import urllib.request
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# URL model OpenCV DNN face detector (res10_300x300_ssd)
DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL    = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

SCRIPT_DIR = Path(__file__).parent
DNN_PROTOTXT = SCRIPT_DIR / "deploy.prototxt"
DNN_MODEL    = SCRIPT_DIR / "res10_300x300_ssd_iter_140000.caffemodel"


def download_model():
    """Download model DNN jika belum ada."""
    if not DNN_PROTOTXT.exists():
        print("Mengunduh deploy.prototxt ...")
        urllib.request.urlretrieve(DNN_PROTOTXT_URL, DNN_PROTOTXT)
    if not DNN_MODEL.exists():
        print("Mengunduh caffemodel (~10MB) ...")
        urllib.request.urlretrieve(DNN_MODEL_URL, DNN_MODEL)


def load_detector():
    """Load DNN detector; fallback ke Haar Cascade jika gagal."""
    try:
        download_model()
        net = cv2.dnn.readNetFromCaffe(str(DNN_PROTOTXT), str(DNN_MODEL))
        print("[INFO] Menggunakan OpenCV DNN face detector.")
        return ("dnn", net)
    except Exception as e:
        print(f"[WARN] DNN gagal ({e}), fallback ke Haar Cascade.")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar = cv2.CascadeClassifier(cascade_path)
        return ("haar", haar)


def load_hand_detector():
    """Load MediaPipe hand detector. Return None jika mediapipe tidak terinstal."""
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
        )
        print("[INFO] MediaPipe hand detector aktif (tangan di-exclude dari blur).")
        return hands
    except ImportError:
        print("[WARN] mediapipe tidak terinstal. Jalankan: pip install mediapipe")
        print("[WARN] --exclude-hands dinonaktifkan.")
        return None


def detect_faces_dnn(net, frame, confidence_thresh=0.5):
    """Deteksi wajah dengan DNN, return list (x1,y1,x2,y2)."""
    h, w = frame.shape[:2]
    # Create blob from image (normalized to 300x300)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    # Extract detections above confidence threshold
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < confidence_thresh:
            continue
        # Scale coordinates back to original frame size
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)
        # Clamp coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_faces_haar(haar, frame):
    """Deteksi wajah dengan Haar Cascade, return list (x1,y1,x2,y2)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxes = []
    for (x, y, ww, hh) in faces:
        boxes.append((x, y, x + ww, y + hh))
    return boxes


def get_hand_boxes(frame, hand_detector):
    """Deteksi tangan dengan MediaPipe, return list (x1,y1,x2,y2)."""
    if hand_detector is None:
        return []
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb)
    boxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            pad = int(min(w, h) * 0.03)
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)
            boxes.append((x1, y1, x2, y2))
    return boxes


def crop_square(frame):
    """Crop frame ke persegi terbesar di tengah (center crop), tanpa resize."""
    h, w = frame.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return frame[y_start:y_start + size, x_start:x_start + size]


def blur_wajah_frame(frame, detector_info, blur_strength: int, confidence: float,
                     circular: bool = False, hand_detector=None):
    """Blur semua wajah yang terdeteksi dalam satu frame."""
    kind, model = detector_info
    # Select detection method (DNN or Haar Cascade)
    if kind == "dnn":
        face_boxes = detect_faces_dnn(model, frame, confidence)
    else:
        face_boxes = detect_faces_haar(model, frame)

    if not face_boxes:
        return frame

    # Ensure blur kernel size is odd (required for GaussianBlur)
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    # Create binary mask for areas to blur
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in face_boxes:
        if circular:
            # Draw elliptical mask (more natural looking)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx = (x2 - x1) // 2
            ry = (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        else:
            # Draw rectangular mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Exclude hands from blur area if hand detector is available
    if hand_detector is not None:
        for (hx1, hy1, hx2, hy2) in get_hand_boxes(frame, hand_detector):
            cv2.rectangle(mask, (hx1, hy1), (hx2, hy2), 0, -1)

    # Apply Gaussian blur and composite using mask
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    mask_3ch = cv2.merge([mask, mask, mask])
    return np.where(mask_3ch > 0, blurred, frame).astype(frame.dtype)


def proses_gambar(input_path: Path, output_path: Path, detector_info,
                  blur_strength: int, confidence: float,
                  do_crop: bool = False, circular: bool = False,
                  hand_detector=None):
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"[SKIP] Tidak bisa baca: {input_path}")
        return

    if do_crop:
        img = crop_square(img)

    hasil = blur_wajah_frame(img, detector_info, blur_strength, confidence,
                             circular, hand_detector)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), hasil)
    print(f"[OK]   {input_path.name}  ->  {output_path}")


def proses_video(input_path: Path, output_path: Path, detector_info,
                 blur_strength: int, confidence: float,
                 do_crop: bool = False, circular: bool = False,
                 hand_detector=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[SKIP] Tidak bisa buka video: {input_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_size = (min(fw, fh), min(fw, fh)) if do_crop else (fw, fh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if do_crop:
            frame = crop_square(frame)
        frame = blur_wajah_frame(frame, detector_info, blur_strength, confidence,
                                 circular, hand_detector)
        out.write(frame)
        idx += 1
        if idx % 10 == 0:
            pct = idx / total * 100 if total > 0 else 0
            print(f"  frame {idx}/{total} ({pct:.0f}%)", end="\r", flush=True)

    cap.release()
    out.release()
    print(f"\n[OK]   {input_path.name}  ->  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Blur wajah otomatis dengan OpenCV DNN")
    parser.add_argument("--input",  required=True,
                        help="Path gambar, folder, atau video")
    parser.add_argument("--output", default=None,
                        help="Folder output (default: <nama>_blurred/)")
    parser.add_argument("--blur",   type=int, default=51,
                        help="Kekuatan blur – bilangan ganjil (default: 51)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min confidence deteksi 0-1 (default: 0.5)")
    parser.add_argument("--crop-square", action="store_true",
                        help="Crop gambar ke persegi terbesar di tengah sebelum blur")
    parser.add_argument("--circular", action="store_true",
                        help="Blur berbentuk bulat/elips mengikuti bounding box wajah")
    parser.add_argument("--exclude-hands", action="store_true",
                        help="Tangan tidak ikut di-blur meskipun menutupi wajah "
                             "(butuh: pip install mediapipe)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[ERROR] Path tidak ditemukan: {input_path}")

    out_root = Path(args.output) if args.output else \
               input_path.parent / (input_path.stem + "_blurred")

    detector_info = load_detector()
    hand_detector = load_hand_detector() if args.exclude_hands else None
    do_crop  = args.crop_square
    circular = args.circular

    if input_path.is_dir():
        files = sorted([f for f in input_path.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS])
        if not files:
            sys.exit("[ERROR] Tidak ada file gambar/video di folder tersebut.")
        print(f"Memproses {len(files)} file ...")
        for f in files:
            out_file = out_root / f.name
            if f.suffix.lower() in IMAGE_EXTS:
                proses_gambar(f, out_file, detector_info, args.blur, args.confidence,
                              do_crop, circular, hand_detector)
            else:
                proses_video(f, out_file, detector_info, args.blur, args.confidence,
                             do_crop, circular, hand_detector)

    elif input_path.suffix.lower() in VIDEO_EXTS:
        proses_video(input_path, out_root / input_path.name,
                     detector_info, args.blur, args.confidence,
                     do_crop, circular, hand_detector)

    elif input_path.suffix.lower() in IMAGE_EXTS:
        proses_gambar(input_path, out_root / input_path.name,
                      detector_info, args.blur, args.confidence,
                      do_crop, circular, hand_detector)

    else:
        sys.exit(f"[ERROR] Format tidak didukung: {input_path.suffix}")

    if hand_detector is not None:
        hand_detector.close()

    print("\nSelesai! Hasil disimpan di:", out_root)


if __name__ == "__main__":
    main()
