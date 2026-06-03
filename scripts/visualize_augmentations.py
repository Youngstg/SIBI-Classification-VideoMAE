"""
Simpan setiap gambar sebagai 1 PNG terpisah, tanpa judul/label.
Struktur output:
  viz_augmentations/
    mixup/
      Aku_vs_Bapak/
        01_asli_Aku.png
        02_lam0.7.png
        03_lam0.5.png
        04_lam0.3.png
        05_asli_Bapak.png
      ...
    cutmix/
      Aku_vs_Bapak/
        01_asli_Aku.png
        02_lam0.75.png  (+ kotak merah)
        03_lam0.50.png
        04_lam0.25.png
        05_asli_Bapak.png
      ...
    random_erasing/
      Aku/
        01_asli.png
        02_erase.png
        03_erase.png
        04_erase.png
        05_erase.png
      ...
"""

import cv2
import numpy as np
import math
import os

np.random.seed(42)

OUT_DIR = r"D:\TUGASAKHIR\SourceCode\viz_augmentations"

LAST_VIDEOS = {
    "Aku":     r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Aku\Aku_090_mult.mp4",
    "Bapak":   r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Bapak\Bapak_090_mult.mp4",
    "Bawa":    r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Bawa\Bawa_088_mult.mp4",
    "Guru":    r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Guru\Guru_090_mult.mp4",
    "Ibu":     r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Ibu\Ibu_096_mult.mp4",
    "Kamu":    r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Kamu\Kamu_093_mult.mp4",
    "Mau":     r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Mau\Mau_099_mult.mp4",
    "Pulang":  r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Pulang\Pulang_100_mult.mp4",
    "Sekolah": r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Sekolah\Sekolah_088_mult.mp4",
    "Suka":    r"D:\TUGASAKHIR\SourceCode\folds_data\videos\Suka\Suka_090_mult.mp4",
}


def get_middle_frame(path, size=(224, 224)):
    """Extract middle frame from video file."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Extract frame at middle position
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return cv2.resize(frame, size)  # BGR format for cv2.imwrite


def save_img(img_bgr, path):
    """Save BGR image to file path with directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)


def mixup_bgr(img1, img2, lam):
    """Apply mixup augmentation: blend two images with mixing ratio lambda.
    Output = lambda * img1 + (1 - lambda) * img2
    """
    return np.clip(lam * img1.astype(np.float64)
                   + (1 - lam) * img2.astype(np.float64), 0, 255).astype(np.uint8)


def cutmix_bgr(img1, img2, lam):
    """Apply cutmix augmentation: replace patch of img1 with img2.
    - Creates rectangular patch from img2
    - Paste into center of img1
    - Draw red rectangle showing cut region
    """
    H, W = img1.shape[:2]
    cut_ratio = math.sqrt(1.0 - lam)  # Calculate patch size ratio
    cw, ch = int(W * cut_ratio), int(H * cut_ratio)  # Patch width and height
    cx, cy = W // 2, H // 2  # Center coordinates
    # Calculate patch boundaries (centered)
    x1 = max(0, cx - cw // 2); x2 = min(W, cx + cw // 2)
    y1 = max(0, cy - ch // 2); y2 = min(H, cy + ch // 2)
    out = img1.copy()
    # Copy patch from img2 to img1
    out[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    # Draw red rectangle to show cut region
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGR: red
    # Calculate actual mixing ratio based on patch size
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return out, lam_actual


def random_erase_bgr(img, seed=0):
    """Apply random erasing augmentation: cover random patch with noise.
    - Randomly select patch size and position
    - Fill with random pixel values
    - Draw green rectangle showing erased region
    """
    np.random.seed(seed)
    H, W = img.shape[:2]
    # Try up to 100 times to find valid patch
    for _ in range(100):
        # Random patch area (2% to 33% of image)
        area = np.random.uniform(0.02, 1/3) * H * W
        # Random aspect ratio
        asp  = np.exp(np.random.uniform(math.log(0.3), math.log(1/0.3)))
        eh, ew = int(math.sqrt(area * asp)), int(math.sqrt(area / asp))
        # Check if patch is smaller than image
        if eh < H and ew < W:
            top  = np.random.randint(0, H - eh)
            left = np.random.randint(0, W - ew)
            out  = img.copy()
            # Fill patch with random noise
            out[top:top+eh, left:left+ew] = np.random.randint(
                0, 256, (eh, ew, 3), dtype=np.uint8)
            # Draw green rectangle showing erased region
            cv2.rectangle(out, (left, top), (left+ew, top+eh), (0, 255, 0), 2)
            return out
    return img.copy()


def main():
    print("Baca frame...")
    # Load middle frame from each video
    F = {n: get_middle_frame(p) for n, p in LAST_VIDEOS.items()}
    print("  OK\n")

    # Define class pairs for augmentation visualization
    pairs = [
        ("Aku",   "Bapak"),
        ("Aku",   "Bawa"),
        ("Aku",   "Guru"),
        ("Aku",   "Ibu"),
        ("Bapak", "Kamu"),
        ("Bapak", "Mau"),
        ("Bawa",  "Pulang"),
        ("Guru",  "Sekolah"),
        ("Ibu",   "Suka"),
        ("Kamu",  "Mau"),
    ]

    # MIXUP VISUALIZATION
    print("Mixup...")
    lambdas_mx = [0.7, 0.5, 0.3]  # Different mixing ratios
    for n1, n2 in pairs:
        d = os.path.join(OUT_DIR, "mixup", f"{n1}_vs_{n2}")
        save_img(F[n1], os.path.join(d, f"01_asli_{n1}.png"))  # Original
        for i, lam in enumerate(lambdas_mx, start=2):
            # Generate mixup with different lambda values
            save_img(mixup_bgr(F[n1], F[n2], lam),
                     os.path.join(d, f"0{i}_lam{lam:.1f}.png"))
        save_img(F[n2], os.path.join(d, f"05_asli_{n2}.png"))  # Original
        print(f"  {n1}_vs_{n2}: 5 file")

    # CUTMIX VISUALIZATION
    print("\nCutMix...")
    lambdas_cm = [0.75, 0.50, 0.25]  # Different patch sizes
    for n1, n2 in pairs:
        d = os.path.join(OUT_DIR, "cutmix", f"{n1}_vs_{n2}")
        save_img(F[n1], os.path.join(d, f"01_asli_{n1}.png"))  # Original
        for i, lam in enumerate(lambdas_cm, start=2):
            # Generate cutmix with different patch sizes
            out, lam_a = cutmix_bgr(F[n1], F[n2], lam)
            save_img(out, os.path.join(d, f"0{i}_lam{lam_a:.2f}.png"))
        save_img(F[n2], os.path.join(d, f"05_asli_{n2}.png"))  # Original
        print(f"  {n1}_vs_{n2}: 5 file")

    # ================================================================
    # RANDOM ERASING
    # ================================================================
    print("\nRandom Erasing...")
    for name in LAST_VIDEOS:
        lbl = list(LAST_VIDEOS).index(name)
        d = os.path.join(OUT_DIR, "random_erasing", name)
        save_img(F[name], os.path.join(d, "01_asli.png"))
        for c in range(1, 5):
            save_img(random_erase_bgr(F[name], seed=lbl * 10 + c),
                     os.path.join(d, f"0{c+1}_erase_{c}.png"))
        print(f"  {name}: 5 file")

    total = (len(pairs) + len(pairs) + len(LAST_VIDEOS)) * 5
    print(f"\nSelesai. {total} PNG tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
