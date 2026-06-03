"""
benchmark_i3d_realvideo.py
---------------------------
Mengukur waktu inferensi END-TO-END I3D RGB (WLASL style):
  preprocessing (decode + sample + resize + normalize) + forward pass

Preprocessing SAMA PERSIS dengan training (nslt_dataset.py):
  - Decode video dengan OpenCV (sama seperti load_rgb_frames_from_video)
  - Sample 64 frame merata dari video
  - Resize ke 256x256, lalu CenterCrop ke 224x224
  - Normalize: (pixel / 255) * 2 - 1  (range [-1, 1])
  - Input model: (1, 3, 64, 224, 224)

Cara pakai:
    python benchmark_i3d_realvideo.py
    python benchmark_i3d_realvideo.py --video_dir D:\TUGASAKHIR\Dataset
    python benchmark_i3d_realvideo.py --n_videos 10 --runs 20
"""

import sys
import time
import statistics
import random
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
I3D_DIR    = SCRIPT_DIR / "WLASL" / "code" / "I3D"
WORK_DIR   = SCRIPT_DIR / "work_dirs_WLASL_ori" / "wlasl_i3d_original_style_red_rerun"
DATASET_DIR = SCRIPT_DIR / "Dataset"

NUM_FOLDS   = 5
NUM_CLASSES = 10
NUM_FRAMES  = 64
RESIZE      = 256
CROP_SIZE   = 224

# ---------------------------------------------------------------------------
# Preprocessing (sama persis dengan nslt_dataset.load_rgb_frames_from_video)
# ---------------------------------------------------------------------------
def preprocess_video(video_path: Path) -> torch.Tensor:
    """
    Decode video (OpenCV) -> sample 64 frame -> resize 256 -> crop 224 -> normalize.
    Return: (1, 3, 64, 224, 224)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = NUM_FRAMES

    # Sample NUM_FRAMES frame merata dari video
    indices = np.linspace(0, max(total - 1, 0), NUM_FRAMES).astype(int)
    frames = []
    read_idx = 0
    buf = {}

    for idx in indices:
        if idx not in buf:
            while read_idx <= idx:
                ret, frame = cap.read()
                if not ret:
                    break
                buf[read_idx] = frame
                read_idx += 1
        frame = buf.get(idx)
        if frame is None:
            # Fallback: ulangi frame terakhir
            frame = frames[-1] if frames else np.zeros((RESIZE, RESIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (RESIZE, RESIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Pad jika kurang dari NUM_FRAMES
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1] if frames else np.zeros((RESIZE, RESIZE, 3), dtype=np.uint8))

    # Stack & normalize: (pixel / 255) * 2 - 1  (sama dengan nslt_dataset)
    arr = np.stack(frames[:NUM_FRAMES], axis=0).astype(np.float32)  # (T, H, W, 3)
    arr = (arr / 255.0) * 2.0 - 1.0

    # CenterCrop 224x224
    h, w = arr.shape[1], arr.shape[2]
    y0 = (h - CROP_SIZE) // 2
    x0 = (w - CROP_SIZE) // 2
    arr = arr[:, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE, :]          # (T, 224, 224, 3)

    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)              # (3, T, 224, 224)
    return tensor.unsqueeze(0)                                        # (1, 3, T, 224, 224)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(ckpt_path: Path, device: torch.device):
    if str(I3D_DIR) not in sys.path:
        sys.path.insert(0, str(I3D_DIR))
    from pytorch_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(NUM_CLASSES)

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Benchmark end-to-end
# ---------------------------------------------------------------------------
def benchmark(model, video_paths, device, warmup, runs):
    """Ukur waktu forward pass saja; preprocessing dilakukan di luar timer."""
    tensors = [preprocess_video(p).to(device) for p in video_paths]

    with torch.no_grad():
        for i in range(warmup):
            model(tensors[i % len(tensors)], pretrained=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for i in range(runs):
            x = tensors[i % len(tensors)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x, pretrained=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean":   statistics.mean(times),
        "std":    statistics.stdev(times),
        "min":    min(times),
        "max":    max(times),
        "median": statistics.median(times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default=str(DATASET_DIR))
    parser.add_argument("--n_videos",  type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs",   type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    video_dir = Path(args.video_dir)
    all_videos = sorted(video_dir.rglob("*.mp4"))
    if not all_videos:
        print(f"[ERROR] Tidak ada video .mp4 di {video_dir}")
        return
    random.seed(42)
    selected = random.sample(all_videos, min(args.n_videos, len(all_videos)))

    print("=" * 60)
    print("  I3D RGB (WLASL style) — End-to-End Inference Benchmark")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Input model: (1, 3, {NUM_FRAMES}, {CROP_SIZE}, {CROP_SIZE})")
    print(f"  Video dipakai: {len(selected)} file")
    print(f"  Warm-up    : {args.warmup}x  |  Runs : {args.runs}x per fold")
    print(f"  Waktu diukur: forward pass saja (tensor dari video asli)")

    fold_means = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt = WORK_DIR / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            print(f"\n  [Fold {fold}] SKIP — checkpoint tidak ditemukan")
            continue

        print(f"\n  [Fold {fold}] Memuat {ckpt.name} ...")
        model = load_model(ckpt, device)

        stats = benchmark(model, selected, device, args.warmup, args.runs)
        fold_means.append(stats["mean"])

        print(f"  [Fold {fold}] mean={stats['mean']:.2f} ms | "
              f"std={stats['std']:.2f} | "
              f"min={stats['min']:.2f} | "
              f"max={stats['max']:.2f} | "
              f"median={stats['median']:.2f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if fold_means:
        avg = sum(fold_means) / len(fold_means)
        print(f"\n  Rata-rata inference time end-to-end (5 fold): {avg:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
