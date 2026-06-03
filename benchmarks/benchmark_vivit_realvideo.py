"""
benchmark_vivit_realvideo.py
-----------------------------
Mengukur waktu inferensi END-TO-END ViViT-B/16x2 K400:
  preprocessing (decode + sample + resize + normalize) + forward pass

Preprocessing SAMA PERSIS dengan training (finetune_vivit.py):
  - Decode video dengan decord
  - Uniform sampling: linspace 32 frame dari seluruh video
  - Resize ke 224x224
  - Normalize: (x / 255 - 0.5) / 0.5
  - Input model: (1, 32, 3, 224, 224)

Cara pakai:
    python benchmark_vivit_realvideo.py
    python benchmark_vivit_realvideo.py --video_dir D:\TUGASAKHIR\Dataset
    python benchmark_vivit_realvideo.py --n_videos 10 --runs 20
"""

import time
import statistics
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VivitForVideoClassification, VivitConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
WORK_DIR    = SCRIPT_DIR / "work_dirs_vivit"
DATASET_DIR = SCRIPT_DIR / "Dataset"

NUM_FOLDS   = 5
NUM_CLASSES = 10
NUM_FRAMES  = 32   # sesuai positional embedding checkpoint
IMG_SIZE    = 224

# ---------------------------------------------------------------------------
# Preprocessing (sama persis dengan VideoFolderDataset._load_video)
# ---------------------------------------------------------------------------
def preprocess_video(video_path: Path) -> torch.Tensor:
    """
    Decode video -> uniform 32 frame -> resize 224 -> normalize (x-0.5)/0.5.
    Return: (1, 32, 3, 224, 224)
    """
    from decord import VideoReader, cpu as dcpu

    vr = VideoReader(str(video_path), ctx=dcpu(0), num_threads=1)
    total = len(vr)

    # Uniform sampling (inference = deterministic linspace)
    indices = np.linspace(0, total - 1, NUM_FRAMES).astype(int)

    frames = vr.get_batch(indices).asnumpy()          # (T, H, W, 3) uint8
    frames = torch.from_numpy(frames).float() / 255.0 # (T, H, W, 3)
    frames = frames.permute(0, 3, 1, 2)               # (T, 3, H, W)

    frames = F.interpolate(frames, size=(IMG_SIZE, IMG_SIZE),
                           mode="bilinear", align_corners=False)

    frames = (frames - 0.5) / 0.5                     # normalize ViViT style
    return frames.unsqueeze(0)                         # (1, T, 3, H, W)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hf_cfg = VivitConfig(**ckpt.get("model_config", {}))
    model = VivitForVideoClassification(hf_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
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
            model(pixel_values=tensors[i % len(tensors)])
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for i in range(runs):
            x = tensors[i % len(tensors)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(pixel_values=x)
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
    print("  ViViT-B/16x2 K400 — End-to-End Inference Benchmark")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Input model: (1, {NUM_FRAMES}, 3, {IMG_SIZE}, {IMG_SIZE})")
    print(f"  Video dipakai: {len(selected)} file")
    print(f"  Warm-up    : {args.warmup}x  |  Runs : {args.runs}x per fold")
    print(f"  Waktu diukur: forward pass saja (tensor dari video asli)")

    fold_means = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt = WORK_DIR / f"fold_{fold}" / "checkpoint-best.pth"
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
