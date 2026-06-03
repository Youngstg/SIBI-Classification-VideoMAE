"""
benchmark_i3d.py
----------------
Mengukur waktu inferensi I3D RGB (WLASL style)
dari best checkpoint setiap fold.

Input: (1, 3, 64, 224, 224)  ->  batch x channel x frames x H x W
Sama persis dengan format training (total_frames=64 di nslt_dataset.py).

Cara pakai:
    python benchmark_i3d.py
    python benchmark_i3d.py --warmup 20 --runs 100
"""

import sys
import time
import statistics
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
I3D_DIR    = SCRIPT_DIR / "WLASL" / "code" / "I3D"
WORK_DIR   = SCRIPT_DIR / "work_dirs_WLASL_ori" / "wlasl_i3d_original_style_red_rerun"
NUM_FOLDS   = 5
NUM_CLASSES = 10
NUM_FRAMES  = 64   # total_frames=64 di nslt_dataset.py
IMG_SIZE    = 224

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
    # Bersihkan prefix 'module.' jika ada (DataParallel)
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model.to(device)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(model, device, warmup, runs):
    # Input dummy sama persis format training: (B, C, T, H, W)
    dummy = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy, pretrained=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy, pretrained=False)
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs",   type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    print("=" * 55)
    print("  I3D RGB (WLASL style) — Inference Time Benchmark")
    print("=" * 55)
    print(f"  Device   : {device}")
    print(f"  Input    : (1, 3, {NUM_FRAMES}, {IMG_SIZE}, {IMG_SIZE})")
    print(f"  Warm-up  : {args.warmup}x  |  Runs : {args.runs}x per fold")

    fold_means = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt = WORK_DIR / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            print(f"\n  [Fold {fold}] SKIP — checkpoint tidak ditemukan")
            continue

        print(f"\n  [Fold {fold}] Memuat {ckpt.name} ...")
        model = load_model(ckpt, device)

        stats = benchmark(model, device, args.warmup, args.runs)
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
        print(f"\n  Rata-rata inference time (5 fold): {avg:.2f} ms")
    print("=" * 55)


if __name__ == "__main__":
    main()
