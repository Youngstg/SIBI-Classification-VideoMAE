"""
benchmark_vivit.py
------------------
Mengukur waktu inferensi ViViT-B/16x2 K400
dari best checkpoint setiap fold.

Input: (1, 32, 3, 224, 224)  ->  batch x frames x channel x H x W
Sesuai positional embedding checkpoint (32 frames, tubelet=2).

Cara pakai:
    python benchmark_vivit.py
    python benchmark_vivit.py --warmup 20 --runs 100
"""

import time
import statistics
from pathlib import Path

import torch
from transformers import VivitForVideoClassification, VivitConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
WORK_DIR    = SCRIPT_DIR / "work_dirs_vivit"
NUM_FOLDS   = 5
NUM_CLASSES = 10
NUM_FRAMES  = 32   # sesuai pos_embed checkpoint: (1, 3137, 768) = 16*196+1
IMG_SIZE    = 224

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    cfg_dict = ckpt.get("model_config", {})
    hf_cfg = VivitConfig(**cfg_dict)
    model = VivitForVideoClassification(hf_cfg)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model.to(device)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(model, device, warmup, runs):
    # Input dummy sama persis format training: (B, T, C, H, W)
    dummy = torch.randn(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(pixel_values=dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(pixel_values=dummy)
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
    print("  ViViT-B/16x2 K400 — Inference Time Benchmark")
    print("=" * 55)
    print(f"  Device   : {device}")
    print(f"  Input    : (1, {NUM_FRAMES}, 3, {IMG_SIZE}, {IMG_SIZE})")
    print(f"  Warm-up  : {args.warmup}x  |  Runs : {args.runs}x per fold")

    fold_means = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt = WORK_DIR / f"fold_{fold}" / "checkpoint-best.pth"
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
