"""
benchmark_vits_ssv2.py
----------------------
Mengukur waktu inferensi VideoMAE ViT-Small SSv2 (two-stage)
dari best checkpoint setiap fold.

Input: (1, 3, 16, 224, 224)  ->  batch x channel x frames x H x W
Sama persis dengan format training (SSv2 uniform loader, 16 frames).

Cara pakai:
    python benchmark_vits_ssv2.py
    python benchmark_vits_ssv2.py --warmup 20 --runs 100
"""

import sys
import time
import statistics
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
VIDEOMAE_DIR = SCRIPT_DIR / "VideoMAE"
WORK_DIR     = SCRIPT_DIR / "work_dirs_vits_ssv2_uniform_twostage"
NUM_FOLDS    = 5
NUM_CLASSES  = 10
NUM_FRAMES   = 16
IMG_SIZE     = 224

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(ckpt_path: Path, device: torch.device):
    if str(VIDEOMAE_DIR) not in sys.path:
        sys.path.insert(0, str(VIDEOMAE_DIR))
    import modeling_finetune

    model = modeling_finetune.vit_small_patch16_224(
        pretrained=False,
        num_classes=NUM_CLASSES,
        all_frames=NUM_FRAMES,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
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
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
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
    print("  VideoMAE ViT-Small SSv2 — Inference Time Benchmark")
    print("=" * 55)
    print(f"  Device   : {device}")
    print(f"  Input    : (1, 3, {NUM_FRAMES}, {IMG_SIZE}, {IMG_SIZE})")
    print(f"  Warm-up  : {args.warmup}x  |  Runs : {args.runs}x per fold")

    fold_means = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt = WORK_DIR / f"fold_{fold}" / "stage2" / "checkpoint-best.pth"
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
