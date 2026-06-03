"""
benchmark_vits_ssv2_realvideo.py
---------------------------------
Mengukur waktu inferensi END-TO-END VideoMAE ViT-Small SSv2:
  preprocessing (decode + sample + resize + normalize) + forward pass

Preprocessing SAMA PERSIS dengan training:
  - Decode video dengan decord
  - Uniform/segment sampling: bagi video jadi 16 segmen, ambil 1 frame tiap segmen
  - Resize + CenterCrop 224x224
  - Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
  - Input model: (1, 3, 16, 224, 224)

Cara pakai:
    python benchmark_vits_ssv2_realvideo.py
    python benchmark_vits_ssv2_realvideo.py --video_dir D:\TUGASAKHIR\Dataset
    python benchmark_vits_ssv2_realvideo.py --n_videos 10 --runs 20
"""

import sys
import time
import statistics
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
VIDEOMAE_DIR = SCRIPT_DIR / "VideoMAE"
WORK_DIR     = SCRIPT_DIR / "work_dirs_vits_ssv2_uniform_twostage"
DATASET_DIR  = SCRIPT_DIR / "Dataset"

NUM_FOLDS   = 5
NUM_CLASSES = 10
NUM_FRAMES  = 16
IMG_SIZE    = 224

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ---------------------------------------------------------------------------
# Preprocessing (sama persis dengan SSv2 uniform/segment loader di training)
# ---------------------------------------------------------------------------
def preprocess_video(video_path: Path) -> torch.Tensor:
    """
    Decode video -> uniform segment sampling -> resize 224 -> normalize.
    Return: (1, 3, 16, 224, 224)
    """
    from decord import VideoReader, cpu as dcpu

    vr = VideoReader(str(video_path), ctx=dcpu(0), num_threads=1)
    total = len(vr)

    # Uniform / segment-based sampling: bagi total frame jadi NUM_FRAMES segmen
    if total <= NUM_FRAMES:
        indices = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
    else:
        # Ambil 1 frame dari tiap segmen (tengah segmen, sesuai SSv2 style)
        boundaries = np.linspace(0, total, NUM_FRAMES + 1).astype(int)
        indices = np.array([
            (boundaries[i] + boundaries[i + 1]) // 2
            for i in range(NUM_FRAMES)
        ])

    frames = vr.get_batch(indices).asnumpy()          # (T, H, W, 3) uint8
    frames = torch.from_numpy(frames).float() / 255.0 # (T, H, W, 3)
    frames = frames.permute(0, 3, 1, 2)               # (T, 3, H, W)

    frames = F.interpolate(frames, size=(IMG_SIZE, IMG_SIZE),
                           mode="bilinear", align_corners=False)

    frames = (frames - MEAN) / STD                    # (T, 3, H, W)
    frames = frames.permute(1, 0, 2, 3)               # (3, T, H, W)
    return frames.unsqueeze(0)                         # (1, 3, T, H, W)


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
# Benchmark end-to-end
# ---------------------------------------------------------------------------
def benchmark(model, video_paths, device, warmup, runs):
    """Ukur waktu forward pass saja; preprocessing dilakukan di luar timer."""
    tensors = [preprocess_video(p).to(device) for p in video_paths]

    with torch.no_grad():
        for i in range(warmup):
            model(tensors[i % len(tensors)])
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for i in range(runs):
            x = tensors[i % len(tensors)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
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
    print("  VideoMAE ViT-Small SSv2 — End-to-End Inference Benchmark")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Input model: (1, 3, {NUM_FRAMES}, {IMG_SIZE}, {IMG_SIZE})")
    print(f"  Video dipakai: {len(selected)} file")
    print(f"  Warm-up    : {args.warmup}x  |  Runs : {args.runs}x per fold")
    print(f"  Waktu diukur: forward pass saja (tensor dari video asli)")

    fold_means = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt = WORK_DIR / f"fold_{fold}" / "stage2" / "checkpoint-best.pth"
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
