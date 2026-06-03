"""
benchmark_inference_time.py
----------------------------
Mengukur inference time (waktu inferensi) per model menggunakan
best checkpoint dari setiap fold (5 fold).

Model yang diukur:
  1. VideoMAE ViT-Base  K400  (work_dirs_vitb_k400_dense_twostage)  - stage2
  2. VideoMAE ViT-Small SSv2  (work_dirs_vits_ssv2_uniform_twostage) - stage2
  3. ViViT-B/16x2 K400        (work_dirs_vivit)
  4. I3D RGB (WLASL style)    (work_dirs_WLASL_ori/wlasl_i3d_original_style_red_rerun)

Cara pakai:
    python benchmark_inference_time.py
    python benchmark_inference_time.py --device cpu
    python benchmark_inference_time.py --warmup 20 --runs 100
    python benchmark_inference_time.py --batch_size 1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
VIDEOMAE_DIR = SCRIPT_DIR / "VideoMAE"
I3D_DIR      = SCRIPT_DIR / "WLASL" / "code" / "I3D"

WORK_VITB   = SCRIPT_DIR / "work_dirs_vitb_k400_dense_twostage"
WORK_VITS   = SCRIPT_DIR / "work_dirs_vits_ssv2_uniform_twostage"
WORK_VIVIT  = SCRIPT_DIR / "work_dirs_vivit"
WORK_I3D    = SCRIPT_DIR / "work_dirs_WLASL_ori" / "wlasl_i3d_original_style_red_rerun"

NUM_FOLDS   = 5
NUM_CLASSES = 10   # SIBI dataset: 10 kata


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_videomae(model_name: str, ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load VideoMAE ViT model (ViT-B atau ViT-S) dari checkpoint."""
    if str(VIDEOMAE_DIR) not in sys.path:
        sys.path.insert(0, str(VIDEOMAE_DIR))

    import modeling_finetune  # noqa: F401 – registers models into timm registry

    # Panggil factory function langsung agar tidak ada kwarg tambahan dari timm.create_model
    factory = getattr(modeling_finetune, model_name)
    model = factory(
        pretrained=False,
        num_classes=num_classes,
        all_frames=16,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Hapus key yang tidak cocok (misalnya pos_embed dengan ukuran beda)
    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    missing = set(model_sd.keys()) - set(filtered.keys())
    if missing:
        print(f"    [WARN] {len(missing)} key tidak dimuat (shape mismatch atau tidak ada)")

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model.to(device)


def _load_vivit(ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load ViViT dari checkpoint lokal."""
    from transformers import VivitForVideoClassification, VivitConfig as HfVivitConfig
    import torch.nn as nn_local

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_cfg_dict = ckpt.get("model_config", {})

    if model_cfg_dict:
        hf_cfg = HfVivitConfig(**model_cfg_dict)
        model = VivitForVideoClassification(hf_cfg)
    else:
        model = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        model.classifier = nn_local.Linear(model.classifier.in_features, num_classes)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def _load_i3d(ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load I3D dari checkpoint WLASL-style."""
    if str(I3D_DIR) not in sys.path:
        sys.path.insert(0, str(I3D_DIR))

    from pytorch_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(num_classes)

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Hapus prefix 'module.' jika ada (dari DataParallel)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_k] = v

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Input shape per model
# ---------------------------------------------------------------------------

def make_dummy_input(model_key: str, batch_size: int, device: torch.device) -> torch.Tensor:
    """Buat tensor dummy sesuai format input tiap model."""
    if model_key in ("vitb_k400", "vits_ssv2"):
        # (B, C, T, H, W)
        return torch.randn(batch_size, 3, 16, 224, 224, device=device)
    elif model_key == "vivit":
        # (B, T, C, H, W)  -> pixel_values ViViT HuggingFace
        return torch.randn(batch_size, 16, 3, 224, 224, device=device)
    elif model_key == "i3d":
        # (B, C, T, H, W)
        return torch.randn(batch_size, 3, 64, 224, 224, device=device)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def run_forward(model: nn.Module, dummy: torch.Tensor, model_key: str) -> None:
    """Jalankan satu forward pass."""
    if model_key == "vivit":
        model(pixel_values=dummy)
    elif model_key == "i3d":
        model(dummy, pretrained=False)
    else:
        model(dummy)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def measure_inference_time(
    model: nn.Module,
    model_key: str,
    batch_size: int,
    device: torch.device,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    """
    Ukur inference time dalam milidetik.
    Return dict: mean_ms, std_ms, min_ms, max_ms.
    """
    dummy = make_dummy_input(model_key, batch_size, device)
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            run_forward(model, dummy, model_key)
            if use_cuda:
                torch.cuda.synchronize()

        # Ukur
        times_ms: List[float] = []
        for _ in range(runs):
            if use_cuda:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_forward(model, dummy, model_key)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                run_forward(model, dummy, model_key)
                t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    import statistics
    return {
        "mean_ms":   statistics.mean(times_ms),
        "std_ms":    statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms":    min(times_ms),
        "max_ms":    max(times_ms),
        "median_ms": statistics.median(times_ms),
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ModelSpec = Tuple[str, str, Callable[[Path], nn.Module], str]

def build_model_specs(num_classes: int, device: torch.device) -> List[Tuple[str, str, Callable, str, Path]]:
    """
    Return list of (label, model_key, loader_fn, ckpt_pattern, work_dir).
    ckpt_pattern: glob pattern relatif ke work_dir/fold_X/
    """
    def vitb_loader(p): return _load_videomae("vit_base_patch16_224", p, num_classes, device)
    def vits_loader(p): return _load_videomae("vit_small_patch16_224", p, num_classes, device)
    def vivit_loader(p): return _load_vivit(p, num_classes, device)
    def i3d_loader(p):  return _load_i3d(p, num_classes, device)

    return [
        ("VideoMAE ViT-B K400",  "vitb_k400", vitb_loader, "stage2/checkpoint-best.pth", WORK_VITB),
        ("VideoMAE ViT-S SSv2",  "vits_ssv2", vits_loader, "stage2/checkpoint-best.pth", WORK_VITS),
        ("ViViT-B/16x2 K400",    "vivit",     vivit_loader, "checkpoint-best.pth",        WORK_VIVIT),
        ("I3D RGB (WLASL style)", "i3d",       i3d_loader,  "best.pt",                     WORK_I3D),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_banner(text: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference time semua model SIBI"
    )
    parser.add_argument("--device",     default="auto",
                        help="cuda / cpu / auto (default: auto)")
    parser.add_argument("--warmup",     type=int, default=10,
                        help="Jumlah forward pass warm-up (default: 10)")
    parser.add_argument("--runs",       type=int, default=50,
                        help="Jumlah pengukuran per fold (default: 50)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size input dummy (default: 1)")
    parser.add_argument("--output",     default="inference_time_results.csv",
                        help="Path file CSV hasil (default: inference_time_results.csv)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print_banner("BENCHMARK INFERENCE TIME - MODEL SIBI")
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Warm-up    : {args.warmup} runs")
    print(f"  Measure    : {args.runs} runs per fold")
    print(f"  Num folds  : {NUM_FOLDS}")

    model_specs = build_model_specs(NUM_CLASSES, device)

    # Hasil akhir: { label: { fold_1: {...}, ..., avg: {...} } }
    all_results: Dict[str, Dict] = {}
    csv_rows: List[Dict] = []

    for label, model_key, loader_fn, ckpt_pattern, work_dir in model_specs:
        print_banner(f"{label}")
        all_results[label] = {}

        fold_means: List[float] = []

        for fold_idx in range(1, NUM_FOLDS + 1):
            ckpt_path = work_dir / f"fold_{fold_idx}" / ckpt_pattern
            if not ckpt_path.exists():
                print(f"  [SKIP] Fold {fold_idx}: checkpoint tidak ditemukan: {ckpt_path}")
                continue

            print(f"\n  [Fold {fold_idx}] Memuat checkpoint: {ckpt_path.name} ...")
            try:
                model = loader_fn(ckpt_path)
            except Exception as e:
                print(f"  [ERROR] Gagal memuat model fold {fold_idx}: {e}")
                continue

            print(f"  [Fold {fold_idx}] Mengukur inference time ...")
            try:
                stats = measure_inference_time(
                    model, model_key, args.batch_size, device,
                    warmup=args.warmup, runs=args.runs,
                )
            except Exception as e:
                print(f"  [ERROR] Gagal mengukur fold {fold_idx}: {e}")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            print(
                f"  [Fold {fold_idx}] mean={stats['mean_ms']:.2f} ms | "
                f"std={stats['std_ms']:.2f} | "
                f"min={stats['min_ms']:.2f} | "
                f"max={stats['max_ms']:.2f} | "
                f"median={stats['median_ms']:.2f}"
            )

            all_results[label][f"fold_{fold_idx}"] = stats
            fold_means.append(stats["mean_ms"])

            csv_rows.append({
                "model": label,
                "fold": fold_idx,
                "mean_ms":   round(stats["mean_ms"],   3),
                "std_ms":    round(stats["std_ms"],    3),
                "min_ms":    round(stats["min_ms"],    3),
                "max_ms":    round(stats["max_ms"],    3),
                "median_ms": round(stats["median_ms"], 3),
            })

            # Bebaskan memori
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Rata-rata lintas fold
        if fold_means:
            avg_mean = sum(fold_means) / len(fold_means)
            print(f"\n  --> Rata-rata mean inference (5 fold): {avg_mean:.2f} ms")
            csv_rows.append({
                "model": label,
                "fold": "avg",
                "mean_ms":   round(avg_mean, 3),
                "std_ms":    "",
                "min_ms":    "",
                "max_ms":    "",
                "median_ms": "",
            })

    # ---------------------------------------------------------------------------
    # Ringkasan
    # ---------------------------------------------------------------------------
    print_banner("RINGKASAN HASIL (mean inference time per model)")
    header = f"{'Model':<30} {'Fold 1':>10} {'Fold 2':>10} {'Fold 3':>10} {'Fold 4':>10} {'Fold 5':>10} {'Avg (ms)':>10}"
    print(header)
    print("-" * len(header))

    for label, fold_data in all_results.items():
        fold_vals = []
        for f in range(1, NUM_FOLDS + 1):
            key = f"fold_{f}"
            if key in fold_data:
                fold_vals.append(f"{fold_data[key]['mean_ms']:.2f}")
            else:
                fold_vals.append("  N/A")

        means = [fold_data[f"fold_{f}"]["mean_ms"] for f in range(1, NUM_FOLDS + 1) if f"fold_{f}" in fold_data]
        avg_str = f"{sum(means)/len(means):.2f}" if means else "N/A"

        row_str = f"{label:<30}"
        for v in fold_vals:
            row_str += f" {v:>10}"
        row_str += f" {avg_str:>10}"
        print(row_str)

    # ---------------------------------------------------------------------------
    # Simpan CSV
    # ---------------------------------------------------------------------------
    output_path = SCRIPT_DIR / args.output
    fieldnames = ["model", "fold", "mean_ms", "std_ms", "min_ms", "max_ms", "median_ms"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[INFO] Hasil disimpan ke: {output_path}")
    print_banner("SELESAI")


if __name__ == "__main__":
    main()
