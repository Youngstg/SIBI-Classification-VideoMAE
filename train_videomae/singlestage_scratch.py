"""Single-stage VideoMAE ViT-Small training on SIBI, from random init.

This follows the official SSV2 fine-tuning recipe (sample=2, 2x3 test clips)
but drops the ``--finetune`` argument so everything trains from scratch.
It's the single-stage counterpart to ``twostage_scratch.py`` and exists as
the "is two-stage even necessary?" ablation.

Run from the project root:
    python train_videomae/singlestage_scratch.py --start_fold 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT.parent))

from train_videomae.evaluation import ReportConfig, generate_reports
from train_videomae.training_utils import (
    banner,
    best_acc_from_log,
    cleanup_checkpoints,
    ensure_paths_exist,
    load_class_names,
    read_log,
    run_stage,
)

PROJECT_ROOT = _PACKAGE_ROOT.parent
VIDEOMAE_DIR = PROJECT_ROOT / "VideoMAE"
FOLDS_DIR = PROJECT_ROOT / "folds_data"


@dataclass
class SingleStageConfig:
    label: str = "ViT-Small | SSV2 | Single-Stage | Scratch"
    output_base: Path = field(
        default_factory=lambda: PROJECT_ROOT / "work_dirs" / "vits_ssv2_singlestage_scratch"
    )
    model_name: str = "vit_small_patch16_224"

    num_folds: int = 5
    num_frames: int = 16
    sampling_rate: int = 1
    batch_size: int = 12
    num_workers: int = 12
    grad_accum: int = 2
    weight_decay: float = 0.05
    epochs: int = 40
    lr: str = "1e-3"
    warmup_epochs: int = 5
    clip_grad: str = "1.0"

    # Number of repeated samples per video during training (official SSV2 = 2).
    num_sample: int = 2
    # Number of test-time clips / crops (official SSV2 = 2x3).
    test_num_segment: int = 2
    test_num_crop: int = 3


CONFIG = SingleStageConfig()


# ---------------------------------------------------------------------------
# Command builder + per-fold runner
# ---------------------------------------------------------------------------

def build_command(
    config: SingleStageConfig,
    num_classes: int,
    train_csv: Path,
    val_csv: Path,
    fold_output_dir: Path,
) -> List[str]:
    """Build command for single-stage training from scratch.
    
    Follows official SSV2 recipe but omits --finetune (trains from random init).
    Includes test-time augmentation parameters (2 segments x 3 crops).
    """
    return [
        sys.executable, "run_class_finetuning.py",
        "--model",            config.model_name,
        "--data_set",         "SSV2",
        "--data_path",        str(train_csv),
        "--eval_data_path",   str(val_csv),
        "--output_dir",       str(fold_output_dir),
        "--log_dir",          str(fold_output_dir),
        "--nb_classes",       str(num_classes),
        "--batch_size",       str(config.batch_size),
        "--num_sample",       str(config.num_sample),
        "--input_size",       "224",
        "--short_side_size",  "224",
        "--num_frames",       str(config.num_frames),
        "--sampling_rate",    str(config.sampling_rate),
        "--num_workers",      str(config.num_workers),
        "--opt",              "adamw",
        "--lr",               config.lr,
        "--weight_decay",     str(config.weight_decay),
        "--epochs",           str(config.epochs),
        "--save_ckpt_freq",   "10",
        "--update_freq",      str(config.grad_accum),
        "--warmup_epochs",    str(config.warmup_epochs),
        "--clip_grad",        config.clip_grad,
        "--test_num_segment", str(config.test_num_segment),
        "--test_num_crop",    str(config.test_num_crop),
        "--use_checkpoint",
    ]


def train_fold(config: SingleStageConfig, fold_idx: int, num_classes: int) -> float:
    """Train single fold with single-stage configuration.
    
    Args:
        config: Training configuration
        fold_idx: Fold number (1-5)
        num_classes: Number of output classes
    
    Returns:
        Best validation accuracy achieved for this fold
    """
    train_csv = FOLDS_DIR / f"train_fold_{fold_idx}.csv"
    val_csv = FOLDS_DIR / f"val_fold_{fold_idx}.csv"
    fold_output_dir = config.output_base / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists() or not val_csv.exists():
        print(f"[SKIP] CSV fold {fold_idx} tidak ditemukan.")
        return 0.0

    banner(f"FOLD {fold_idx} - SINGLE-STAGE SCRATCH", char="-")

    cmd = build_command(config, num_classes, train_csv, val_csv, fold_output_dir)
    ok, acc = run_stage(
        cmd,
        stage_name=f"SingleStage Fold{fold_idx}",
        epochs_total=config.epochs,
        cwd=str(VIDEOMAE_DIR),
        early_stop_patience=None,
    )
    if not ok:
        return 0.0

    best_acc = best_acc_from_log(fold_output_dir / "log.txt") or acc
    print(f"\n[RESULT] Fold {fold_idx} | Best Acc@1: {best_acc:.2f}%")
    cleanup_checkpoints(fold_output_dir)
    return best_acc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_run_header(config: SingleStageConfig, num_classes: int, class_names: Sequence[str], start_fold: int) -> None:
    banner(f"SINGLE-STAGE TRAINING - {config.label}")
    print(f"  Kelas        : {num_classes} ({', '.join(class_names)})")
    print("  Init model   : random weights")
    print("  Data loader  : SSV2 (uniform / segment-based sampling)")
    print(f"  Num frames   : {config.num_frames}")
    print(f"  Num sample   : {config.num_sample}")
    print(f"  Test clips   : {config.test_num_segment} clip x {config.test_num_crop} crop")
    print(f"  Sampling arg : {config.sampling_rate} (dikirim untuk kompatibilitas)")
    print(f"  Num workers  : {config.num_workers}")
    print(f"  Batch size   : {config.batch_size} (effective: {config.batch_size * config.grad_accum})")
    print(f"  Epochs       : {config.epochs}")
    print(f"  LR           : {config.lr}")
    print(f"  Warmup       : {config.warmup_epochs}")
    print(f"  Clip grad    : {config.clip_grad}")
    print("  Pretrained   : tidak digunakan")
    print("  Early stop   : tidak digunakan")
    print(f"  Output       : {config.output_base}")
    print(f"  Start fold   : {start_fold}")


def _report_config_for(config: SingleStageConfig, class_names: Sequence[str]) -> ReportConfig:
    return ReportConfig(
        output_dir=config.output_base,
        folds_dir=FOLDS_DIR,
        label=config.label,
        num_folds=config.num_folds,
        class_names=class_names,
        header_lines=[
            f"Single-Stage: Epochs={config.epochs} | LR={config.lr}",
            f"Data Loader: SSV2 | Num Frames: {config.num_frames} | "
            f"Num Sample: {config.num_sample} | "
            f"Test Clips: {config.test_num_segment}x{config.test_num_crop}",
        ],
    )


def run(config: SingleStageConfig, start_fold: int = 1) -> None:
    ensure_paths_exist([
        (FOLDS_DIR, "folds_data"),
        (VIDEOMAE_DIR, "VideoMAE"),
    ])

    class_names = load_class_names(FOLDS_DIR)
    num_classes = len(class_names)

    config.output_base.mkdir(parents=True, exist_ok=True)
    _print_run_header(config, num_classes, class_names, start_fold)

    fold_accuracies: List[float] = []
    fold_logs: List[List[dict]] = []

    for fold_idx in range(1, start_fold):
        log_file = config.output_base / f"fold_{fold_idx}" / "log.txt"
        epochs_data = read_log(log_file)
        fold_logs.append(epochs_data)
        best = max((row["val_acc1"] for row in epochs_data), default=0.0)
        fold_accuracies.append(best)
        print(f"[INFO] Fold {fold_idx} sudah selesai. Best Acc@1: {best:.2f}%")

    for fold_idx in range(start_fold, config.num_folds + 1):
        banner(f"FOLD {fold_idx}/{config.num_folds}")
        best_acc = train_fold(config, fold_idx, num_classes)
        fold_accuracies.append(best_acc)

        log_file = config.output_base / f"fold_{fold_idx}" / "log.txt"
        fold_logs.append(read_log(log_file))
        cleanup_checkpoints(config.output_base / f"fold_{fold_idx}")

    banner("SELESAI - RINGKASAN HASIL")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {i} Best Acc@1 : {acc:.2f}%")
    valid = [a for a in fold_accuracies if a > 0]
    if valid:
        print("  " + "-" * 40)
        print(f"  Mean Acc@1       : {sum(valid) / len(valid):.2f}%")

    print("\n[INFO] Membuat laporan...")
    generate_reports(
        config=_report_config_for(config, class_names),
        fold_accuracies=fold_accuracies,
        fold_logs=fold_logs,
    )

    banner("SEMUA SELESAI")
    print(f"  Output: {config.output_base}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-stage scratch training ViT-Small on SIBI"
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(CONFIG, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
