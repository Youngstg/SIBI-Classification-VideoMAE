"""Benchmark four pretrained VideoMAE variants sequentially on SIBI.

Single-stage fine-tuning, 25 epochs each, running through:
    ViT-B K400 -> ViT-B SSV2 -> ViT-S K400 -> ViT-S SSV2

For each model the script writes a full set of evaluation artefacts to
``work_dirs/<model_output_dir>/`` via the shared reporting module.

Run from the project root:
    python train_videomae/all_models_singlestage.py --start_model 1 --start_fold 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT.parent))

from train_videomae.evaluation import ReportConfig, generate_reports
from train_videomae.training_utils import (
    banner,
    cleanup_checkpoints,
    ensure_paths_exist,
    load_class_names,
    read_log,
    run_stage,
)

PROJECT_ROOT = _PACKAGE_ROOT.parent
VIDEOMAE_DIR = PROJECT_ROOT / "VideoMAE"
FOLDS_DIR = PROJECT_ROOT / "folds_data"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
OUTPUT_BASE = PROJECT_ROOT / "work_dirs"


# ---------------------------------------------------------------------------
# Per-model and global configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    model_id: int
    label: str
    data_set: str
    model_name: str
    pretrained_file: str
    output_subdir: str
    sampling_rate: int
    lr: str
    batch_size: int
    num_workers: int


MODELS: Sequence[ModelConfig] = (
    ModelConfig(1, "ViT-Base  | K400", "Kinetics-400", "vit_base_patch16_224",
                "videomae_base_patch16_224_k400.pth", "vitb_k400",
                sampling_rate=4, lr="5e-5", batch_size=8, num_workers=4),
    ModelConfig(2, "ViT-Base  | SSV2", "SSV2", "vit_base_patch16_224",
                "videomae_base_patch16_224_ssv2.pth", "vitb_ssv2",
                sampling_rate=2, lr="5e-5", batch_size=8, num_workers=4),
    ModelConfig(3, "ViT-Small | K400", "Kinetics-400", "vit_small_patch16_224",
                "videomae_small_patch16_224_k400.pth", "vits_k400",
                sampling_rate=4, lr="5e-5", batch_size=12, num_workers=8),
    ModelConfig(4, "ViT-Small | SSV2", "SSV2", "vit_small_patch16_224",
                "videomae_small_patch16_224_ssv2.pth", "vits_ssv2",
                sampling_rate=2, lr="1e-4", batch_size=12, num_workers=8),
)

NUM_FOLDS = 5
EPOCHS = 25
NUM_FRAMES = 16
WEIGHT_DECAY = 0.05
GRAD_ACCUM = 2
WARMUP_EPOCHS = 5


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(
    model_cfg: ModelConfig,
    num_classes: int,
    train_csv: Path,
    val_csv: Path,
    fold_output_dir: Path,
) -> List[str]:
    """Build command for running VideoMAE fine-tuning.
    
    Args:
        model_cfg: Model configuration (architecture, pretrained weights)
        num_classes: Number of output classes
        train_csv: Path to training fold CSV
        val_csv: Path to validation fold CSV
        fold_output_dir: Output directory for this fold
    
    Returns:
        Command line arguments for VideoMAE training script
    """
    pretrained = PRETRAINED_DIR / model_cfg.pretrained_file
    return [
        sys.executable, "run_class_finetuning.py",
        "--model",          model_cfg.model_name,
        "--data_set",       model_cfg.data_set,
        "--data_path",      str(train_csv),
        "--eval_data_path", str(val_csv),
        "--finetune",       str(pretrained),
        "--output_dir",     str(fold_output_dir),
        "--log_dir",        str(fold_output_dir),
        "--nb_classes",     str(num_classes),
        "--batch_size",     str(model_cfg.batch_size),
        "--num_frames",     str(NUM_FRAMES),
        "--sampling_rate",  str(model_cfg.sampling_rate),
        "--num_workers",    str(model_cfg.num_workers),
        "--opt",            "adamw",
        "--lr",             model_cfg.lr,
        "--weight_decay",   str(WEIGHT_DECAY),
        "--epochs",         str(EPOCHS),
        "--save_ckpt_freq", "5",
        "--update_freq",    str(GRAD_ACCUM),
        "--warmup_epochs",  str(WARMUP_EPOCHS),
        "--use_checkpoint",
    ]


# ---------------------------------------------------------------------------
# Per-model training loop
# ---------------------------------------------------------------------------

def _best_acc_from_fold(fold_output_dir: Path) -> tuple[List[dict], float]:
    epochs_data = read_log(fold_output_dir / "log.txt")
    best = max((row["val_acc1"] for row in epochs_data), default=0.0)
    return epochs_data, best


def _resume_completed_folds(
    output_root: Path, start_fold: int,
) -> tuple[List[float], List[List[dict]]]:
    fold_accuracies: List[float] = []
    fold_logs: List[List[dict]] = []
    for fold_idx in range(1, start_fold):
        epochs_data, best = _best_acc_from_fold(output_root / f"fold_{fold_idx}")
        fold_logs.append(epochs_data)
        fold_accuracies.append(best)
        if best > 0:
            print(f"[INFO] Fold {fold_idx} sudah selesai. Best Acc@1: {best:.2f}%")
    return fold_accuracies, fold_logs


def train_model(
    model_cfg: ModelConfig,
    num_classes: int,
    class_names: Sequence[str],
    start_fold: int = 1,
) -> tuple[List[float], List[List[dict]]]:
    """Train all folds for one model variant.
    
    Args:
        model_cfg: Model configuration
        num_classes: Number of classes
        class_names: List of class names
        start_fold: Starting fold (resume capability)
    
    Returns:
        Tuple of (list of best accuracies per fold, list of epoch logs per fold)
    """
    # Resume previously completed folds if restarting
    fold_accuracies, fold_logs = _resume_completed_folds(model_cfg.output_dir, start_fold)""
    output_root = OUTPUT_BASE / model_cfg.output_subdir
    pretrained = PRETRAINED_DIR / model_cfg.pretrained_file

    banner(f"MULAI TRAINING: {model_cfg.label}")
    print(f"  Pretrained : {pretrained}")
    print(f"  Data loader: {model_cfg.data_set}")
    print(f"  Output     : {output_root}")
    print(f"  Start Fold : {start_fold}")
    print(f"  Batch size : {model_cfg.batch_size}")
    print(f"  LR         : {model_cfg.lr}")
    print(f"  Sampling   : {model_cfg.sampling_rate}")

    if not pretrained.exists():
        print(f"[ERROR] File pretrained tidak ditemukan: {pretrained}")
        print(f"[SKIP] Model {model_cfg.label} dilewati.")
        return [], []

    fold_accuracies, fold_logs = _resume_completed_folds(output_root, start_fold)

    for fold_idx in range(start_fold, NUM_FOLDS + 1):
        banner(f"{model_cfg.label}  |  FOLD {fold_idx}/{NUM_FOLDS}", char="-")

        train_csv = FOLDS_DIR / f"train_fold_{fold_idx}.csv"
        val_csv = FOLDS_DIR / f"val_fold_{fold_idx}.csv"
        fold_output_dir = output_root / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        if not train_csv.exists() or not val_csv.exists():
            print(f"[SKIP] CSV fold {fold_idx} tidak ditemukan.")
            fold_logs.append([])
            fold_accuracies.append(0.0)
            continue

        cmd = build_command(model_cfg, num_classes, train_csv, val_csv, fold_output_dir)
        ok, _ = run_stage(
            cmd,
            stage_name=f"{model_cfg.label} Fold{fold_idx}",
            epochs_total=EPOCHS,
            cwd=str(VIDEOMAE_DIR),
            early_stop_patience=None,
        )

        epochs_data, best_acc = _best_acc_from_fold(fold_output_dir)
        fold_logs.append(epochs_data)
        fold_accuracies.append(best_acc)

        if best_acc > 0:
            print(f"\n[RESULT] {model_cfg.label} | Fold {fold_idx} Best Acc@1: {best_acc:.2f}%")
        elif not ok:
            print(f"[WARN] Fold {fold_idx} gagal atau log kosong, lanjut fold berikutnya.")

        cleanup_checkpoints(fold_output_dir)

    # Per-model summary
    banner(f"SELESAI: {model_cfg.label}")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {i} Best Acc@1 : {acc:.2f}%")
    valid = [a for a in fold_accuracies if a > 0]
    if valid:
        print("  " + "-" * 35)
        print(f"  Mean Acc@1 : {sum(valid) / len(valid):.2f}%")

    print(f"\n[INFO] Membuat laporan untuk {model_cfg.label}...")
    generate_reports(
        config=_report_config_for(model_cfg, class_names),
        fold_accuracies=fold_accuracies,
        fold_logs=fold_logs,
        detailed_per_fold=True,
    )
    return fold_accuracies, fold_logs


def _report_config_for(model_cfg: ModelConfig, class_names: Sequence[str]) -> ReportConfig:
    return ReportConfig(
        output_dir=OUTPUT_BASE / model_cfg.output_subdir,
        folds_dir=FOLDS_DIR,
        label=model_cfg.label,
        num_folds=NUM_FOLDS,
        class_names=class_names,
        header_lines=[
            f"Model     : {model_cfg.label}",
            f"DataLoader: {model_cfg.data_set}",
            f"Pretrained: {model_cfg.pretrained_file}",
        ],
        # This benchmark covers Kinetics-400 too, so the SSV2 baseline would
        # be misleading on some panels.
        show_ssv2_baseline=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 4 VideoMAE models sequentially: "
                    "ViT-B K400 -> ViT-B SSV2 -> ViT-S K400 -> ViT-S SSV2"
    )
    parser.add_argument(
        "--start_model", type=int, default=1, choices=[1, 2, 3, 4],
        help="Mulai dari model ke-N (1=ViT-B K400, 2=ViT-B SSV2, 3=ViT-S K400, 4=ViT-S SSV2)",
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N pada --start_model",
    )
    args = parser.parse_args()

    ensure_paths_exist([
        (FOLDS_DIR, "folds_data"),
        (VIDEOMAE_DIR, "VideoMAE"),
    ])

    class_names = load_class_names(FOLDS_DIR)
    num_classes = len(class_names)

    banner("TRAINING SEMUA MODEL - VideoMAE")
    print("  Urutan  : ViT-B K400 -> ViT-B SSV2 -> ViT-S K400 -> ViT-S SSV2")
    print(f"  Kelas   : {num_classes} ({', '.join(class_names)})")
    print(f"  Folds   : {NUM_FOLDS}  |  Epochs : {EPOCHS}  |  Frames : {NUM_FRAMES}")
    print(f"  Output  : {OUTPUT_BASE}/")
    print(f"  Start   : Model {args.start_model}, Fold {args.start_fold}")

    all_results: Dict[str, float] = {}

    for model_cfg in MODELS:
        if model_cfg.model_id < args.start_model:
            print(f"\n[SKIP] Model {model_cfg.model_id}: {model_cfg.label} (sudah selesai)")
            continue

        start_fold = args.start_fold if model_cfg.model_id == args.start_model else 1
        accs, _ = train_model(model_cfg, num_classes, class_names, start_fold=start_fold)
        valid = [a for a in accs if a > 0]
        all_results[model_cfg.label] = sum(valid) / len(valid) if valid else 0.0

    banner("RINGKASAN AKHIR SEMUA MODEL")
    for label, mean_acc in all_results.items():
        print(f"  {label:<25} Mean Acc@1 : {mean_acc:.2f}%")
    if all_results:
        best_model = max(all_results, key=all_results.get)
        print(f"\n  Model terbaik : {best_model}  ({all_results[best_model]:.2f}%)")

    print("\n  Laporan tersimpan di:")
    for model_cfg in MODELS:
        if model_cfg.model_id >= args.start_model:
            print(f"  -> {OUTPUT_BASE / model_cfg.output_subdir}")
    print()


if __name__ == "__main__":
    main()
