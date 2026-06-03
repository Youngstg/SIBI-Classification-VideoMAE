"""Two-stage fine-tuning of VideoMAE ViT-Small on SIBI from SSV2 weights.

Stage 1: freeze the backbone, train the classification head only. The head is
    initialised from scratch so a higher LR + no warmup is appropriate.
Stage 2: unfreeze every layer and fine-tune end-to-end with a small LR, using
    the best Stage 1 checkpoint as the starting point.

Run from the project root:
    python train_videomae/twostage_pretrained.py --start_fold 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence

# Allow ``python train_videomae/twostage_pretrained.py`` style execution.
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
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class TwoStageConfig:
    """Configuration for two-stage VideoMAE training.
    
    Stage 1: Train classification head only (frozen backbone, high LR, no warmup)
    Stage 2: Fine-tune full model (unfrozen backbone, low LR, gradual warmup)
    
    Wrapper scripts create instances via dataclasses.replace() for experiments.
    """

    label: str = "ViT-Small | SSV2 | Two-Stage"
    output_base: Path = field(default_factory=lambda: PROJECT_ROOT / "work_dirs" / "vits_ssv2_twostage")
    model_name: str = "vit_small_patch16_224"
    pretrained_file: Optional[str] = "videomae_small_patch16_224_ssv2.pth"

    num_folds: int = 5
    num_frames: int = 16
    sampling_rate: int = 1
    batch_size: int = 12
    num_workers: int = 12
    grad_accum: int = 2
    weight_decay: float = 0.05
    early_stop_patience: Optional[int] = 5

    # Stage 1: head-only or scratch warm-up.
    stage1_epochs: int = 5
    stage1_lr: str = "1e-3"
    stage1_warmup: int = 0
    stage1_clip_grad: str = "1.0"
    stage1_mixup_off: bool = True

    # Stage 2: full fine-tune from the best Stage 1 checkpoint.
    stage2_epochs: int = 50
    stage2_lr: str = "1e-5"
    stage2_warmup: int = 5
    stage2_clip_grad: str = "1.0"
    stage2_mixup_off: bool = False

    def pretrained_path(self) -> Optional[Path]:
        if self.pretrained_file is None:
            return None
        return PRETRAINED_DIR / self.pretrained_file


DEFAULT_CONFIG = TwoStageConfig()


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _common_finetune_args(
    config: TwoStageConfig,
    num_classes: int,
    train_csv: Path,
    val_csv: Path,
    stage_output_dir: Path,
) -> List[str]:
    return [
        sys.executable, "run_class_finetuning.py",
        "--model",          config.model_name,
        "--data_set",       "SSV2",
        "--data_path",      str(train_csv),
        "--eval_data_path", str(val_csv),
        "--output_dir",     str(stage_output_dir),
        "--log_dir",        str(stage_output_dir),
        "--nb_classes",     str(num_classes),
        "--batch_size",     str(config.batch_size),
        "--num_frames",     str(config.num_frames),
        "--sampling_rate",  str(config.sampling_rate),
        "--num_workers",    str(config.num_workers),
        "--opt",            "adamw",
        "--weight_decay",   str(config.weight_decay),
        "--update_freq",    str(config.grad_accum),
    ]


def build_stage1_command(
    config: TwoStageConfig,
    num_classes: int,
    train_csv: Path,
    val_csv: Path,
    fold_output_dir: Path,
) -> List[str]:
    """Build Stage 1 command: train head only with frozen backbone.
    
    Stage 1 characteristics:
    - Loads pretrained VideoMAE weights
    - Freezes backbone layers
    - Trains only classification head (random init)
    - High learning rate (no pretrained backbone decay needed)
    - No warmup (head is learning from scratch)
    - Mixup/cutmix disabled for clean label learning
    """
    stage_dir = fold_output_dir / "stage1"
    cmd = _common_finetune_args(config, num_classes, train_csv, val_csv, stage_dir)

    pretrained = config.pretrained_path()
    if pretrained is not None:
        cmd += ["--finetune", str(pretrained), "--model_key", "module|model"]

    cmd += [
        "--lr",             config.stage1_lr,
        "--epochs",         str(config.stage1_epochs),
        # Save exactly one checkpoint at the end of Stage 1.
        "--save_ckpt_freq", str(config.stage1_epochs),
        "--warmup_epochs",  str(config.stage1_warmup),
        "--clip_grad",      config.stage1_clip_grad,
        "--use_checkpoint",
    ]

    if config.stage1_mixup_off:
        # Learn the true labels on a clean head first.
        cmd += ["--mixup", "0", "--cutmix", "0", "--reprob", "0"]

    return cmd


def build_stage2_command(
    config: TwoStageConfig,
    num_classes: int,
    train_csv: Path,
    val_csv: Path,
    fold_output_dir: Path,
    stage1_checkpoint: Path,
) -> List[str]:
    """Build Stage 2 command: fine-tune full model from Stage 1 best checkpoint.
    
    Stage 2 characteristics:
    - Loads checkpoint from Stage 1 (already tuned head + backbone)
    - Unfreezes all layers (full fine-tuning)
    - Uses lower learning rate (to avoid destroying learned features)
    - Includes warmup for stable learning
    - Can optionally disable mixup/cutmix for label-sensitive tasks
    """
    stage_dir = fold_output_dir / "stage2"
    cmd = _common_finetune_args(config, num_classes, train_csv, val_csv, stage_dir)
    cmd += [
        "--finetune",       str(stage1_checkpoint),
        "--lr",             config.stage2_lr,
        "--epochs",         str(config.stage2_epochs),
        "--save_ckpt_freq", "5",
        "--warmup_epochs",  str(config.stage2_warmup),
        "--clip_grad",      config.stage2_clip_grad,
        "--use_checkpoint",
    ]

    if config.stage2_mixup_off:
        # Sign-language labels are sensitive to hand shape/orientation; avoid
        # label-mixing regularizers when explicitly requested.
        cmd += ["--mixup", "0", "--cutmix", "0", "--reprob", "0"]

    return cmd


# ---------------------------------------------------------------------------
# Fold orchestration
# ---------------------------------------------------------------------------

def find_stage1_checkpoint(fold_output_dir: Path, stage1_epochs: int) -> Optional[Path]:
    """Locate the best Stage 1 checkpoint (best.pth > last > any available).
    
    Returns checkpoint path for Stage 2 initialization, or None if not found.
    """
    stage_dir = fold_output_dir / "stage1"
    # Prefer best checkpoint if available
    best = stage_dir / "checkpoint-best.pth"
    if best.exists():
        return best

    last = stage_dir / f"checkpoint-{stage1_epochs - 1}.pth"
    if last.exists():
        return last

    any_ckpts = sorted(stage_dir.glob("checkpoint-*.pth"))
    return any_ckpts[-1] if any_ckpts else None


def train_fold(config: TwoStageConfig, fold_idx: int, num_classes: int) -> float:
    """Run both stages for one fold and return the best Stage 2 val_acc1."""
    train_csv = FOLDS_DIR / f"train_fold_{fold_idx}.csv"
    val_csv = FOLDS_DIR / f"val_fold_{fold_idx}.csv"
    fold_output_dir = config.output_base / f"fold_{fold_idx}"

    (fold_output_dir / "stage1").mkdir(parents=True, exist_ok=True)
    (fold_output_dir / "stage2").mkdir(parents=True, exist_ok=True)

    if not train_csv.exists() or not val_csv.exists():
        print(f"[SKIP] CSV fold {fold_idx} tidak ditemukan.")
        return 0.0

    # ------- Stage 1 -------
    banner(f"FOLD {fold_idx} - STAGE 1: Freeze Backbone (Head Only)", char="-")
    cmd_s1 = build_stage1_command(config, num_classes, train_csv, val_csv, fold_output_dir)
    ok_s1, acc_s1 = run_stage(
        cmd_s1,
        stage_name=f"Stage1 Fold{fold_idx}",
        epochs_total=config.stage1_epochs,
        cwd=str(VIDEOMAE_DIR),
        early_stop_patience=config.early_stop_patience,
    )
    if not ok_s1:
        return 0.0

    stage1_ckpt = find_stage1_checkpoint(fold_output_dir, config.stage1_epochs)
    if stage1_ckpt is None:
        print(f"[ERROR] Tidak ada checkpoint Stage 1 di fold {fold_idx}")
        return 0.0

    print(f"\n[INFO] Stage 1 selesai. Best Acc@1: {acc_s1:.2f}%")
    print(f"[INFO] Checkpoint Stage 1: {stage1_ckpt}")

    # ------- Stage 2 -------
    banner(f"FOLD {fold_idx} - STAGE 2: Unfreeze All (Full Fine-tune)", char="-")
    cmd_s2 = build_stage2_command(
        config, num_classes, train_csv, val_csv, fold_output_dir, stage1_ckpt,
    )
    ok_s2, acc_s2 = run_stage(
        cmd_s2,
        stage_name=f"Stage2 Fold{fold_idx}",
        epochs_total=config.stage2_epochs,
        cwd=str(VIDEOMAE_DIR),
        early_stop_patience=config.early_stop_patience,
    )
    if not ok_s2:
        return 0.0

    best_acc = best_acc_from_log(fold_output_dir / "stage2" / "log.txt") or acc_s2
    print(f"\n[RESULT] Fold {fold_idx} | Stage2 Best Acc@1: {best_acc:.2f}%")

    cleanup_checkpoints(fold_output_dir / "stage1")
    cleanup_checkpoints(fold_output_dir / "stage2")
    return best_acc


# ---------------------------------------------------------------------------
# Run-level helpers
# ---------------------------------------------------------------------------

def print_run_header(config: TwoStageConfig, num_classes: int, class_names: Sequence[str], start_fold: int) -> None:
    banner(f"TWO-STAGE TRAINING - {config.label}")
    print(f"  Kelas        : {num_classes} ({', '.join(class_names)})")
    if config.pretrained_file is not None:
        print(f"  Pretrained   : {config.pretrained_file}")
    else:
        print("  Pretrained   : (tidak digunakan - scratch)")
    print("  Data loader  : SSV2 (segment-based temporal sampling)")
    print(f"  Num frames   : {config.num_frames}")
    print(f"  Sampling arg : {config.sampling_rate} (untuk kompatibilitas)")
    print(f"  Num workers  : {config.num_workers}")
    print(f"  Batch size   : {config.batch_size} (effective: {config.batch_size * config.grad_accum})")

    print("\n  STAGE 1:")
    print(f"    Epochs     : {config.stage1_epochs}")
    print(f"    LR         : {config.stage1_lr}")
    print(f"    Mixup      : {'DIMATIKAN' if config.stage1_mixup_off else 'AKTIF'}")
    print(f"    Early stop : {config.early_stop_patience} epoch tanpa peningkatan")

    print("\n  STAGE 2:")
    print(f"    Epochs     : {config.stage2_epochs}")
    print(f"    LR         : {config.stage2_lr}")
    print(f"    Mixup      : {'DIMATIKAN' if config.stage2_mixup_off else 'AKTIF'}")
    print(f"    Warmup     : {config.stage2_warmup} epoch")
    print(f"    Clip grad  : {config.stage2_clip_grad}")
    print(f"    Early stop : {config.early_stop_patience} epoch tanpa peningkatan")

    print(f"\n  Output       : {config.output_base}")
    print(f"  Start fold   : {start_fold}")


def _resume_prior_folds(config: TwoStageConfig, start_fold: int) -> tuple[List[float], List[List[dict]]]:
    """Load best-acc and per-epoch data for folds that were already completed."""
    fold_accuracies: List[float] = []
    fold_logs: List[List[dict]] = []
    for fold_idx in range(1, start_fold):
        log_file = config.output_base / f"fold_{fold_idx}" / "stage2" / "log.txt"
        epochs_data = read_log(log_file)
        fold_logs.append(epochs_data)
        best = max((row["val_acc1"] for row in epochs_data), default=0.0)
        fold_accuracies.append(best)
        print(f"[INFO] Fold {fold_idx} sudah selesai. Best Acc@1: {best:.2f}%")
    return fold_accuracies, fold_logs


def _print_summary(fold_accuracies: Sequence[float]) -> None:
    banner("SELESAI - RINGKASAN HASIL")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {i} Best Acc@1 : {acc:.2f}%")
    valid = [a for a in fold_accuracies if a > 0]
    if valid:
        mean_acc = sum(valid) / len(valid)
        print("  " + "-" * 40)
        print(f"  Mean Acc@1       : {mean_acc:.2f}%")


def _report_config_for(config: TwoStageConfig, class_names: Sequence[str]) -> ReportConfig:
    return ReportConfig(
        output_dir=config.output_base,
        folds_dir=FOLDS_DIR,
        label=config.label,
        num_folds=config.num_folds,
        class_names=class_names,
        log_subdir="stage2",
        pred_subdir="stage2",
        stage_split_epoch=config.stage1_epochs,
        header_lines=[
            f"Two-Stage: Stage1={config.stage1_epochs}ep LR={config.stage1_lr} "
            f"| Stage2={config.stage2_epochs}ep LR={config.stage2_lr}",
            f"Data Loader: SSV2 | Num Frames: {config.num_frames} "
            f"| Sampling Rate Arg: {config.sampling_rate} | Num Workers: {config.num_workers} "
            f"| Stage2 Mixup: {'off' if config.stage2_mixup_off else 'on'}",
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: TwoStageConfig, start_fold: int = 1) -> None:
    """Run the full two-stage experiment according to ``config``."""
    ensure_paths_exist([
        (FOLDS_DIR, "folds_data"),
        (VIDEOMAE_DIR, "VideoMAE"),
    ])

    pretrained = config.pretrained_path()
    if pretrained is not None and not pretrained.exists():
        print(f"[ERROR] Pretrained tidak ditemukan: {pretrained}")
        sys.exit(1)

    class_names = load_class_names(FOLDS_DIR)
    num_classes = len(class_names)

    config.output_base.mkdir(parents=True, exist_ok=True)
    print_run_header(config, num_classes, class_names, start_fold)

    fold_accuracies, fold_logs = _resume_prior_folds(config, start_fold)

    for fold_idx in range(start_fold, config.num_folds + 1):
        banner(f"FOLD {fold_idx}/{config.num_folds}")

        best_acc = train_fold(config, fold_idx, num_classes)
        fold_accuracies.append(best_acc)

        log_file = config.output_base / f"fold_{fold_idx}" / "stage2" / "log.txt"
        fold_logs.append(read_log(log_file))

        cleanup_checkpoints(config.output_base / f"fold_{fold_idx}" / "stage2")

    _print_summary(fold_accuracies)

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
        description="Two-stage fine-tuning ViT-Small on SIBI (SSV2 pretrained)"
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(DEFAULT_CONFIG, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
