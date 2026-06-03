"""Two-stage VideoMAE training on WLASL dataset (following paper methodology).

Follows official WLASL paper (Li et al., WACV 2020) methodology:
  - Training: Combines train + val splits (trainval_wlaslN.csv)
  - Evaluation: Test set only (test_wlaslN.csv)
  - No early stopping (no separate val set for monitoring)

This enables direct comparison with I3D results from the original paper.

Preparation steps:
    1. Download WLASL videos → wlasl_data/videos/{video_id}.mp4
    2. python preprocess_wlasl.py --subset 300
    3. python train_videomae/twostage_wlasl.py --subset 300 --start_fold 1

Paper baseline (I3D RGB accuracy):
    - WLASL-100:  65.89%
    - WLASL-300:  56.14%
    - WLASL-1000: 47.33%
    - WLASL-2000: 32.48%
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT.parent))

from train_videomae.evaluation import ReportConfig, generate_reports
from train_videomae.training_utils import (
    banner,
    best_acc_from_log,
    cleanup_checkpoints,
    ensure_paths_exist,
    read_log,
    run_stage,
)

PROJECT_ROOT   = _PACKAGE_ROOT.parent
VIDEOMAE_DIR   = PROJECT_ROOT / "VideoMAE"
WLASL_DATA_DIR = PROJECT_ROOT / "wlasl_data"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

# Angka pembanding dari paper WLASL (Li et al., WACV 2020), I3D RGB.
I3D_PAPER_ACCURACY = {
    100:  65.89,
    300:  56.14,
    1000: 47.33,
    2000: 32.48,
}


# ---------------------------------------------------------------------------
# Konfigurasi eksperimen
# ---------------------------------------------------------------------------

@dataclass
class WLASLConfig:
    """Configuration for two-stage WLASL training (paper methodology).
    
    Key difference: Uses trainval set (train + val combined) for training,
    test set only for evaluation (no separate validation set).
    This matches the official WLASL paper methodology.
    
    No early stopping since there's no held-out validation set.
    """

    subset: int = 300

    label: str = "VideoMAE ViT-Small | SSV2 → WLASL | 2×RTX4080S"
    output_base: Path = field(default_factory=lambda: PROJECT_ROOT / "work_dirs" / "wlasl300_twostage")
    model_name: str = "vit_small_patch16_224"
    pretrained_file: Optional[str] = "videomae_small_patch16_224_ssv2.pth"

    num_frames: int = 16
    sampling_rate: int = 1
    batch_size: int = 8    # per GPU; 2×8 = 16/step × grad_accum 2 = effective 32
    num_workers: int = 8   # per GPU
    grad_accum: int = 2
    weight_decay: float = 0.05

    # Tidak ada early stopping karena train+val digabung (ikuti paper).
    early_stop_patience: Optional[int] = None

    # Stage 1: head saja — LR besar, mixup off, tanpa warmup.
    stage1_epochs: int = 5
    stage1_lr: str = "1e-3"
    stage1_warmup: int = 0
    stage1_clip_grad: str = "1.0"
    stage1_mixup_off: bool = True

    # Stage 2: full fine-tune dari checkpoint terbaik Stage 1.
    stage2_epochs: int = 50
    stage2_lr: str = "1e-5"
    stage2_warmup: int = 5
    stage2_clip_grad: str = "1.0"

    def pretrained_path(self) -> Optional[Path]:
        if self.pretrained_file is None:
            return None
        return PRETRAINED_DIR / self.pretrained_file

    def ssv2_dir(self) -> Path:
        """Folder berisi train.csv/val.csv dengan absolute paths.

        datasets.py pada VideoMAE untuk --data_set SSV2 mengharapkan
        sebuah direktori yang berisi train.csv dan val.csv, bukan
        path langsung ke file CSV.  Path video di dalamnya harus absolut
        karena datasets.py menyetel data_path='/'.
        """
        return WLASL_DATA_DIR / "ssv2_splits" / str(self.subset)

    def label_map_path(self) -> Path:
        return WLASL_DATA_DIR / f"label_map_wlasl{self.subset}.json"


DEFAULT_CONFIG = WLASLConfig()


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _common_finetune_args(
    config: WLASLConfig,
    num_classes: int,
    stage_output_dir: Path,
) -> List[str]:
    # --data_path harus direktori (bukan file CSV) untuk --data_set SSV2.
    # datasets.py akan append /train.csv dan /val.csv ke path ini.
    # Path video di dalam CSV harus absolut (datasets.py set data_path='/').
    import shutil
    torchrun = shutil.which("torchrun") or str(Path(sys.executable).parent / "torchrun")
    return [
        torchrun, "--nproc_per_node=2",
        "run_class_finetuning.py",
        "--model",      config.model_name,
        "--data_set",   "SSV2",
        "--data_path",      str(config.ssv2_dir()),
        "--eval_data_path", str(config.ssv2_dir()),
        "--dist_eval",
        "--output_dir", str(stage_output_dir),
        "--log_dir",    str(stage_output_dir),
        "--nb_classes", str(num_classes),
        "--batch_size", str(config.batch_size),
        "--num_frames", str(config.num_frames),
        "--sampling_rate", str(config.sampling_rate),
        "--num_workers",   str(config.num_workers),
        "--opt",        "adamw",
        "--weight_decay", str(config.weight_decay),
        "--update_freq",  str(config.grad_accum),
    ]


def build_stage1_command(
    config: WLASLConfig,
    num_classes: int,
    run_output_dir: Path,
) -> List[str]:
    """Stage 1: load pretrained, latih head saja, evaluasi di test set."""
    stage_dir = run_output_dir / "stage1"
    cmd = _common_finetune_args(config, num_classes, stage_dir)

    pretrained = config.pretrained_path()
    if pretrained is not None:
        cmd += ["--finetune", str(pretrained), "--model_key", "model"]

    cmd += [
        "--lr",             config.stage1_lr,
        "--epochs",         str(config.stage1_epochs),
        "--save_ckpt_freq", str(config.stage1_epochs),
        "--warmup_epochs",  str(config.stage1_warmup),
        "--clip_grad",      config.stage1_clip_grad,
    ]

    if config.stage1_mixup_off:
        cmd += ["--mixup", "0", "--cutmix", "0", "--reprob", "0"]

    return cmd


def build_stage2_command(
    config: WLASLConfig,
    num_classes: int,
    run_output_dir: Path,
    stage1_checkpoint: Path,
) -> List[str]:
    """Stage 2: full fine-tune dari checkpoint terbaik Stage 1."""
    stage_dir = run_output_dir / "stage2"
    cmd = _common_finetune_args(config, num_classes, stage_dir)
    cmd += [
        "--finetune",       str(stage1_checkpoint),
        "--lr",             config.stage2_lr,
        "--epochs",         str(config.stage2_epochs),
        "--save_ckpt_freq", "5",
        "--warmup_epochs",  str(config.stage2_warmup),
        "--clip_grad",      config.stage2_clip_grad,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def find_stage1_checkpoint(run_output_dir: Path, stage1_epochs: int) -> Optional[Path]:
    stage_dir = run_output_dir / "stage1"
    for candidate in [
        stage_dir / "checkpoint-best.pth",
        stage_dir / f"checkpoint-{stage1_epochs - 1}.pth",
    ]:
        if candidate.exists():
            return candidate
    any_ckpts = sorted(stage_dir.glob("checkpoint-*.pth"))
    return any_ckpts[-1] if any_ckpts else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: WLASLConfig, num_classes: int) -> float:
    """Jalankan Stage 1 + Stage 2. Return best test Acc@1."""
    run_output_dir = config.output_base / "fold_1"
    (run_output_dir / "stage1").mkdir(parents=True, exist_ok=True)
    (run_output_dir / "stage2").mkdir(parents=True, exist_ok=True)

    ssv2_dir = config.ssv2_dir()
    train_csv = ssv2_dir / "train.csv"
    val_csv   = ssv2_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print(f"[ERROR] SSV2 splits tidak ditemukan di: {ssv2_dir}")
        print(f"        Jalankan dulu:")
        print(f"        python preprocess_wlasl.py --subset {config.subset}")
        return 0.0

    # -------- Stage 1 --------
    banner(f"WLASL{config.subset} - STAGE 1: Freeze Backbone (Head Only)", char="-")
    print(f"[INFO] Data dir : {ssv2_dir}")
    print(f"[INFO] Train    : {_count_lines(train_csv)} sampel (train.csv)")
    print(f"[INFO] Eval     : {_count_lines(val_csv)} sampel (val.csv)\n")

    cmd_s1 = build_stage1_command(config, num_classes, run_output_dir)
    ok_s1, acc_s1 = run_stage(
        cmd_s1,
        stage_name=f"Stage1_WLASL{config.subset}",
        epochs_total=config.stage1_epochs,
        cwd=str(VIDEOMAE_DIR),
        early_stop_patience=None,
    )
    if not ok_s1:
        return 0.0

    stage1_ckpt = find_stage1_checkpoint(run_output_dir, config.stage1_epochs)
    if stage1_ckpt is None:
        print(f"[ERROR] Tidak ada checkpoint Stage 1 ditemukan.")
        return 0.0

    print(f"\n[INFO] Stage 1 selesai. Test Acc@1: {acc_s1:.2f}%")
    print(f"[INFO] Checkpoint Stage 1: {stage1_ckpt}")

    # -------- Stage 2 --------
    banner(f"WLASL{config.subset} - STAGE 2: Unfreeze All (Full Fine-tune)", char="-")
    print(f"[INFO] Data dir : {ssv2_dir}")
    print(f"[INFO] Train    : {_count_lines(train_csv)} sampel (train.csv)")
    print(f"[INFO] Eval     : {_count_lines(val_csv)} sampel (val.csv)\n")

    cmd_s2 = build_stage2_command(config, num_classes, run_output_dir, stage1_ckpt)
    ok_s2, acc_s2 = run_stage(
        cmd_s2,
        stage_name=f"Stage2_WLASL{config.subset}",
        epochs_total=config.stage2_epochs,
        cwd=str(VIDEOMAE_DIR),
        early_stop_patience=None,   # tidak ada early stop
    )
    if not ok_s2:
        return 0.0

    best_acc = best_acc_from_log(run_output_dir / "stage2" / "log.txt") or acc_s2
    print(f"\n[RESULT] WLASL{config.subset} | Stage 2 Best Test Acc@1: {best_acc:.2f}%")

    cleanup_checkpoints(run_output_dir / "stage1")
    cleanup_checkpoints(run_output_dir / "stage2")
    return best_acc


def _count_lines(csv_path: Path) -> int:
    """Hitung jumlah baris non-kosong dalam CSV."""
    try:
        with open(csv_path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_num_classes(config: WLASLConfig) -> tuple[int, list[str]]:
    import json
    label_map_path = config.label_map_path()
    if not label_map_path.exists():
        print(f"[ERROR] label_map tidak ditemukan: {label_map_path}")
        print(f"        Jalankan: python preprocess_wlasl.py --subset {config.subset}")
        sys.exit(1)
    with open(label_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data["labels"]
    if isinstance(labels, dict):
        class_names = [n for n, _ in sorted(labels.items(), key=lambda x: x[1])]
    else:
        class_names = list(labels)
    return len(class_names), class_names


def print_run_header(config: WLASLConfig, num_classes: int) -> None:
    banner(f"TWO-STAGE TRAINING  |  {config.label}  |  WLASL{config.subset}")
    print(f"  Dataset      : WLASL{config.subset} ({num_classes} kelas)")
    print(f"  Metodologi   : paper-faithful — train+val → test (tanpa early stop)")
    if config.pretrained_file:
        print(f"  Pretrained   : {config.pretrained_file}")
    else:
        print("  Pretrained   : (scratch)")
    print(f"  Num frames   : {config.num_frames}")
    print(f"  Batch size   : {config.batch_size} × grad_accum {config.grad_accum} "
          f"= effective {config.batch_size * config.grad_accum}")

    print("\n  STAGE 1 (Head Only):")
    print(f"    Epochs     : {config.stage1_epochs}")
    print(f"    LR         : {config.stage1_lr}")
    print(f"    Mixup      : {'DIMATIKAN' if config.stage1_mixup_off else 'AKTIF'}")
    print(f"    Early stop : TIDAK (train+val digabung, ikuti paper)")

    print("\n  STAGE 2 (Full Fine-tune):")
    print(f"    Epochs     : {config.stage2_epochs}")
    print(f"    LR         : {config.stage2_lr}")
    print(f"    Warmup     : {config.stage2_warmup} epoch")
    print(f"    Early stop : TIDAK (train+val digabung, ikuti paper)")

    i3d_ref = I3D_PAPER_ACCURACY.get(config.subset)
    if i3d_ref:
        print(f"\n  Target       : > I3D {i3d_ref:.2f}%  (WLASL{config.subset}, paper)")

    print(f"\n  Output       : {config.output_base}")


def _report_config(config: WLASLConfig, class_names: Sequence[str]) -> ReportConfig:
    i3d_ref = I3D_PAPER_ACCURACY.get(config.subset)
    return ReportConfig(
        output_dir=config.output_base,
        folds_dir=WLASL_DATA_DIR,   # berisi val_fold_1.csv → alias test set
        label=config.label,
        num_folds=1,
        class_names=class_names,
        log_subdir="stage2",
        pred_subdir="stage2",
        stage_split_epoch=config.stage1_epochs,
        show_ssv2_baseline=False,
        header_lines=[
            f"WLASL{config.subset} | paper-faithful split (trainval→test, no early stop)",
            f"Two-Stage: Stage1={config.stage1_epochs}ep LR={config.stage1_lr} | "
            f"Stage2={config.stage2_epochs}ep LR={config.stage2_lr} | "
            f"I3D baseline: {i3d_ref:.2f}%" if i3d_ref else "",
            f"Model: {config.model_name} | Pretrained: {config.pretrained_file} | "
            f"Frames: {config.num_frames} | Batch: {config.batch_size}×{config.grad_accum}",
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: WLASLConfig) -> None:
    ensure_paths_exist([
        (WLASL_DATA_DIR, "wlasl_data"),
        (VIDEOMAE_DIR,   "VideoMAE"),
    ])

    ssv2_dir = config.ssv2_dir()
    if not ssv2_dir.exists():
        print(f"[ERROR] SSV2 splits belum dibuat: {ssv2_dir}")
        print(f"        Jalankan: python preprocess_wlasl.py --subset {config.subset}")
        sys.exit(1)

    pretrained = config.pretrained_path()
    if pretrained is not None and not pretrained.exists():
        print(f"[ERROR] Pretrained tidak ditemukan: {pretrained}")
        sys.exit(1)

    num_classes, class_names = _load_num_classes(config)
    config.output_base.mkdir(parents=True, exist_ok=True)

    print_run_header(config, num_classes)

    best_acc = train(config, num_classes)

    i3d_ref = I3D_PAPER_ACCURACY.get(config.subset, 0.0)
    diff    = best_acc - i3d_ref

    banner("SELESAI - HASIL AKHIR")
    print(f"  VideoMAE WLASL{config.subset} Test Acc@1 : {best_acc:.2f}%")
    print(f"  I3D (paper) WLASL{config.subset}         : {i3d_ref:.2f}%")
    print(f"  Selisih                           : {diff:+.2f}%  "
          f"({'LEBIH BAIK' if diff > 0 else 'lebih rendah'})")
    print(f"  Output                            : {config.output_base}")

    print("\n[INFO] Membuat laporan...")
    log_data = read_log(config.output_base / "fold_1" / "stage2" / "log.txt")
    generate_reports(
        config=_report_config(config, class_names),
        fold_accuracies=[best_acc],
        fold_logs=[log_data],
    )

    banner("SEMUA SELESAI")
    print(f"  Output: {config.output_base}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage VideoMAE pada WLASL, metodologi paper-faithful"
    )
    parser.add_argument(
        "--subset", type=int, default=300, choices=[100, 300, 1000, 2000],
        help="Subset WLASL (default: 300)",
    )
    parser.add_argument(
        "--stage2_epochs", type=int, default=None,
        help="Override jumlah epoch Stage 2 (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size (default: 12)",
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Latih dari scratch tanpa pretrained weights",
    )
    args = parser.parse_args()

    config = replace(
        DEFAULT_CONFIG,
        subset=args.subset,
        output_base=PROJECT_ROOT / "work_dirs" / f"wlasl{args.subset}_twostage",
        label=f"VideoMAE ViT-Small | SSV2 → WLASL{args.subset}",
    )

    if args.stage2_epochs is not None:
        config = replace(config, stage2_epochs=args.stage2_epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.no_pretrained:
        config = replace(config, pretrained_file=None)

    run(config)


if __name__ == "__main__":
    main()
