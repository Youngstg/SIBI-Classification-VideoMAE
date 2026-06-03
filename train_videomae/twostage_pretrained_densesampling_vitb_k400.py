"""Two-stage ViT-Base training with Kinetics-400 pretraining and dense sampling.

Experiment configuration:
  - Backbone: ViT-Base (larger than ViT-Small)
  - Pretrained: Kinetics-400 checkpoint
  - Sampling: Kinetics dense temporal (sampling_rate=4, every 4th frame)
  - Total epochs: 55 (5 + 50)
  - Early stopping: 5 epochs patience
  - Stage 2: Higher LR (5e-5) + no mixup/cutmix

Run from project root:
    python train_videomae/twostage_pretrained_densesampling_vitb_k400.py --start_fold 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from train_videomae.twostage_pretrained_densesampling import DEFAULT_CONFIG, run

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_VITB_K400 = replace(
    DEFAULT_CONFIG,
    label="ViT-Base | K400 pretrained | Kinetics loader | Two-Stage",
    output_base=PROJECT_ROOT / "work_dirs_VideoMAE_twostage" / "vitb_k400_dense_twostage",
    model_name="vit_base_patch16_224",
    pretrained_file="videomae_base_patch16_224_k400.pth",
    sampling_rate=4,
    batch_size=8,
    num_workers=8,
    stage2_lr="5e-5",
    stage2_mixup_off=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage VideoMAE ViT-Base on SIBI (K400 pretrained, dense temporal sampling)"
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(CONFIG_VITB_K400, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
