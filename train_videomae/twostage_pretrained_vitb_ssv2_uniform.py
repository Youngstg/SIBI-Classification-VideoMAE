"""Two-stage ViT-Base training with SSV2 pretraining and uniform sampling.

Experiment configuration:
  - Backbone: ViT-Base (larger than default ViT-Small)
  - Pretrained: SSV2 checkpoint
  - Sampling: SSV2 uniform/segment-based (sampling_rate=1)
  - Total epochs: 55 (5 + 50)
  - Early stopping: 5 epochs patience
  - Stage 2: Higher LR (5e-5) + no mixup/cutmix

Run from project root:
    python train_videomae/twostage_pretrained_vitb_ssv2_uniform.py --start_fold 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT.parent))

from train_videomae.twostage_pretrained import DEFAULT_CONFIG, run

PROJECT_ROOT = _PACKAGE_ROOT.parent

CONFIG_VITB_SSV2_UNIFORM = replace(
    DEFAULT_CONFIG,
    label="ViT-Base | SSV2 pretrained | SSV2 uniform loader | Two-Stage",
    output_base=PROJECT_ROOT / "work_dirs_VideoMAE_twostage" / "vitb_ssv2_uniform_twostage",
    model_name="vit_base_patch16_224",
    pretrained_file="videomae_base_patch16_224_ssv2.pth",
    sampling_rate=1,
    batch_size=8,
    num_workers=8,
    stage2_lr="5e-5",
    stage2_mixup_off=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage VideoMAE ViT-Base on SIBI (SSV2 pretrained, SSV2 uniform sampling)"
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(CONFIG_VITB_SSV2_UNIFORM, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
