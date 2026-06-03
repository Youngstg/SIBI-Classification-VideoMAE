"""Two-stage training from random initialization (no pretrained weights).

Same pipeline as twostage_pretrained, but both stages start from scratch.
This ablation tests: "How much benefit does pretraining really provide?"

Stage 1: Train head from random init (frozen backbone from random weights)
Stage 2: Full fine-tune with unfrozen random-init backbone

Run from project root:
    python train_videomae/twostage_scratch.py --start_fold 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT.parent))

from train_videomae.twostage_pretrained import DEFAULT_CONFIG, PROJECT_ROOT, run

SCRATCH_CONFIG = replace(
    DEFAULT_CONFIG,
    label="ViT-Small | SSV2 | Two-Stage | Scratch",
    output_base=PROJECT_ROOT / "work_dirs" / "vits_ssv2_twostage_scratch",
    # No pretrained weights - everything starts from scratch.
    pretrained_file=None,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage scratch training ViT-Small on SIBI"
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(SCRATCH_CONFIG, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
