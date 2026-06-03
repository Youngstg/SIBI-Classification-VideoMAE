"""Quick smoke-test variant: both stages limited to 10 epochs total.

Useful for rapid iteration and debugging before committing to full training.
Stage 1: 5 epochs | Stage 2: 10 epochs (vs default 5+50)

Run from project root:
    python train_videomae/twostage_pretrained_10ep.py --start_fold 1
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

CONFIG_10EP = replace(
    DEFAULT_CONFIG,
    label="ViT-Small | SSV2 | Two-Stage | 10 Epoch",
    output_base=PROJECT_ROOT / "work_dirs" / "vits_ssv2_twostage_10ep",
    stage1_epochs=5,
    stage2_epochs=10,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage pretrained, 10 epoch variant")
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(CONFIG_10EP, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
