"""Extended Stage 2 variant: 100 epochs for investigating plateau behavior.

Stage 1: 5 epochs (default) | Stage 2: 100 epochs (vs default 50)
Used to determine if validation plateau at epoch ~50 is real or premature.

Run from project root:
    python train_videomae/twostage_pretrained_100ep.py --start_fold 1
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

CONFIG_STAGE2_100EP = replace(
    DEFAULT_CONFIG,
    label="ViT-Small | SSV2 | Two-Stage | Stage 2 100 Epoch",
    output_base=PROJECT_ROOT / "work_dirs" / "vits_ssv2_twostage_stage2_100ep",
    stage2_epochs=100,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage pretrained, Stage 2 = 100 epochs")
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()
    run(CONFIG_STAGE2_100EP, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
