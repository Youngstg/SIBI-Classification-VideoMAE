"""VideoMAE training scripts for SIBI dataset (ViT-Small / ViT-Base).

This package contains the training entry points and their shared helpers:
    - training_utils : subprocess runner, log parser, checkpoint cleanup
    - evaluation     : learning curves, confusion matrix, classification report

Every training script is meant to be executed directly from the project root:
    python train_videomae/twostage_pretrained.py --start_fold 1
"""
