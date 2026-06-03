# SIBI Sign Language Recognition with VideoMAE

Source code for undergraduate thesis:  
**"Analisis Kinerja Model VideoMAE dalam Klasifikasi Gerakan Visual Bahasa Isyarat SIBI"**

**Author:** Lucky Immanuel Sitanggang (122140179)  
**Institution:** Institut Teknologi Sumatera (ITERA)

---

## Overview

This repository contains all training, evaluation, and inference code used to benchmark video transformer architectures (VideoMAE and ViViT) on the SIBI (Sistem Isyarat Bahasa Indonesia) sign language dataset. Experiments use 5-fold stratified cross-validation and cover backbone variants, training strategies, sampling methods, epoch counts, augmentation effects, and cross-architecture comparisons.

**Best result:** ViT-B/16 (K400 weights) · dense sampling · two-stage · no mixup/CutMix → **97.18% accuracy**

---

## Project Structure

```
SourceCode/
├── VideoMAE/                  # Core VideoMAE model & dataset code (adapted)
├── train_videomae/            # VideoMAE fine-tuning scripts
│   ├── twostage_pretrained.py           # E5  – ViT-S SSV2, uniform, 50ep
│   ├── twostage_pretrained_10ep.py      # E8  – ViT-S SSV2, uniform, 10ep
│   ├── twostage_pretrained_100ep.py     # E10 – ViT-S SSV2, uniform, 100ep
│   ├── twostage_pretrained_densesampling.py          # E11 – ViT-S SSV2, dense
│   ├── twostage_pretrained_densesampling_vitb_k400.py# E14 – ViT-B K400, dense
│   ├── twostage_pretrained_vitb_ssv2_uniform.py      # E3  – ViT-B SSV2
│   ├── twostage_pretrained_vits_ssv2_uniform.py      # E13 – ViT-S SSV2, no aug
│   ├── twostage_scratch.py              # E6  – two-stage from scratch
│   ├── singlestage_scratch.py           # E7  – single-stage from scratch
│   ├── all_models_singlestage.py        # E1-E4 – backbone comparison (single-stage)
│   ├── training_utils.py                # Shared training helpers
│   └── evaluation.py                    # Shared evaluation & report generation
├── train_vivit/               # ViViT fine-tuning
│   └── finetune_vivit.py      # Fine-tune HuggingFace ViViT on SIBI folds
├── scripts/                   # Utility & preprocessing scripts
│   ├── preprocessing.py       # Video standardization + 5-fold stratified split
│   ├── setup_sibi.py          # Dataset download/setup helper
│   ├── inferensi_manual.py    # Manual inference on a single video
│   ├── visualize_augmentations.py
│   └── ...
├── benchmarks/                # Inference-time benchmark scripts
│   ├── benchmark_vits_ssv2.py
│   ├── benchmark_vitb_k400.py
│   ├── benchmark_vivit.py
│   ├── benchmark_i3d.py
│   └── run_benchmark.sh
├── SIBI_Recognition_VideoMAE-demo/  # Gradio demo (HuggingFace Spaces)
├── folds_data/                # Generated CSV fold splits + label map
├── pretrained/                # Pre-trained VideoMAE weights (.pth)
├── work_dirs_all_experiment/  # Saved results from all experiments
└── requirements.txt
```

---

## Experiments

| ID | Script | Config | Accuracy |
|----|--------|--------|----------|
| E1 | `all_models_singlestage.py` | ViT-B K400, single-stage | — |
| E2 | `all_models_singlestage.py` | ViT-B SSV2, single-stage | — |
| E3 | `all_models_singlestage.py` | ViT-S K400, single-stage | — |
| E4 | `all_models_singlestage.py` | ViT-S SSV2, single-stage | — |
| E5 | `twostage_pretrained.py` | ViT-S SSV2, two-stage, uniform, 50ep | 63.85% |
| E6 | `twostage_scratch.py` | Two-stage, from scratch | 11.36% |
| E7 | `singlestage_scratch.py` | Single-stage, from scratch | 12.01% |
| E8 | `twostage_pretrained_10ep.py` | ViT-S SSV2, two-stage, 10ep | 60.28% |
| E10 | `twostage_pretrained_100ep.py` | ViT-S SSV2, two-stage, 100ep | 66.56% |
| E11 | `twostage_pretrained_densesampling.py` | ViT-S SSV2, dense sampling | 56.07% |
| E13 | `twostage_pretrained_vits_ssv2_uniform.py` | ViT-S SSV2, no mixup/CutMix | 93.94% |
| E14 | `twostage_pretrained_densesampling_vitb_k400.py` | ViT-B K400, dense, no aug | **97.18%** |
| — | `train_vivit/finetune_vivit.py` | ViViT-B/16x2 from K400 | 94.81% |
| — | *(I3D baseline)* | I3D ResNet-50 | 96.86% |

---

## Installation

```bash
# 1. Clone this repo
git clone <repo-url>
cd SourceCode

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install PyTorch (CUDA build – adjust version for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** On RunPod with RTX 5090 (Blackwell), install PyTorch separately using a Blackwell-compatible build before running `pip install -r requirements.txt`.

---

## Dataset Preparation

```bash
python scripts/preprocessing.py \
    --dataset_root Dataset/dataset_bersih \
    --output_dir folds_data \
    --num_folds 5
```

This script:
1. Standardizes videos to 224×224, 30 FPS
2. Generates 5-fold stratified CSV splits under `folds_data/`
3. Writes `label_map.json`

---

## Training

### VideoMAE – Two-Stage Fine-Tuning (recommended)

```bash
# Example: E5 – ViT-S SSV2, uniform sampling, 50 epochs
python train_videomae/twostage_pretrained.py --start_fold 1

# Example: E14 – ViT-B K400, dense sampling (best result)
python train_videomae/twostage_pretrained_densesampling_vitb_k400.py --start_fold 1
```

Training resumes automatically from the last completed fold. Results are saved to `work_dirs/<experiment_name>/`.

### VideoMAE – All Backbones Single-Stage (E1–E4)

```bash
python train_videomae/all_models_singlestage.py --start_model 1 --start_fold 1
```

### ViViT Fine-Tuning

```bash
python train_vivit/finetune_vivit.py --start_fold 1
```

---

## Evaluation Outputs

Each experiment produces the following under `work_dirs/<experiment>/`:

```
fold_1/
  log.txt                         # Per-epoch train/val metrics
  0.txt                           # Prediction scores
learning_curves_per_fold.png
learning_curves_mean.png
accuracy_per_fold.png
confusion_matrix.png
classification_report_all_folds.txt
```

---

## Pretrained Weights

Download pretrained VideoMAE weights and place them in `pretrained/`:

| File | Source |
|------|--------|
| `videomae_small_patch16_224_ssv2.pth` | [VideoMAE official](https://github.com/MCG-NJU/VideoMAE) |
| `videomae_small_patch16_224_k400.pth` | [VideoMAE official](https://github.com/MCG-NJU/VideoMAE) |
| `videomae_base_patch16_224_ssv2.pth` | [VideoMAE official](https://github.com/MCG-NJU/VideoMAE) |
| `videomae_base_patch16_224_k400.pth` | [VideoMAE official](https://github.com/MCG-NJU/VideoMAE) |

---

## Inference

### Single video inference

```bash
python scripts/inferensi_manual.py --video path/to/video.mp4
```

### Gradio demo (local)

```bash
cd SIBI_Recognition_VideoMAE-demo
pip install gradio
python app.py
```

---

## Benchmark

Measure inference time across architectures:

```bash
cd benchmarks
bash run_benchmark.sh
```

Results are saved to `benchmarks/benchmark_results.txt` and `benchmarks/inference_time_results.csv`.

---

## Repository Notes

- `folds_data/videos/` — preprocessed video clips (not tracked by git, generate locally)
- `pretrained/` — model weights (not tracked by git, download manually)
- `work_dirs_all_experiment/` — archived experiment results from the thesis

---

## Citation

If you use this code, please cite:

```
Lucky Immanuel Sitanggang. (2025). Analisis Kinerja Model VideoMAE dalam 
Klasifikasi Gerakan Visual Bahasa Isyarat SIBI. Tugas Akhir, Institut 
Teknologi Sumatera (ITERA).
```

---

## License

The `VideoMAE/` directory contains code adapted from [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) under the CC-BY-NC 4.0 license. All other code in this repository is for academic use.
