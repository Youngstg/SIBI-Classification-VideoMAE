"""Single-stage I3D training on the SIBI dataset, WLASL-paper style.

This script follows the original WLASL paper's single-stage recipe: load the
Kinetics-pretrained RGB-I3D weights, swap the final classifier to the SIBI
class count, and fine-tune end-to-end with early stopping on validation
accuracy.

Run from this directory:
    python train_sibi_i3d_singlestage.py --fold 1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms

import videotransforms
from configs import Config
from datasets.nslt_dataset import NSLT as Dataset
from pytorch_i3d import InceptionI3d


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

I3D_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = I3D_DIR.parents[2]
FOLDS_DIR = PROJECT_ROOT / "folds_data"
VIDEOS_DIR = FOLDS_DIR / "videos"
PREPROCESS_DIR = I3D_DIR / "preprocess"
WEIGHTS_DIR = I3D_DIR / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "rgb_imagenet.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "work_dirs" / "sibi_i3d_singlestage"

NUM_FOLDS = 5
MODEL_LABEL = "I3D Single-Stage"

# Plot palette (reddish tones, matches the existing learning curve style).
PLOT_RED_DARK = "#8B1E1E"
PLOT_RED = "#C62828"
PLOT_RED_LIGHT = "#E57373"
PLOT_ROSE = "#F8B4B4"


# ---------------------------------------------------------------------------
# Simple IO helpers
# ---------------------------------------------------------------------------

def read_index_file(txt_path: Path | str) -> List[int]:
    """Read a file of one-integer-per-line. Missing file -> empty list."""
    path = Path(txt_path)
    if not path.exists():
        return []

    values: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(line))
            except ValueError:
                continue
    return values


def save_index_file(txt_path: Path | str, values: Sequence[int]) -> None:
    with open(txt_path, "w") as f:
        for value in values:
            f.write(f"{int(value)}\n")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner module when wrapped in ``DataParallel``."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_checkpoint_state(
    model: nn.Module,
    checkpoint_path: Path | str,
    map_location: str = "cpu",
    strict: bool = True,
):
    """Load a state dict, stripping any ``module.`` prefix from the keys."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return unwrap_model(model).load_state_dict(cleaned, strict=strict)


def load_class_names(num_classes: Optional[int] = None) -> List[str]:
    """Load SIBI class names from ``folds_data/label_map.json``.

    Falls back to ``class_0 .. class_N-1`` when the file is missing or
    underfilled.
    """
    label_map_path = FOLDS_DIR / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        labels = label_data.get("labels", {})
        if isinstance(labels, dict):
            class_names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
        elif isinstance(labels, list):
            class_names = list(labels)
        else:
            class_names = []

        if class_names:
            if num_classes is None:
                return class_names
            if len(class_names) >= num_classes:
                return class_names[:num_classes]

    if num_classes is None:
        return []
    return [f"class_{i}" for i in range(num_classes)]


def load_existing_best_val_score(save_model: Path | str) -> Optional[float]:
    """Return the previous best-val score in ``evaluation_summary.json``."""
    summary_file = Path(save_model) / "evaluation_summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if "best_val_accuracy" in summary:
        return float(summary["best_val_accuracy"])
    if "best_val_accuracy_percent" in summary:
        return float(summary["best_val_accuracy_percent"]) / 100.0
    return None


# ---------------------------------------------------------------------------
# Training history (per-fold learning curves)
# ---------------------------------------------------------------------------

def save_training_history(save_model: Path, history_rows: Sequence[dict]) -> None:
    if not history_rows:
        return

    json_path = save_model / "training_history.json"
    csv_path = save_model / "training_history.csv"
    plot_path = save_model / "learning_curve.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(history_rows), f, indent=2)

    fieldnames = list(history_rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    _plot_single_fold_curve(plot_path, history_rows)

    print(f"Training history saved to {json_path}")
    print(f"Training history CSV saved to {csv_path}")
    print(f"Learning curve saved to {plot_path}")


def _plot_single_fold_curve(plot_path: Path, history_rows: Sequence[dict]) -> None:
    epochs = [row["epoch"] for row in history_rows]
    train_total_loss = [row["train_total_loss"] for row in history_rows]
    val_total_loss = [row["val_total_loss"] for row in history_rows]
    train_acc = [row["train_acc"] * 100.0 for row in history_rows]
    val_acc = [row["val_acc"] * 100.0 for row in history_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_total_loss, marker="o", linewidth=2, color=PLOT_RED_LIGHT, label="Train")
    axes[0].plot(epochs, val_total_loss, marker="s", linewidth=2, color=PLOT_RED_DARK, label="Validation")
    axes[0].set_title(f"Loss Curve\n{MODEL_LABEL}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2, color=PLOT_RED_LIGHT, label="Train")
    axes[1].plot(epochs, val_acc, marker="s", linewidth=2, color=PLOT_RED_DARK, label="Validation")
    axes[1].set_title(f"Accuracy Curve\n{MODEL_LABEL}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_training_history(history_path: Path | str) -> List[dict]:
    path = Path(history_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_all_fold_histories(
    output_dir: Path | str,
    num_folds: int = NUM_FOLDS,
) -> List[Tuple[int, List[dict]]]:
    output_dir = Path(output_dir)
    histories: List[Tuple[int, List[dict]]] = []
    for fold_idx in range(1, num_folds + 1):
        history = load_training_history(output_dir / f"fold_{fold_idx}" / "training_history.json")
        if history:
            histories.append((fold_idx, history))
    return histories


# ---------------------------------------------------------------------------
# Cross-fold reporting
# ---------------------------------------------------------------------------

def generate_learning_curves_per_fold(
    output_dir: Path,
    histories: Sequence[Tuple[int, List[dict]]],
) -> None:
    if not histories:
        print(f"[INFO] No training history found in {output_dir}")
        return

    fig, axes = plt.subplots(NUM_FOLDS, 2, figsize=(16, 5 * NUM_FOLDS))
    if NUM_FOLDS == 1:
        axes = np.array([axes])

    for fold_idx in range(1, NUM_FOLDS + 1):
        ax_acc = axes[fold_idx - 1][0]
        ax_loss = axes[fold_idx - 1][1]
        matched = next((history for idx, history in histories if idx == fold_idx), None)

        if not matched:
            _set_empty_fold_plot(ax_acc, f"Fold {fold_idx} - Validation Accuracy")
            _set_empty_fold_plot(ax_loss, f"Fold {fold_idx} - Loss Curve")
            continue

        _plot_fold_accuracy_panel(ax_acc, fold_idx, matched)
        _plot_fold_loss_panel(ax_loss, fold_idx, matched)

    plt.suptitle(f"Learning Curves per Fold\n{MODEL_LABEL}", fontsize=14, y=1.005)
    plt.tight_layout()
    path = output_dir / "learning_curves_per_fold.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path.name}")


def _set_empty_fold_plot(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.text(0.5, 0.5, "No history", ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")


def _plot_fold_accuracy_panel(ax, fold_idx: int, history: Sequence[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    val_acc = [row.get("val_acc", 0.0) * 100.0 for row in history]

    ax.plot(epochs, val_acc, color=PLOT_RED, marker="o",
            markersize=3, linewidth=1.8, label="Val Acc@1")
    best_idx = int(np.argmax(val_acc))
    ax.axvline(
        x=epochs[best_idx], color=PLOT_RED_DARK, linestyle="--", alpha=0.7,
        label=f"Best ep{epochs[best_idx]}: {val_acc[best_idx]:.2f}%",
    )
    ax.set_title(f"Fold {fold_idx} - Validation Accuracy", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acc@1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_fold_loss_panel(ax, fold_idx: int, history: Sequence[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_total_loss", 0.0) for row in history]
    val_loss = [row.get("val_total_loss", 0.0) for row in history]

    ax.plot(epochs, train_loss, color=PLOT_RED_LIGHT, marker="o",
            markersize=3, linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_loss, color=PLOT_RED_DARK, marker="s",
            markersize=3, linewidth=1.5, label="Val Loss")
    ax.set_title(f"Fold {fold_idx} - Loss Curve", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def generate_mean_learning_curve(
    output_dir: Path,
    histories: Sequence[Tuple[int, List[dict]]],
) -> None:
    if not histories:
        print(f"[INFO] No training history found in {output_dir}")
        return

    all_val_acc = [[row.get("val_acc", 0.0) * 100.0 for row in hist] for _, hist in histories]
    all_val_loss = [[row.get("val_total_loss", 0.0) for row in hist] for _, hist in histories]
    all_train_loss = [[row.get("train_total_loss", 0.0) for row in hist] for _, hist in histories]

    max_len = max(len(curve) for curve in all_val_acc)

    def pad(values: Sequence[float]) -> List[float]:
        return list(values) + [values[-1]] * (max_len - len(values))

    mean_acc = np.array([pad(curve) for curve in all_val_acc]).mean(axis=0)
    std_acc = np.array([pad(curve) for curve in all_val_acc]).std(axis=0)
    mean_val_loss = np.array([pad(curve) for curve in all_val_loss]).mean(axis=0)
    std_val_loss = np.array([pad(curve) for curve in all_val_loss]).std(axis=0)
    mean_train_loss = np.array([pad(curve) for curve in all_train_loss]).mean(axis=0)
    epoch_range = list(range(1, max_len + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epoch_range, mean_acc, color=PLOT_RED, linewidth=2, label="Mean Val Acc@1")
    axes[0].fill_between(
        epoch_range, mean_acc - std_acc, mean_acc + std_acc,
        alpha=0.25, color=PLOT_ROSE, label="+/-1 Std",
    )
    axes[0].set_title("Mean Validation Accuracy (5-Fold CV)", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Acc@1 (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch_range, mean_train_loss, color=PLOT_RED_LIGHT, linewidth=2, label="Mean Train Loss")
    axes[1].plot(epoch_range, mean_val_loss, color=PLOT_RED_DARK, linewidth=2, label="Mean Val Loss")
    axes[1].fill_between(
        epoch_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
        alpha=0.25, color=PLOT_ROSE, label="Val +/-1 Std",
    )
    axes[1].set_title("Mean Loss Curve (5-Fold CV)", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"{MODEL_LABEL}\nMean Learning Curves", fontsize=14)
    plt.tight_layout()
    plot_path = output_dir / "learning_curves_mean.png"
    legacy_plot_path = output_dir / "learning_curve_all_folds.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.savefig(legacy_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {plot_path.name}")


def save_fold_evaluation(
    save_model: Path,
    fold_idx: int,
    num_classes: int,
    best_val_score: float,
    final_test_accuracy: float,
    all_labels: Sequence[int],
    all_preds: Sequence[int],
    report: str,
) -> None:
    gt_file = save_model / "ground_truths.txt"
    pred_file = save_model / "predictions.txt"
    report_file = save_model / "classification_report.txt"
    summary_file = save_model / "evaluation_summary.json"

    save_index_file(gt_file, all_labels)
    save_index_file(pred_file, all_preds)

    header = [
        f"Fold: {fold_idx}",
        f"Model: {MODEL_LABEL}",
        f"Best Val Acc@1: {best_val_score * 100:.2f}%",
        f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%",
        f"Total Samples: {len(all_labels)}",
        "",
        report,
    ]
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header).strip() + "\n")

    summary = {
        "fold": fold_idx,
        "model": MODEL_LABEL,
        "num_classes": num_classes,
        "num_samples": len(all_labels),
        "best_val_accuracy": best_val_score,
        "best_val_accuracy_percent": best_val_score * 100.0,
        "final_test_accuracy": final_test_accuracy,
        "final_test_accuracy_percent": final_test_accuracy * 100.0,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Ground truths saved to {gt_file}")
    print(f"Predictions saved to {pred_file}")
    print(f"Classification report saved to {report_file}")
    print(f"Evaluation summary saved to {summary_file}")


@dataclass
class FoldEntry:
    fold_idx: int
    labels: List[int]
    preds: List[int]
    num_samples: int
    num_classes: int
    best_val_acc: float
    final_test_acc: float


def _load_fold_entries(output_dir: Path, num_folds: int) -> List[FoldEntry]:
    """Read the per-fold saved evaluation artefacts for report aggregation."""
    entries: List[FoldEntry] = []
    for fold_idx in range(1, num_folds + 1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        gt_file = fold_dir / "ground_truths.txt"
        pred_file = fold_dir / "predictions.txt"
        summary_file = fold_dir / "evaluation_summary.json"

        if not gt_file.exists() or not pred_file.exists():
            continue

        labels = read_index_file(gt_file)
        preds = read_index_file(pred_file)
        n = min(len(labels), len(preds))
        if n == 0:
            continue

        summary: Dict = {}
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)

        best_val_acc = float(summary.get(
            "best_val_accuracy_percent",
            float(summary.get("best_val_accuracy", 0.0)) * 100.0,
        ))
        final_test_acc = float(summary.get(
            "final_test_accuracy_percent",
            float(summary.get("final_test_accuracy", 0.0)) * 100.0,
        ))

        entries.append(FoldEntry(
            fold_idx=fold_idx,
            labels=labels[:n],
            preds=preds[:n],
            num_samples=n,
            num_classes=int(summary.get("num_classes", 0)),
            best_val_acc=best_val_acc,
            final_test_acc=final_test_acc,
        ))
    return entries


def _plot_accuracy_bar(output_dir: Path, fold_accuracies: Sequence[float], num_folds: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    xlabels = [f"Fold {i}" for i in range(1, num_folds + 1)]
    bars = ax.bar(xlabels, fold_accuracies, color=PLOT_RED, edgecolor="black", linewidth=0.7)

    valid = [a for a in fold_accuracies if a > 0]
    mean_v = sum(valid) / len(valid) if valid else 0.0
    ax.axhline(y=mean_v, color=PLOT_RED_DARK, linestyle="--",
               linewidth=1.5, label=f"Mean: {mean_v:.2f}%")

    for bar, acc in zip(bars, fold_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title(f"Best Val Acc@1 per Fold\n{MODEL_LABEL}", fontsize=13)
    ax.set_ylabel("Best Val Acc@1 (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "accuracy_per_fold.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path.name}")


def _plot_overall_confusion(
    output_dir: Path,
    all_labels: Sequence[int],
    all_preds: Sequence[int],
    num_classes: int,
    class_names: Sequence[str],
    mean_acc: float,
) -> str:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Jumlah Sampel", fontsize=10)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black", fontsize=10,
            )

    ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontsize=12)
    ax.set_xlabel("Prediksi Model", fontsize=12)
    ax.set_title(
        f"Confusion Matrix - {MODEL_LABEL}\n"
        f"Mean Best Val Acc@1: {mean_acc:.2f}% | Total Sampel: {len(all_labels)}",
        fontsize=13,
    )
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path.name}")

    return classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=list(class_names),
        zero_division=0,
    )


def _write_overall_report(
    output_dir: Path,
    fold_entries: Sequence[FoldEntry],
    num_folds: int,
    overall_report: str,
) -> None:
    width = 72
    lines: List[str] = []
    lines.append("=" * width)
    lines.append(f"  CLASSIFICATION REPORT - {MODEL_LABEL}")
    lines.append(f"  Pretrained           : {DEFAULT_WEIGHTS.name}")
    lines.append(f"  Num Folds            : {num_folds}")
    lines.append("  Data Loader          : NSLT RGB")
    lines.append("  Early Stop Patience  : sesuai config training")
    lines.append("=" * width + "\n")

    fold_map = {entry.fold_idx: entry for entry in fold_entries}
    for fold_idx in range(1, num_folds + 1):
        entry = fold_map.get(fold_idx)
        best_v = entry.best_val_acc if entry else 0.0
        final_acc = entry.final_test_acc if entry else 0.0
        num_samples = entry.num_samples if entry else 0
        lines.append("-" * width)
        lines.append(
            f"  FOLD {fold_idx}  -  Best Val Acc@1: {best_v:.2f}% | "
            f"Final Test Acc: {final_acc:.2f}% | Sampel: {num_samples}"
        )
        lines.append("-" * width + "\n")

    mean_acc = sum(e.best_val_acc for e in fold_entries) / len(fold_entries)
    mean_test_acc = sum(e.final_test_acc for e in fold_entries) / len(fold_entries)
    total_samples = sum(e.num_samples for e in fold_entries)

    lines.append(f"\n{'-' * width}")
    lines.append("  RINGKASAN")
    lines.append(f"{'-' * width}")
    for entry in fold_entries:
        lines.append(
            f"  Fold {entry.fold_idx}  Best Val Acc@1 : {entry.best_val_acc:.2f}% | "
            f"Final Test Acc : {entry.final_test_acc:.2f}%"
        )
    lines.append(f"  {'-' * 45}")
    lines.append(f"  Mean Best Val Acc@1  : {mean_acc:.2f}%")
    lines.append(f"  Mean Final Test Acc  : {mean_test_acc:.2f}%")
    lines.append(f"  Total Sampel         : {total_samples}")
    lines.append("\n" + "=" * width)
    lines.append("  OVERALL CLASSIFICATION REPORT")
    lines.append("=" * width + "\n")
    lines.append(overall_report)
    lines.append("\n" + "=" * width + "\n")

    full = "\n".join(lines)
    print(full)
    path = output_dir / "classification_report_all_folds.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"[OK] {path.name}")


def generate_reports(output_dir: Path | str, num_folds: int = NUM_FOLDS) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_all_fold_histories(output_dir, num_folds=num_folds)
    generate_learning_curves_per_fold(output_dir, histories)
    generate_mean_learning_curve(output_dir, histories)

    fold_entries = _load_fold_entries(output_dir, num_folds)
    if not fold_entries:
        print(f"[INFO] No completed fold evaluations found in {output_dir}")
        return

    fold_map = {entry.fold_idx: entry for entry in fold_entries}
    fold_accuracies = [fold_map.get(i, FoldEntry(i, [], [], 0, 0, 0.0, 0.0)).best_val_acc
                       for i in range(1, num_folds + 1)]
    mean_acc = sum(e.best_val_acc for e in fold_entries) / len(fold_entries)

    num_classes = max(
        max((e.num_classes for e in fold_entries), default=0),
        max((max(e.labels + e.preds) + 1 for e in fold_entries), default=0),
    )
    class_names = load_class_names(num_classes)

    try:
        _plot_accuracy_bar(output_dir, fold_accuracies, num_folds)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Bar chart gagal: {exc}")

    all_labels: List[int] = []
    all_preds: List[int] = []
    for entry in fold_entries:
        all_labels.extend(entry.labels)
        all_preds.extend(entry.preds)

    try:
        overall_report = _plot_overall_confusion(
            output_dir, all_labels, all_preds, num_classes, class_names, mean_acc,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Confusion matrix gagal: {exc}")
        return

    try:
        _write_overall_report(output_dir, fold_entries, num_folds, overall_report)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Classification report gagal: {exc}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_dataloaders(configs, train_split: str, val_split: str, root) -> Dict[str, torch.utils.data.DataLoader]:
    """Build train and test loaders using the official NSLT dataset class."""
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224),
    ])

    train_dataset = Dataset(train_split, "train", root, "rgb", train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = Dataset(val_split, "test", root, "rgb", test_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    return {"train": train_loader, "test": val_loader}


def _build_model(num_classes: int, extra_weights: Optional[str]) -> nn.Module:
    """Construct I3D, load Kinetics weights, replace the classifier head."""
    i3d = InceptionI3d(400, in_channels=3)
    if DEFAULT_WEIGHTS.exists():
        print(f"loading weights {DEFAULT_WEIGHTS}")
        load_checkpoint_state(i3d, DEFAULT_WEIGHTS)
    else:
        print(f"[WARN] pretrained tidak ditemukan: {DEFAULT_WEIGHTS}")
        print("[WARN] training from scratch")

    i3d.replace_logits(num_classes)

    if extra_weights:
        print(f"loading weights {extra_weights}")
        load_checkpoint_state(i3d, extra_weights)

    i3d = i3d.cuda()
    return nn.DataParallel(i3d)


def _run_phase(
    phase: str,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    num_steps_per_update: int,
    steps: int,
    epoch: int,
    max_steps: int,
    num_classes: int,
) -> Tuple[dict, int, bool]:
    """Run one train or test phase of a single epoch.

    Returns ``(metrics, new_steps, stop_training)``.
    """
    is_training = phase == "train"
    model.train(is_training)

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    epoch_total_loss = 0.0
    epoch_loc_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_batches = 0
    num_iter = 0
    optimizer.zero_grad()

    phase_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    stop_training = False

    for data in loader:
        num_iter += 1
        if data == -1:
            continue

        inputs, labels, _ = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        t = inputs.size(2)

        per_frame_logits = model(inputs, pretrained=False)
        per_frame_logits = F.interpolate(
            per_frame_logits, size=t, mode="linear", align_corners=False,
        )

        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        tot_loc_loss += loc_loss.item()
        epoch_loc_loss += loc_loss.item()

        predictions = torch.max(per_frame_logits, dim=2)[0]
        gt = torch.max(labels, dim=2)[0]

        cls_loss = F.binary_cross_entropy_with_logits(predictions, gt)
        tot_cls_loss += cls_loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_total_loss += 0.5 * (loc_loss.item() + cls_loss.item())
        epoch_batches += 1

        for idx in range(per_frame_logits.shape[0]):
            phase_confusion[
                torch.argmax(gt[idx]).item(),
                torch.argmax(predictions[idx]).item(),
            ] += 1

        loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
        tot_loss += loss.item()

        if num_iter == max(1, num_steps_per_update // 2):
            print(epoch, steps, loss.item())

        if is_training:
            loss.backward()

        if num_iter == num_steps_per_update and is_training:
            steps += 1
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()

            if steps % 10 == 0:
                acc = float(np.trace(phase_confusion)) / max(np.sum(phase_confusion), 1)
                print(
                    "Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}".format(
                        epoch, phase,
                        tot_loc_loss / (10 * num_steps_per_update),
                        tot_cls_loss / (10 * num_steps_per_update),
                        tot_loss / 10,
                        acc,
                    )
                )
                tot_loss = 0.0
                tot_loc_loss = 0.0
                tot_cls_loss = 0.0

        if max_steps > 0 and steps >= max_steps:
            stop_training = True
            break

    batch_count = max(epoch_batches, 1)
    metrics = {
        "loc_loss": epoch_loc_loss / batch_count,
        "cls_loss": epoch_cls_loss / batch_count,
        "total_loss": epoch_total_loss / batch_count,
        "acc": float(np.trace(phase_confusion)) / max(np.sum(phase_confusion), 1),
    }
    return metrics, steps, stop_training


def _evaluate_checkpoint(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
) -> Tuple[List[int], List[int], float]:
    """Run the final eval pass used for the saved report."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for data in loader:
            if data == -1:
                continue
            inputs, labels, _ = data
            inputs = inputs.cuda()
            per_frame_logits = model(inputs, pretrained=False)
            predictions = torch.max(per_frame_logits, dim=2)[0]
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            gts = torch.argmax(torch.max(labels, dim=2)[0], dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(gts)

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return all_labels, all_preds, acc


def _save_confusion_matrix(
    save_model: Path,
    cm: np.ndarray,
    class_names: Sequence[str],
    fold_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    ax.set_title(f"Confusion Matrix - Fold {fold_label}")
    fig.tight_layout()
    cm_path = save_model / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


def run(
    configs,
    train_split: str,
    val_split: str,
    save_model: Path,
    root,
    weights: Optional[str] = None,
    eval_only: bool = False,
) -> None:
    """Train or evaluate a single fold end-to-end."""
    print(configs)

    dataloaders = _build_dataloaders(configs, train_split, val_split, root)
    num_classes = dataloaders["train"].dataset.num_classes

    i3d = _build_model(num_classes, extra_weights=weights)

    optimizer = optim.Adam(
        i3d.parameters(),
        lr=configs.init_lr,
        weight_decay=configs.adam_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.3,
    )

    steps = 0
    epoch = 0
    best_val_score = float("-inf")
    best_model_path = save_model / "best.pt"
    epochs_since_improvement = 0
    history_rows: List[dict] = []

    if eval_only:
        print("[INFO] Eval-only mode aktif: training dilewati.")
    else:
        while epoch < configs.max_epochs and (configs.max_steps <= 0 or steps < configs.max_steps):
            step_label = (
                f"Step {steps}/{configs.max_steps}"
                if configs.max_steps > 0 else
                f"Step {steps}/unlimited"
            )
            print(f"Epoch {epoch}/{configs.max_epochs} | {step_label}")
            print("-" * 10)

            epoch += 1
            epoch_metrics = {"epoch": epoch}
            stop_training = False

            for phase in ["train", "test"]:
                metrics, steps, stop_training = _run_phase(
                    phase=phase,
                    model=i3d,
                    loader=dataloaders[phase],
                    optimizer=optimizer,
                    num_steps_per_update=configs.update_per_step,
                    steps=steps,
                    epoch=epoch,
                    max_steps=configs.max_steps,
                    num_classes=num_classes,
                )

                prefix = "train" if phase == "train" else "val"
                epoch_metrics[f"{prefix}_loc_loss"] = metrics["loc_loss"]
                epoch_metrics[f"{prefix}_cls_loss"] = metrics["cls_loss"]
                epoch_metrics[f"{prefix}_total_loss"] = metrics["total_loss"]
                epoch_metrics[f"{prefix}_acc"] = metrics["acc"]

                if phase == "test":
                    val_score = metrics["acc"]
                    if val_score > best_val_score:
                        epochs_since_improvement = 0
                        best_val_score = val_score
                        ckpt_name = save_model / f"nslt_sibi_{num_classes}_{str(steps).zfill(6)}_{val_score:.6f}.pt"
                        torch.save(i3d.module.state_dict(), ckpt_name)
                        torch.save(i3d.module.state_dict(), best_model_path)
                        print(ckpt_name)
                        print(f"Updated best model: {best_model_path}")
                    else:
                        epochs_since_improvement += 1

                    if epochs_since_improvement >= configs.early_stop_patience:
                        print(
                            f"Early stopping after {epoch} epochs: "
                            f"no validation improvement for {configs.early_stop_patience} "
                            "consecutive epochs."
                        )
                        stop_training = True

                    print(
                        "VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}".format(
                            phase,
                            metrics["loc_loss"],
                            metrics["cls_loss"],
                            metrics["total_loss"],
                            val_score,
                        )
                    )

                    scheduler.step(metrics["total_loss"])

                if stop_training:
                    break

            if "train_total_loss" in epoch_metrics and "val_total_loss" in epoch_metrics:
                history_rows.append(epoch_metrics)

            if stop_training:
                break

        save_training_history(save_model, history_rows)

    # ----------------- Final evaluation -----------------
    print("=== FINAL EVALUATION ===")
    checkpoint_for_eval = Path(weights) if weights else best_model_path
    if checkpoint_for_eval.exists():
        load_checkpoint_state(i3d, checkpoint_for_eval)
        print(f"Loaded model for evaluation: {checkpoint_for_eval}")
    else:
        if eval_only:
            raise FileNotFoundError(f"Checkpoint evaluasi tidak ditemukan: {checkpoint_for_eval}")
        print("No best model found, using last state.")

    all_labels, all_preds, acc = _evaluate_checkpoint(i3d, dataloaders["test"], num_classes)

    class_names = load_class_names(num_classes)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )

    print(f"Final Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    fold_idx = int(Path(val_split).stem.split("fold")[-1])
    if best_val_score == float("-inf"):
        best_val_score = load_existing_best_val_score(save_model)
    if best_val_score is None or best_val_score == float("-inf"):
        best_val_score = acc

    save_fold_evaluation(
        save_model=save_model,
        fold_idx=fold_idx,
        num_classes=num_classes,
        best_val_score=best_val_score,
        final_test_accuracy=acc,
        all_labels=all_labels,
        all_preds=all_preds,
        report=report,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    _save_confusion_matrix(
        save_model, cm, class_names,
        fold_label=Path(val_split).stem.split("_")[-1],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SIBI with the single-stage WLASL-style I3D recipe."
    )
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument(
        "--config", type=str,
        default=str(I3D_DIR / "configfiles" / "asl2000.ini"),
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Checkpoint override untuk resume/fine-tune",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Lewati training dan langsung evaluasi checkpoint.",
    )
    args = parser.parse_args()

    train_split = PREPROCESS_DIR / f"sibi_train_fold{args.fold}.json"
    val_split = PREPROCESS_DIR / f"sibi_val_fold{args.fold}.json"
    save_model = DEFAULT_OUTPUT / f"fold_{args.fold}"
    save_model.mkdir(parents=True, exist_ok=True)

    root = {"word": str(VIDEOS_DIR)}
    configs = Config(args.config)

    print(root, train_split, val_split)
    run(
        configs=configs,
        train_split=str(train_split),
        val_split=str(val_split),
        save_model=save_model,
        root=root,
        weights=args.weights,
        eval_only=args.eval_only,
    )
    print("\n[INFO] Membuat laporan agregat lintas fold...")
    generate_reports(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
