"""Shared evaluation + reporting for VideoMAE training scripts.

Every training script in this package produces the same artefacts per fold
(``log.txt`` with per-epoch metrics, a ``0.txt`` with prediction scores), so
the reporting can be shared. The main entry point is :func:`generate_reports`
which emits:

    - learning_curves_per_fold.png
    - learning_curves_mean.png
    - accuracy_per_fold.png
    - confusion_matrix.png
    - classification_report_all_folds.txt

All functions are pure w.r.t. filesystem side effects (read input dirs,
write output files); no training logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from .training_utils import read_gt_labels, read_log, read_predictions

REPORT_WIDTH = 72
# Kept as a historical reference so new runs can be compared against the
# original SSV2 baseline from the VideoMAE paper.
SSV2_BASELINE_ACC = 35.18


# ---------------------------------------------------------------------------
# Configuration describing one experiment
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Configuration for experiment evaluation and reporting.
    
    Defines paths to fold artifacts (logs, predictions) and report parameters.
    Supports two-stage training with optional stage split visualization.
    """

    output_dir: Path  # Root output directory containing fold_{i} folders
    folds_dir: Path   # Directory with CSV files (train_fold_i.csv, val_fold_i.csv)
    label: str        # Experiment name/label
    num_folds: int = 5
    class_names: Sequence[str] = field(default_factory=list)  # SIBI class names

    # Subdir within fold_{i}/ where logs and predictions are stored
    # (e.g., 'stage2' for two-stage training)
    log_subdir: str = ""
    pred_subdir: str = ""

    # Epoch number to mark stage transition in learning curves
    stage_split_epoch: Optional[int] = None

    # Printed in the header of the classification report.
    header_lines: Sequence[str] = field(default_factory=list)

    # Whether to include the classic 35.18% SSV2 baseline on charts.
    show_ssv2_baseline: bool = True

    def fold_dir(self, fold_idx: int) -> Path:
        return self.output_dir / f"fold_{fold_idx}"

    def fold_log(self, fold_idx: int) -> Path:
        base = self.fold_dir(fold_idx)
        return (base / self.log_subdir / "log.txt") if self.log_subdir else (base / "log.txt")

    def fold_pred(self, fold_idx: int) -> Path:
        base = self.fold_dir(fold_idx)
        return (base / self.pred_subdir / "0.txt") if self.pred_subdir else (base / "0.txt")


# ---------------------------------------------------------------------------
# Loading fold artefacts
# ---------------------------------------------------------------------------

def collect_fold_logs(config: ReportConfig) -> List[List[dict]]:
    """Return per-fold lists of epoch dicts, one list per fold."""
    return [read_log(config.fold_log(i)) for i in range(1, config.num_folds + 1)]


def best_acc_per_fold(fold_logs: Sequence[Sequence[dict]]) -> List[float]:
    """Extract the best ``val_acc1`` value for each fold."""
    return [
        max((row["val_acc1"] for row in epochs), default=0.0)
        for epochs in fold_logs
    ]


def collect_fold_predictions(
    config: ReportConfig,
) -> tuple[List[int], List[int]]:
    """Read ground truth and predictions from all completed folds.
    
    Returns:
        Tuple of (all_gt_labels, all_predictions) concatenated across folds
    """
    all_labels: List[int] = []
    all_preds: List[int] = []

    for fold_idx in range(1, config.num_folds + 1):
        # Check if fold has both CSV and predictions file
        pred_file = config.fold_pred(fold_idx)
        val_csv = config.folds_dir / f"val_fold_{fold_idx}.csv"
        if not val_csv.exists() or not pred_file.exists():
            continue

        # Read ground truth labels from validation CSV
        gt = read_gt_labels(val_csv)
        # Read model predictions from log file
        pred = read_predictions(pred_file)
        if not pred:
            continue

        # Ensure equal lengths and concatenate
        n = min(len(gt), len(pred))
        all_labels.extend(gt[:n])
        all_preds.extend(pred[:n])
    return all_labels, all_preds


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _pad_to_length(values: Sequence[float], length: int) -> List[float]:
    """Pad ``values`` with its last element up to ``length``."""
    values = list(values)
    if not values:
        return [0.0] * length
    return values + [values[-1]] * (length - len(values))


def _draw_stage_split(ax, config: ReportConfig, label: str = "Stage 2 start") -> None:
    if config.stage_split_epoch is None:
        return
    ax.axvline(
        x=config.stage_split_epoch,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=label,
    )


def plot_learning_curves_per_fold(
    config: ReportConfig,
    fold_logs: Sequence[Sequence[dict]],
) -> None:
    fig, axes = plt.subplots(config.num_folds, 2, figsize=(16, 5 * config.num_folds))
    if config.num_folds == 1:
        axes = np.array([axes])

    for fold_idx, epochs_data in enumerate(fold_logs, 1):
        if not epochs_data:
            continue

        epochs = [row["epoch"] + 1 for row in epochs_data]
        train_loss = [row.get("train_loss", 0) for row in epochs_data]
        val_loss = [row.get("val_loss", 0) for row in epochs_data]
        val_acc = [row.get("val_acc1", 0) for row in epochs_data]

        ax_acc = axes[fold_idx - 1][0]
        ax_loss = axes[fold_idx - 1][1]

        ax_acc.plot(epochs, val_acc, "b-o", markersize=3, linewidth=1.5, label="Val Acc@1")
        best_epoch = epochs[int(np.argmax(val_acc))]
        best_val = max(val_acc)
        ax_acc.axvline(
            x=best_epoch, color="red", linestyle="--", alpha=0.6,
            label=f"Best ep{best_epoch}: {best_val:.2f}%",
        )
        _draw_stage_split(ax_acc, config)
        ax_acc.set_title(f"Fold {fold_idx} - Validation Accuracy", fontsize=11)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Acc@1 (%)")
        ax_acc.legend(fontsize=7)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, 100)

        ax_loss.plot(epochs, train_loss, "r-o", markersize=3, linewidth=1.5, label="Train Loss")
        ax_loss.plot(epochs, val_loss, "b-o", markersize=3, linewidth=1.5, label="Val Loss")
        _draw_stage_split(ax_loss, config)
        ax_loss.set_title(f"Fold {fold_idx} - Loss Curve", fontsize=11)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7)
        ax_loss.grid(True, alpha=0.3)

    plt.suptitle(f"Learning Curves per Fold\n{config.label}", fontsize=14, y=1.005)
    plt.tight_layout()
    _save_figure(config.output_dir / "learning_curves_per_fold.png")


def plot_mean_learning_curve(
    config: ReportConfig,
    fold_logs: Sequence[Sequence[dict]],
) -> None:
    all_val_acc = []
    all_val_loss = []
    all_train_loss = []
    for epochs_data in fold_logs:
        if not epochs_data:
            continue
        all_val_acc.append([row.get("val_acc1", 0) for row in epochs_data])
        all_val_loss.append([row.get("val_loss", 0) for row in epochs_data])
        all_train_loss.append([row.get("train_loss", 0) for row in epochs_data])

    if not all_val_acc:
        return

    max_len = max(len(curve) for curve in all_val_acc)
    mean_val_acc = np.array([_pad_to_length(c, max_len) for c in all_val_acc]).mean(axis=0)
    std_val_acc = np.array([_pad_to_length(c, max_len) for c in all_val_acc]).std(axis=0)
    mean_val_loss = np.array([_pad_to_length(c, max_len) for c in all_val_loss]).mean(axis=0)
    std_val_loss = np.array([_pad_to_length(c, max_len) for c in all_val_loss]).std(axis=0)
    mean_train_loss = np.array([_pad_to_length(c, max_len) for c in all_train_loss]).mean(axis=0)
    epoch_range = list(range(1, max_len + 1))

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    ax_acc.plot(epoch_range, mean_val_acc, "b-", linewidth=2, label="Mean Val Acc@1")
    ax_acc.fill_between(
        epoch_range, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
        alpha=0.2, color="blue", label="+/-1 Std",
    )
    _draw_stage_split(ax_acc, config)
    ax_acc.set_title("Mean Validation Accuracy (5-Fold CV)", fontsize=12)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Acc@1 (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    ax_loss.plot(epoch_range, mean_train_loss, "r-", linewidth=2, label="Mean Train Loss")
    ax_loss.plot(epoch_range, mean_val_loss, "b-", linewidth=2, label="Mean Val Loss")
    ax_loss.fill_between(
        epoch_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
        alpha=0.2, color="blue", label="Val +/-1 Std",
    )
    _draw_stage_split(ax_loss, config)
    ax_loss.set_title("Mean Loss Curve (5-Fold CV)", fontsize=12)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.suptitle(f"{config.label}\nMean Learning Curves", fontsize=14)
    plt.tight_layout()
    _save_figure(config.output_dir / "learning_curves_mean.png")


def plot_accuracy_per_fold(
    config: ReportConfig,
    fold_accuracies: Sequence[float],
) -> None:
    if not fold_accuracies:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    xlabels = [f"Fold {i}" for i in range(1, len(fold_accuracies) + 1)]
    bars = ax.bar(xlabels, fold_accuracies, color="#2E75B6", edgecolor="black", linewidth=0.7)

    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    ax.axhline(y=mean_acc, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_acc:.2f}%")
    if config.show_ssv2_baseline:
        ax.axhline(
            y=SSV2_BASELINE_ACC, color="orange", linestyle="--",
            linewidth=1.5, alpha=0.7, label=f"Baseline: {SSV2_BASELINE_ACC:.2f}%",
        )

    for bar, acc in zip(bars, fold_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title(f"Best Val Acc@1 per Fold\n{config.label}", fontsize=13)
    ax.set_ylabel("Best Val Acc@1 (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_figure(config.output_dir / "accuracy_per_fold.png")


def plot_confusion_matrix(
    config: ReportConfig,
    all_labels: Sequence[int],
    all_preds: Sequence[int],
    fold_accuracies: Sequence[float],
) -> None:
    if not all_labels:
        return

    num_classes = len(config.class_names)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label("Jumlah Sampel", fontsize=10)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(config.class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(config.class_names, fontsize=11)

    threshold = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=24,
            )

    mean_acc = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0.0
    ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontsize=12)
    ax.set_xlabel("Prediksi Model", fontsize=12)
    ax.set_title(
        f"Confusion Matrix - {config.label}\n"
        f"Mean Best Val Acc@1: {mean_acc:.2f}% | Total Sampel: {len(all_labels)}",
        fontsize=13,
    )
    plt.tight_layout()
    _save_figure(config.output_dir / "confusion_matrix.png")


def _save_figure(path: Path) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path.name}")


# ---------------------------------------------------------------------------
# Classification report (text)
# ---------------------------------------------------------------------------

def write_classification_report(
    config: ReportConfig,
    fold_accuracies: Sequence[float],
    all_labels: Sequence[int],
    all_preds: Sequence[int],
    detailed_per_fold: bool = False,
) -> None:
    """Write a human-readable classification report.

    If ``detailed_per_fold`` is True, per-fold MAE / micro / macro metrics
    and the sklearn ``classification_report`` are included for each fold.
    """
    lines: List[str] = []
    lines.append("=" * REPORT_WIDTH)
    lines.append(f"  CLASSIFICATION REPORT - {config.label}")
    for header in config.header_lines:
        lines.append(f"  {header}")
    lines.append("=" * REPORT_WIDTH + "\n")

    if detailed_per_fold and all_labels:
        _append_per_fold_metrics(lines, config, fold_accuracies, all_labels, all_preds)
    else:
        for fold_idx in range(1, config.num_folds + 1):
            best_v = (
                fold_accuracies[fold_idx - 1]
                if fold_idx <= len(fold_accuracies) else 0.0
            )
            lines.append("-" * REPORT_WIDTH)
            lines.append(f"  FOLD {fold_idx}  -  Best Val Acc@1: {best_v:.2f}%")
            lines.append("-" * REPORT_WIDTH + "\n")

    if fold_accuracies:
        mean_acc = sum(fold_accuracies) / len(fold_accuracies)
        lines.append(f"\n{'-' * REPORT_WIDTH}")
        lines.append("  RINGKASAN")
        lines.append(f"{'-' * REPORT_WIDTH}")
        for i, acc in enumerate(fold_accuracies, 1):
            lines.append(f"  Fold {i}  Best Val Acc@1 : {acc:.2f}%")
        lines.append(f"  {'-' * 45}")
        lines.append(f"  Mean Best Val Acc@1  : {mean_acc:.2f}%")
        if config.show_ssv2_baseline:
            lines.append(f"  Baseline (SSV2 orig) : {SSV2_BASELINE_ACC:.2f}%")
            lines.append(f"  Peningkatan          : {mean_acc - SSV2_BASELINE_ACC:+.2f}%")

    lines.append("\n" + "=" * REPORT_WIDTH + "\n")

    full = "\n".join(lines)
    print(full)

    path = config.output_dir / "classification_report_all_folds.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"  [OK] {path.name}")


def _append_per_fold_metrics(
    lines: List[str],
    config: ReportConfig,
    fold_accuracies: Sequence[float],
    all_labels: Sequence[int],
    all_preds: Sequence[int],
) -> None:
    """Populate ``lines`` with MAE / micro / macro / per-class metrics."""
    num_classes = len(config.class_names)

    # Per-fold metrics aggregated from per-fold prediction files.
    for fold_idx in range(1, config.num_folds + 1):
        pred_file = config.fold_pred(fold_idx)
        val_csv = config.folds_dir / f"val_fold_{fold_idx}.csv"
        gt = read_gt_labels(val_csv) if val_csv.exists() else []
        prd = read_predictions(pred_file) if pred_file.exists() else []
        best_v = fold_accuracies[fold_idx - 1] if fold_idx <= len(fold_accuracies) else 0.0

        lines.append("-" * REPORT_WIDTH)
        lines.append(f"  FOLD {fold_idx}")
        lines.append("-" * REPORT_WIDTH)
        lines.append(f"  Best Val Acc@1  : {best_v:.2f}%")

        if not prd or not gt:
            lines.append("  [SKIP] Tidak ada prediksi.\n")
            continue

        n = min(len(gt), len(prd))
        gt, prd = gt[:n], prd[:n]
        lines.append(f"  Jumlah sampel   : {n}\n")

        lines.append(f"  Mean Absolute Error (MAE) : {mean_absolute_error(gt, prd):.4f}")
        _append_average_block(lines, "MICRO AVERAGE", gt, prd, average="micro")
        _append_average_block(lines, "MACRO AVERAGE", gt, prd, average="macro")

        lines.append("\n  [PER CLASS]")
        per_class = classification_report(
            gt, prd, target_names=list(config.class_names),
            labels=list(range(num_classes)), zero_division=0,
        )
        lines.extend("  " + ln for ln in per_class.splitlines())
        lines.append("")

    # Overall (all folds concatenated).
    lines.append("\n" + "=" * REPORT_WIDTH)
    lines.append("  GABUNGAN SEMUA FOLD (Cross-Validation Summary)")
    lines.append("=" * REPORT_WIDTH)
    lines.append(f"\n  Total sampel        : {len(all_labels)}")
    lines.append(f"  Overall Accuracy    : {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    lines.append(f"  Mean Absolute Error : {mean_absolute_error(all_labels, all_preds):.4f}")
    _append_average_block(lines, "MICRO - semua fold digabung", all_labels, all_preds, average="micro")
    _append_average_block(lines, "MACRO - semua fold digabung", all_labels, all_preds, average="macro")

    lines.append("\n  [PER CLASS - semua fold digabung]")
    overall = classification_report(
        all_labels, all_preds, target_names=list(config.class_names),
        labels=list(range(num_classes)), zero_division=0,
    )
    lines.extend("  " + ln for ln in overall.splitlines())


def _append_average_block(
    lines: List[str],
    title: str,
    labels: Sequence[int],
    preds: Sequence[int],
    average: str,
) -> None:
    f1 = f1_score(labels, preds, average=average, zero_division=0) * 100
    precision = precision_score(labels, preds, average=average, zero_division=0) * 100
    recall = recall_score(labels, preds, average=average, zero_division=0) * 100
    if average == "micro":
        acc = accuracy_score(labels, preds) * 100
        lines.append(f"\n  [{title}]")
        lines.append(f"  Accuracy   : {acc:.2f}%")
    else:
        lines.append(f"\n  [{title}]")
    lines.append(f"  Precision  : {precision:.2f}%")
    lines.append(f"  Recall     : {recall:.2f}%")
    lines.append(f"  F1-Score   : {f1:.2f}%")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_reports(
    config: ReportConfig,
    fold_accuracies: Optional[Sequence[float]] = None,
    fold_logs: Optional[Sequence[Sequence[dict]]] = None,
    detailed_per_fold: bool = False,
) -> None:
    """Produce every reporting artefact for one experiment.

    If ``fold_logs`` or ``fold_accuracies`` are omitted they are re-derived
    from files on disk, so this function can also be called after the fact.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if fold_logs is None:
        fold_logs = collect_fold_logs(config)
    if fold_accuracies is None:
        fold_accuracies = best_acc_per_fold(fold_logs)

    _safe_run("learning curves per fold", plot_learning_curves_per_fold, config, fold_logs)
    _safe_run("mean learning curve", plot_mean_learning_curve, config, fold_logs)
    _safe_run("accuracy per fold bar chart", plot_accuracy_per_fold, config, fold_accuracies)

    all_labels, all_preds = collect_fold_predictions(config)
    _safe_run(
        "confusion matrix",
        plot_confusion_matrix,
        config, all_labels, all_preds, fold_accuracies,
    )
    _safe_run(
        "classification report",
        write_classification_report,
        config, fold_accuracies, all_labels, all_preds, detailed_per_fold,
    )


def _safe_run(description: str, func: Callable, *args) -> None:
    """Run a reporting step and continue on failure.

    Reporting is non-critical: a broken plot should not stop the next plot
    from being produced.
    """
    try:
        func(*args)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch at report boundary
        print(f"[WARN] {description} gagal: {exc}")
