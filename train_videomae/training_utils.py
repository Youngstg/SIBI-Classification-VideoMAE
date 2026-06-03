"""Shared training helpers for every VideoMAE script in this package.

The training scripts differ only in their stage configuration (single-stage
vs two-stage, pretrained vs scratch, epoch counts). The subprocess runner,
log parser and checkpoint cleanup are identical, so they live here.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# Regex to catch "Validation Acc@1: 12.34%" lines emitted by VideoMAE.
_VAL_ACC_LINE_RE = re.compile(r"Validation Acc@1:\s*([0-9]+(?:\.[0-9]+)?)%")

# Return codes considered "soft failures": VideoMAE often exits with 1
# because of a DDP barrier race even when training finished successfully.
_ACCEPTABLE_RETURN_CODES = (0, 1)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def banner(text: str, width: int = 62, char: str = "=") -> None:
    """Print a titled banner for section separation in training logs."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


# ---------------------------------------------------------------------------
# Subprocess runner for a VideoMAE training stage
# ---------------------------------------------------------------------------

def _videomae_env() -> dict:
    """Environment used by every VideoMAE subprocess call."""
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["NVIDIA_TF32_OVERRIDE"] = "1"
    return env


def _format_epoch_line(
    stage_name: str,
    epoch: int,
    epochs_total: int,
    acc1: float,
    acc5: float,
    loss: float,
    best_acc: float,
    patience: Optional[int],
    patience_limit: Optional[int],
) -> str:
    base = (
        f"  [{stage_name}] "
        f"EPOCH {epoch + 1:>2}/{epochs_total} | "
        f"Acc@1: {acc1:.2f}% | "
        f"Acc@5: {acc5:.2f}% | "
        f"Loss: {loss:.4f} | "
        f"Best: {best_acc:.2f}%"
    )
    if patience is not None and patience_limit is not None:
        base += f"\n  Patience   : {patience}/{patience_limit}"
    return base


def run_stage(
    cmd: Sequence[str],
    stage_name: str,
    epochs_total: int,
    cwd: str,
    early_stop_patience: Optional[int] = None,
) -> tuple[bool, float]:
    """Run one VideoMAE training stage and stream its output.

    Parses per-epoch JSON metrics emitted by run_class_finetuning.py and
    optionally triggers early stop when val_acc1 plateaus.

    Returns (success, best_val_acc1).
    """
    print(f"\n[CMD] {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=_videomae_env(),
        )
    except OSError as exc:
        print(f"[ERROR] Gagal memulai proses: {exc}")
        return False, 0.0

    # best_acc  : nilai terbaik yang pernah tercatat (dari regex ATAU JSON),
    #             digunakan sebagai return value dan tampilan.
    # best_patience_acc : nilai terbaik yang ditracking KHUSUS dari JSON
    #             epoch summary, digunakan untuk menghitung patience.
    #             Dipisah agar update regex "Validation Acc@1: X%" yang
    #             datang *sebelum* baris JSON pada epoch yang sama tidak
    #             membuat acc1 == best_patience_acc sehingga patience selalu
    #             naik meski epoch baru saja improve.
    best_acc = 0.0
    best_patience_acc = 0.0
    no_improve_count = 0
    early_stopped = False

    for line in process.stdout:
        print(line, end="")
        stripped = line.strip()

        # Live display: update best_acc dari baris "Validation Acc@1: X%"
        # agar angka terbaru terlihat sebelum JSON summary tiba.
        # Tidak dipakai untuk patience — lihat best_patience_acc di bawah.
        match = _VAL_ACC_LINE_RE.search(stripped)
        if match:
            acc_from_line = float(match.group(1))
            if acc_from_line > best_acc:
                best_acc = acc_from_line

        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if "val_acc1" not in data or "epoch" not in data:
            continue

        epoch = data["epoch"]
        acc1 = data["val_acc1"]
        acc5 = data.get("val_acc5", 0)
        loss = data.get("val_loss", 0)

        # Sync best_acc dengan JSON (JSON adalah sumber kebenaran).
        if acc1 > best_acc:
            best_acc = acc1

        # Patience dihitung HANYA dari JSON, bukan dari regex line,
        # sehingga epoch yang improve benar-benar mereset counter.
        if acc1 > best_patience_acc:
            best_patience_acc = acc1
            no_improve_count = 0
        else:
            no_improve_count += 1

        print("\n" + "-" * 60)
        print(_format_epoch_line(
            stage_name, epoch, epochs_total, acc1, acc5, loss, best_acc,
            no_improve_count if early_stop_patience is not None else None,
            early_stop_patience,
        ))
        print("-" * 60 + "\n")

        if (
            early_stop_patience is not None
            and no_improve_count >= early_stop_patience
        ):
            print(
                f"[EARLY STOP] {stage_name}: val_acc1 tidak membaik "
                f"selama {early_stop_patience} epoch berturut-turut."
            )
            process.terminate()
            early_stopped = True
            break

    process.wait()
    rc = process.returncode

    if early_stopped:
        return True, best_acc

    if rc not in _ACCEPTABLE_RETURN_CODES:
        print(f"\n[ERROR] {stage_name} gagal (return code {rc})")
        return False, best_acc

    return True, best_acc


# ---------------------------------------------------------------------------
# Log + prediction parsers
# ---------------------------------------------------------------------------

def read_log(log_file: Path | str) -> List[dict]:
    """Parse VideoMAE log.txt into list of per-epoch dictionaries.
    
    Extracts JSON lines containing epoch metrics (val_acc1, val_acc5, val_loss).
    Returns sorted list by epoch number.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return []

    data_list: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_acc1" in entry and "epoch" in entry:
                data_list.append(entry)

    data_list.sort(key=lambda row: row["epoch"])
    return data_list


def best_acc_from_log(log_file: Path | str) -> float:
    """Extract maximum validation accuracy from log file."""
    return max((row["val_acc1"] for row in read_log(log_file)), default=0.0)


def read_gt_labels(val_csv: Path | str) -> List[int]:
    """Read ground truth labels from validation CSV file.
    
    Format: space-delimited with path and label columns.
    Extracts label (last column, converted to int).
    """
    labels: List[int] = []
    with open(val_csv, "r") as f:
        for row in csv.reader(f, delimiter=" "):
            if len(row) < 2:
                continue
            try:
                labels.append(int(row[-1]))
            except ValueError:
                continue
    return labels


def read_predictions(pred_file: Path | str) -> List[int]:
    """Read argmax class predictions from a VideoMAE 0.txt file.

    Lines look like: ``0,0.123,[0.1, 0.2, ...]`` where the bracketed list
    holds class scores. Lines starting with ``0.0,`` or ``0,`` are metadata
    and skipped.
    """
    import numpy as np  # Local import: parsing predictions is optional work.

    pred_path = Path(pred_file)
    if not pred_path.exists():
        return []

    preds: List[int] = []
    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("0.0,") or line.startswith("0,"):
                continue
            try:
                bracket_start = line.index("[")
                bracket_end = line.index("]")
            except ValueError:
                continue
            scores_str = line[bracket_start + 1:bracket_end]
            try:
                scores = [float(s.strip()) for s in scores_str.split(",")]
            except ValueError:
                continue
            preds.append(int(np.argmax(scores)))
    return preds


# ---------------------------------------------------------------------------
# Checkpoint / label-map utilities
# ---------------------------------------------------------------------------

def cleanup_checkpoints(fold_dir: Path | str) -> None:
    """Delete all intermediate checkpoints, keep only checkpoint-best.pth.
    
    Saves disk space after training completion by removing epoch checkpoints.
    Only the best checkpoint is retained for inference/resuming.
    """
    deleted = 0
    for ckpt in Path(fold_dir).glob("checkpoint-*.pth"):
        if ckpt.name != "checkpoint-best.pth":
            ckpt.unlink()
            deleted += 1
    if deleted > 0:
        print(f"[INFO] Hapus {deleted} checkpoint intermediate di {fold_dir}")


def load_class_names(folds_dir: Path | str) -> List[str]:
    """Load class names from label_map.json in folds directory.
    
    Supports both formats:
    - Dict: {"labels": {"Aku": 0, "Bapak": 1, ...}}
    - List: ["Aku", "Bapak", ...]
    
    Exits with error if file not found.
    """
    label_map_path = Path(folds_dir) / "label_map.json"
    if not label_map_path.exists():
        print(f"[ERROR] label_map.json tidak ditemukan: {label_map_path}")
        sys.exit(1)

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    labels = label_data["labels"]
    if isinstance(labels, dict):
        return [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    return list(labels)


def ensure_paths_exist(paths: Iterable[tuple[Path, str]]) -> None:
    """Exit with a clear error if any required (path, name) is missing."""
    for path, name in paths:
        if not path.exists():
            print(f"[ERROR] Folder '{name}' tidak ditemukan: {path}")
            sys.exit(1)
