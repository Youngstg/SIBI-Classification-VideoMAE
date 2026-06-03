"""Fine-tune Hugging Face ViViT on the local SIBI folds dataset.

Run from the project root:
    python train_vivit/finetune_vivit.py --start_fold 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import VivitForVideoClassification

try:
    from decord import VideoReader, cpu
except ImportError as exc:  # pragma: no cover - import error is environment-specific
    raise ImportError(
        "decord is required to read .mp4 videos. Install dependencies from requirements.txt first."
    ) from exc


_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT.parent))

from train_videomae.evaluation import ReportConfig, generate_reports
from train_videomae.training_utils import banner, ensure_paths_exist, load_class_names, read_log

PROJECT_ROOT = _PACKAGE_ROOT.parent
FOLDS_DIR = PROJECT_ROOT / "folds_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "vivit"


@dataclass(frozen=True)
class VideoSample:
    path: Path
    label: int


@dataclass
class VivitConfig:
    model_name: str = "google/vivit-b-16x2-kinetics400"
    label: str = "ViViT-B/16x2 | Kinetics400 -> SIBI"
    output_base: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    num_frames: int = 16
    image_size: int = 224
    patch_size: int = 16

    batch_size: int = 2
    num_workers: int = 4
    grad_accum_steps: int = 1

    # PDF reference:
    # - Learning Rate (ViViT): 0.001, 0.0001
    # - Batch Size (ViViT): 2
    # - Optimizer: Adam
    # The PDF does not pin one final LR among the listed ViViT candidates, so
    # the default keeps 1e-4. Unspecified knobs stay on the existing recipe.
    epochs: int = 55
    warmup_epochs: float = 2.5
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    dropout: float = 0.0
    stochastic_depth: float = 0.1
    early_stop_patience: Optional[int] = 5

    num_folds: int = 5
    hf_cache_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "hf_cache")
    local_files_only: bool = False


CONFIG = VivitConfig()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def configure_hf_cache(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_dir = cache_dir / "hub"
    assets_dir = cache_dir / "assets"
    hub_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_ASSETS_CACHE"] = str(assets_dir)
    return hub_dir


def discover_fold_indices(folds_dir: Path) -> List[int]:
    fold_indices: List[int] = []
    for csv_path in sorted(folds_dir.glob("train_fold_*.csv")):
        suffix = csv_path.stem.rsplit("_", maxsplit=1)[-1]
        if not suffix.isdigit():
            continue
        fold_idx = int(suffix)
        if (folds_dir / f"val_fold_{fold_idx}.csv").exists():
            fold_indices.append(fold_idx)
    return fold_indices


def discover_class_names(folds_dir: Path) -> List[str]:
    videos_root = folds_dir / "videos"
    folder_names = sorted(path.name for path in videos_root.iterdir() if path.is_dir())

    label_map_path = folds_dir / "label_map.json"
    if label_map_path.exists():
        mapped_names = load_class_names(folds_dir)
        if not folder_names or set(mapped_names) == set(folder_names):
            return mapped_names

    if folder_names:
        return folder_names

    raise FileNotFoundError(f"Tidak menemukan folder kelas di: {videos_root}")


def load_fold_samples(csv_path: Path, folds_dir: Path) -> List[VideoSample]:
    samples: List[VideoSample] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter=" "):
            if len(row) < 2:
                continue
            relative_path = row[0].strip()
            label = int(row[-1])
            video_path = folds_dir / relative_path
            if not video_path.exists():
                raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")
            samples.append(VideoSample(path=video_path, label=label))

    if not samples:
        raise ValueError(f"Tidak ada sampel valid di {csv_path}")
    return samples


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VideoSample],
        num_frames: int,
        image_size: int,
        training: bool,
    ) -> None:
        self.samples = list(samples)
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        for _ in range(3):
            try:
                pixel_values = self._load_video(sample.path)
                return {
                    "pixel_values": pixel_values,
                    "labels": torch.tensor(sample.label, dtype=torch.long),
                }
            except Exception:
                sample = self.samples[random.randrange(len(self.samples))]

        raise RuntimeError(f"Gagal membaca video setelah 3 percobaan: {self.samples[index].path}")

    def _sample_frame_indices(self, total_frames: int) -> np.ndarray:
        if total_frames <= 0:
            raise ValueError("Video kosong atau gagal dibaca.")

        if total_frames <= self.num_frames:
            return np.linspace(0, total_frames - 1, num=self.num_frames).clip(0, total_frames - 1).astype(np.int64)

        if not self.training:
            return np.linspace(0, total_frames - 1, num=self.num_frames).astype(np.int64)

        boundaries = np.linspace(0, total_frames, num=self.num_frames + 1).astype(np.int64)
        indices = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            high = max(start + 1, end)
            indices.append(np.random.randint(start, high))
        return np.asarray(indices, dtype=np.int64)

    def _load_video(self, path: Path) -> torch.Tensor:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        indices = self._sample_frame_indices(len(reader))
        frames = reader.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames = F.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        frames = (frames - 0.5) / 0.5
        return frames.contiguous()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states

        keep_prob = 1.0 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor.floor_()
        return hidden_states.div(keep_prob) * random_tensor


def enable_stochastic_depth(model: VivitForVideoClassification, rate: float) -> None:
    if rate <= 0:
        return

    layers = list(model.vivit.encoder.layer)
    if not layers:
        return

    depth = len(layers)
    for layer_idx, layer in enumerate(layers):
        layer_rate = rate * layer_idx / max(depth - 1, 1)
        layer.drop_path1 = DropPath(layer_rate)
        layer.drop_path2 = DropPath(layer_rate)

        def _forward(self, hidden_states, head_mask=None, output_attentions=False):
            self_attention_outputs = self.attention(
                self.layernorm_before(hidden_states),
                head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]

            hidden_states = self.drop_path1(attention_output) + hidden_states

            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output.dense(layer_output)
            layer_output = self.output.dropout(layer_output)
            layer_output = self.drop_path2(layer_output) + hidden_states

            outputs = (layer_output,) + outputs
            return outputs

        layer.forward = MethodType(_forward, layer)


def reset_classifier(model: VivitForVideoClassification, class_names: Sequence[str]) -> None:
    classifier = nn.Linear(model.classifier.in_features, len(class_names))
    model.classifier = classifier
    model._init_weights(model.classifier)
    model.num_labels = len(class_names)
    model.config.num_labels = len(class_names)
    model.config.id2label = {idx: name for idx, name in enumerate(class_names)}
    model.config.label2id = {name: idx for idx, name in enumerate(class_names)}


def create_model(config: VivitConfig, class_names: Sequence[str], hub_cache_dir: Path) -> VivitForVideoClassification:
    try:
        model = VivitForVideoClassification.from_pretrained(
            config.model_name,
            cache_dir=str(hub_cache_dir),
            local_files_only=config.local_files_only,
        )
    except Exception as exc:  # noqa: BLE001 - download/cache errors vary by environment
        raise RuntimeError(
            "Gagal memuat pretrained ViViT. Pastikan model dapat diunduh atau sudah ada di cache lokal "
            f"({hub_cache_dir})."
        ) from exc

    reset_classifier(model, class_names)
    model.config.num_frames = config.num_frames
    model.config.image_size = config.image_size
    model.config.tubelet_size = [2, config.patch_size, config.patch_size]
    model.config.hidden_dropout_prob = config.dropout
    model.config.attention_probs_dropout_prob = config.dropout

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = config.dropout

    enable_stochastic_depth(model, config.stochastic_depth)
    return model


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def create_scheduler(optimizer: Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: Sequence[int] = (1, 5)) -> List[float]:
    if logits.ndim != 2:
        raise ValueError(f"Logits harus 2D, dapat: {tuple(logits.shape)}")

    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    accuracies: List[float] = []
    for k in topk:
        k = min(k, logits.shape[1])
        acc = correct[:k].any(dim=0).float().mean().item() * 100.0
        accuracies.append(acc)
    return accuracies


def train_one_epoch(
    model: VivitForVideoClassification,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: VivitForVideoClassification,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    top5_total = 0.0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(pixel_values=pixel_values, labels=labels)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        acc1, acc5 = topk_accuracy(logits, labels)

        total_loss += outputs.loss.item() * batch_size
        total_samples += batch_size
        top5_total += acc5 * batch_size

        preds = logits.argmax(dim=-1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds) * 100.0 if all_labels else 0.0
    return {
        "val_loss": total_loss / max(1, total_samples),
        "val_acc1": accuracy,
        "val_acc5": top5_total / max(1, total_samples),
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def save_checkpoint(
    path: Path,
    model: VivitForVideoClassification,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    best_acc1: float,
) -> None:
    payload = {
        "epoch": epoch,
        "best_acc1": best_acc1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config.to_dict(),
    }
    torch.save(payload, path)


def write_predictions(path: Path, labels: Sequence[int], probs: Sequence[Sequence[float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for index, (label, scores) in enumerate(zip(labels, probs), start=1):
            score_text = ", ".join(f"{score:.8f}" for score in scores)
            handle.write(f"{index},{label},[{score_text}]\n")


def append_log(log_path: Path, epoch_metrics: dict[str, float]) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(epoch_metrics) + "\n")


def train_fold(
    config: VivitConfig,
    fold_idx: int,
    class_names: Sequence[str],
    device: torch.device,
    hub_cache_dir: Path,
) -> float:
    train_csv = FOLDS_DIR / f"train_fold_{fold_idx}.csv"
    val_csv = FOLDS_DIR / f"val_fold_{fold_idx}.csv"
    train_samples = load_fold_samples(train_csv, FOLDS_DIR)
    val_samples = load_fold_samples(val_csv, FOLDS_DIR)

    fold_dir = config.output_base / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "log.txt"
    pred_path = fold_dir / "0.txt"
    best_ckpt_path = fold_dir / "checkpoint-best.pth"
    last_ckpt_path = fold_dir / "checkpoint-last.pth"

    if log_path.exists():
        log_path.unlink()
    for stale_path in (pred_path, best_ckpt_path, last_ckpt_path):
        if stale_path.exists():
            stale_path.unlink()

    train_loader = build_dataloader(
        VideoFolderDataset(train_samples, config.num_frames, config.image_size, training=True),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        VideoFolderDataset(val_samples, config.num_frames, config.image_size, training=False),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    model = create_model(config, class_names, hub_cache_dir).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(1, config.grad_accum_steps))
    total_steps = config.epochs * updates_per_epoch
    warmup_steps = math.ceil(config.warmup_epochs * updates_per_epoch)
    scheduler = create_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc1 = 0.0
    best_probs: List[List[float]] = []
    best_labels: List[int] = []
    epochs_without_improvement = 0

    banner(f"FOLD {fold_idx} - ViViT", char="-")
    print(f"  Train samples : {len(train_samples)}")
    print(f"  Val samples   : {len(val_samples)}")
    print(f"  Output        : {fold_dir}")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=config.grad_accum_steps,
        )
        val_metrics = evaluate(model=model, loader=val_loader, device=device)
        current_acc1 = float(val_metrics["val_acc1"])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_metrics["val_loss"]), 6),
            "val_acc1": round(current_acc1, 4),
            "val_acc5": round(float(val_metrics["val_acc5"]), 4),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 10),
        }
        append_log(log_path, epoch_metrics)
        print(json.dumps(epoch_metrics))

        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_acc1)

        if current_acc1 > best_acc1:
            best_acc1 = current_acc1
            best_probs = list(val_metrics["probs"])
            best_labels = list(val_metrics["labels"])
            epochs_without_improvement = 0
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_acc1)
            write_predictions(pred_path, best_labels, best_probs)
        else:
            epochs_without_improvement += 1

        print(
            f"  [Fold {fold_idx}] Epoch {epoch + 1:02d}/{config.epochs} | "
            f"Val Acc@1: {current_acc1:.2f}% | Best: {best_acc1:.2f}% | "
            f"Patience: {epochs_without_improvement}/"
            f"{config.early_stop_patience if config.early_stop_patience is not None else '-'}"
        )

        if (
            config.early_stop_patience is not None
            and epochs_without_improvement >= config.early_stop_patience
        ):
            print(
                f"[EARLY STOP] Fold {fold_idx}: val_acc1 tidak membaik selama "
                f"{config.early_stop_patience} epoch."
            )
            break

    if best_probs and best_labels:
        write_predictions(pred_path, best_labels, best_probs)

    return best_acc1


def report_config_for(config: VivitConfig, class_names: Sequence[str], num_folds: int) -> ReportConfig:
    return ReportConfig(
        output_dir=config.output_base,
        folds_dir=FOLDS_DIR,
        label=config.label,
        num_folds=num_folds,
        class_names=class_names,
        header_lines=[
            f"Model: {config.model_name}",
            (
                "Optimizer: Adam | LR=1e-4 (PDF lists 1e-3/1e-4) | "
                "Epochs=55 | Warmup=2.5 ep | Cosine decay | EarlyStop=5"
            ),
            (
                f"Frames={config.num_frames} | ImageSize={config.image_size} | "
                f"Patch={config.patch_size} | Dropout={config.dropout} | "
                f"StochasticDepth={config.stochastic_depth}"
            ),
        ],
        show_ssv2_baseline=False,
    )


def print_run_header(config: VivitConfig, class_names: Sequence[str], fold_indices: Sequence[int], device: torch.device) -> None:
    banner("ViViT FINE-TUNING")
    print(f"  Model        : {config.model_name}")
    print(f"  Device       : {device}")
    print(f"  Output       : {config.output_base}")
    print(f"  Classes      : {len(class_names)} ({', '.join(class_names)})")
    print(f"  Folds        : {list(fold_indices)}")
    print(f"  Frames       : {config.num_frames}")
    print(f"  Image size   : {config.image_size}")
    print(f"  Patch size   : {config.patch_size}")
    print(f"  Batch size   : {config.batch_size}")
    print(f"  Num workers  : {config.num_workers}")
    print(f"  Epochs       : {config.epochs}")
    print(f"  LR           : {config.learning_rate}")
    print(f"  Warmup       : {config.warmup_epochs}")
    print(f"  Weight decay : {config.weight_decay}")
    print(f"  Dropout      : {config.dropout}")
    print(f"  Stoch depth  : {config.stochastic_depth}")
    print(f"  Early stop   : {config.early_stop_patience}")


def run(config: VivitConfig, start_fold: int = 1) -> None:
    ensure_paths_exist([
        (FOLDS_DIR, "folds_data"),
        (FOLDS_DIR / "videos", "folds_data/videos"),
    ])

    if not torch.cuda.is_available():
        raise RuntimeError("Script ini membutuhkan satu CUDA GPU, tetapi CUDA tidak tersedia.")

    seed_everything()
    configure_runtime()
    hub_cache_dir = configure_hf_cache(config.hf_cache_dir)

    class_names = discover_class_names(FOLDS_DIR)
    fold_indices = discover_fold_indices(FOLDS_DIR)
    if not fold_indices:
        raise RuntimeError("Tidak menemukan pasangan train_fold_*.csv dan val_fold_*.csv di folds_data.")

    fold_indices = [fold for fold in fold_indices if fold >= start_fold]
    if not fold_indices:
        raise ValueError(f"Tidak ada fold yang dimulai dari start_fold={start_fold}.")

    num_folds = len(discover_fold_indices(FOLDS_DIR))
    device = torch.device("cuda:0")

    config.output_base.mkdir(parents=True, exist_ok=True)
    print_run_header(config, class_names, fold_indices, device)

    fold_accuracies: List[float] = []
    fold_logs: List[List[dict]] = []

    for prior_fold in range(1, start_fold):
        prior_log = config.output_base / f"fold_{prior_fold}" / "log.txt"
        epochs_data = read_log(prior_log)
        fold_logs.append(epochs_data)
        fold_accuracies.append(max((row["val_acc1"] for row in epochs_data), default=0.0))

    for fold_idx in fold_indices:
        best_acc = train_fold(config, fold_idx, class_names, device, hub_cache_dir)
        fold_accuracies.append(best_acc)
        fold_logs.append(read_log(config.output_base / f"fold_{fold_idx}" / "log.txt"))

    banner("SELESAI - RINGKASAN HASIL")
    for idx, acc in enumerate(fold_accuracies, start=1):
        print(f"  Fold {idx} Best Acc@1 : {acc:.2f}%")

    valid_accs = [acc for acc in fold_accuracies if acc > 0]
    if valid_accs:
        print("  " + "-" * 40)
        print(f"  Mean Acc@1          : {sum(valid_accs) / len(valid_accs):.2f}%")

    print("\n[INFO] Membuat laporan...")
    generate_reports(
        config=report_config_for(config, class_names, num_folds=num_folds),
        fold_accuracies=fold_accuracies,
        fold_logs=fold_logs,
        detailed_per_fold=True,
    )

    banner("SEMUA SELESAI")
    print(f"  Output: {config.output_base}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViViT-B/16x2 on the local folds_data dataset")
    parser.add_argument(
        "--start_fold",
        type=int,
        default=1,
        help="Mulai dari fold ke-N (default: 1).",
    )
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--num_workers", type=int, default=CONFIG.num_workers)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Gunakan hanya cache lokal Hugging Face, tanpa mencoba download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VivitConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        local_files_only=args.local_files_only,
    )
    run(config=config, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
