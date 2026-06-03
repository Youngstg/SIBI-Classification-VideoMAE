"""
train_sibi_wlasl_i3d.ddp.py
===========================
Training SIBI 5-fold dengan model InceptionI3d asli WLASL.

Script ini mempertahankan pipeline fold/report dari train_sibi.ddp(p).py,
tetapi kembali memakai model bawaan repo WLASL (pytorch_i3d.InceptionI3d)
tanpa model tambahan seperti resnet_i3d.py.
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =====================================================================
# KONFIGURASI PATH
# =====================================================================
I3D_DIR = Path(__file__).resolve().parent
WLASL_ROOT = I3D_DIR.parents[1]
PROJECT_ROOT = I3D_DIR.parents[2]

FOLDS_DIR = PROJECT_ROOT / "folds_data"
VIDEOS_DIR = FOLDS_DIR / "videos"
PREPROCESS_DIR = I3D_DIR / "preprocess"
OUTPUT_BASE = PROJECT_ROOT / "work_dirs" / "i3d_wlasl_ddp"
RGB_PRETRAINED = I3D_DIR / "weights" / "rgb_imagenet.pt"

if str(I3D_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(I3D_DIR))

import videotransforms  # noqa: E402
from datasets.nslt_dataset import NSLT as Dataset  # noqa: E402
from pytorch_i3d import InceptionI3d  # noqa: E402


# =====================================================================
# KONFIGURASI TRAINING
# =====================================================================
MODEL_LABEL = "WLASL-I3D | RGB | DDP"
NUM_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 2
NUM_WORKERS = 4
FRAME_SIZE = 224
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 5
LR_FACTOR = 0.3
SAVE_FREQ = 10
ES_PATIENCE = 5


# =====================================================================
# DDP UTILITY
# =====================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def reduce_tensor(tensor):
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


# =====================================================================
# MODEL
# =====================================================================
def build_training_model(num_classes):
    model = InceptionI3d(400, in_channels=3)
    if RGB_PRETRAINED.exists():
        if is_main_process():
            print(f"[INFO] Load pretrained I3D: {RGB_PRETRAINED}")
        state_dict = torch.load(RGB_PRETRAINED, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        if is_main_process():
            print(f"[WARN] Pretrained I3D tidak ditemukan: {RGB_PRETRAINED}")
            print("[WARN] Training akan berjalan dari scratch.")
    model.replace_logits(num_classes)
    return model


def build_inference_model(num_classes, checkpoint_path, local_rank):
    model = InceptionI3d(num_classes, in_channels=3)
    state_dict = torch.load(
        checkpoint_path,
        map_location=f"cuda:{local_rank}",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.to(local_rank)


# =====================================================================
# TRAIN / VAL SATU EPOCH
# =====================================================================
def run_epoch(model, loader, optimizer, scaler, device, phase):
    is_train = phase == "train"
    model.train(is_train)

    running_loss = 0.0
    running_loc_loss = 0.0
    running_cls_loss = 0.0
    corrects = 0
    total = 0

    for data in loader:
        if data == -1:
            continue

        inputs, labels, _ = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        time_steps = inputs.size(2)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            per_frame_logits = model(inputs, pretrained=False)
            per_frame_logits = F.interpolate(
                per_frame_logits,
                size=time_steps,
                mode="linear",
                align_corners=False,
            )

            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            cls_logits = torch.max(per_frame_logits, dim=2)[0]
            cls_targets = torch.max(labels, dim=2)[0]
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
            loss = 0.5 * (loc_loss + cls_loss)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = torch.argmax(cls_logits, dim=1)
        targets = torch.argmax(cls_targets, dim=1)
        batch_size = inputs.size(0)

        running_loss += loss.item() * batch_size
        running_loc_loss += loc_loss.item() * batch_size
        running_cls_loss += cls_loss.item() * batch_size
        corrects += (preds == targets).sum().item()
        total += batch_size

    if total == 0:
        return None

    metrics = {
        "loss": reduce_tensor(torch.tensor(running_loss / total, device=device)).item(),
        "loc_loss": reduce_tensor(torch.tensor(running_loc_loss / total, device=device)).item(),
        "cls_loss": reduce_tensor(torch.tensor(running_cls_loss / total, device=device)).item(),
        "acc": reduce_tensor(torch.tensor(corrects / total * 100.0, device=device)).item(),
    }
    return metrics


# =====================================================================
# TRAINING SATU FOLD
# =====================================================================
def train_fold(fold_idx, num_classes, local_rank):
    train_split = str(PREPROCESS_DIR / f"sibi_train_fold{fold_idx}.json")
    val_split = str(PREPROCESS_DIR / f"sibi_val_fold{fold_idx}.json")

    fold_output_dir = OUTPUT_BASE / f"fold_{fold_idx}"
    if is_main_process():
        fold_output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    log_file = fold_output_dir / "log.txt"
    if not Path(train_split).exists():
        if is_main_process():
            print(f"[SKIP] JSON fold {fold_idx} tidak ditemukan.")
        return 0.0

    if is_main_process():
        print(f"  Device  : cuda:{local_rank} (rank {dist.get_rank()})")
        print(f"  Train   : {train_split}")
        print(f"  Val     : {val_split}")

    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(FRAME_SIZE),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(FRAME_SIZE),
    ])

    root = {"word": str(VIDEOS_DIR)}
    train_dataset = Dataset(train_split, "train", root, "rgb", train_transforms)
    val_dataset = Dataset(val_split, "test", root, "rgb", test_transforms)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_training_model(num_classes).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=LR_PATIENCE,
        factor=LR_FACTOR,
    )
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    best_ep = 0
    es_counter = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            local_rank,
            "train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            local_rank,
            "test",
        )

        if train_metrics is None or val_metrics is None:
            continue

        if is_main_process():
            print(
                f"Epoch {epoch+1}/{EPOCHS} | TRAIN | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Loc: {train_metrics['loc_loss']:.4f} | "
                f"Cls: {train_metrics['cls_loss']:.4f} | "
                f"Acc: {train_metrics['acc']:.2f}%"
            )
            print(
                f"Epoch {epoch+1}/{EPOCHS} | TEST  | "
                f"Loss: {val_metrics['loss']:.4f} | "
                f"Loc: {val_metrics['loc_loss']:.4f} | "
                f"Cls: {val_metrics['cls_loss']:.4f} | "
                f"Acc: {val_metrics['acc']:.2f}%"
            )

        scheduler.step(val_metrics["loss"])

        if is_main_process():
            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                best_ep = epoch + 1
                torch.save(
                    model.module.state_dict(),
                    fold_output_dir / "checkpoint-best.pth",
                )
                print(f"  * Best: {best_acc:.2f}%")

            if (epoch + 1) % SAVE_FREQ == 0:
                torch.save(
                    model.module.state_dict(),
                    fold_output_dir / f"checkpoint-{epoch+1}.pth",
                )

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": round(train_metrics["loss"], 4),
                "train_acc1": round(train_metrics["acc"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_acc1": round(val_metrics["acc"], 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry) + "\n")

            if best_ep == epoch + 1:
                es_counter = 0
            else:
                es_counter += 1
                print(f"  [ES] Tidak ada peningkatan: {es_counter}/{ES_PATIENCE}")

            if es_counter >= ES_PATIENCE:
                print(f"  [EARLY STOP] Berhenti di epoch {epoch+1}.")

        stop_signal = torch.tensor(
            1 if (is_main_process() and es_counter >= ES_PATIENCE) else 0,
            device=local_rank,
        )
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    if is_main_process():
        for checkpoint in fold_output_dir.glob("checkpoint-*.pth"):
            if checkpoint.name != "checkpoint-best.pth":
                checkpoint.unlink()
        print(f"[RESULT] Fold {fold_idx} | Best Val Acc@1: {best_acc:.2f}% (epoch {best_ep})")

    dist.barrier()
    return best_acc


# =====================================================================
# PREDIKSI
# =====================================================================
def generate_predictions(fold_idx, num_classes, local_rank):
    if not is_main_process():
        return [], []

    val_split = str(PREPROCESS_DIR / f"sibi_val_fold{fold_idx}.json")
    fold_output_dir = OUTPUT_BASE / f"fold_{fold_idx}"
    checkpoint_path = fold_output_dir / "checkpoint-best.pth"
    pred_file = fold_output_dir / "predictions.json"

    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint tidak ada: {checkpoint_path}")
        return [], []

    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(FRAME_SIZE),
    ])
    root = {"word": str(VIDEOS_DIR)}
    val_dataset = Dataset(val_split, "test", root, "rgb", test_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_inference_model(num_classes, checkpoint_path, local_rank)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in val_loader:
            if data == -1:
                continue

            inputs, labels, _ = data
            inputs = inputs.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)

            outputs = model(inputs, pretrained=False)
            outputs = F.interpolate(
                outputs,
                size=inputs.size(2),
                mode="linear",
                align_corners=False,
            )
            cls_logits = torch.max(outputs, dim=2)[0]
            cls_targets = torch.max(labels, dim=2)[0]

            preds = torch.argmax(cls_logits, dim=1)
            targets = torch.argmax(cls_targets, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(targets.cpu().numpy().tolist())

    with open(pred_file, "w", encoding="utf-8") as handle:
        json.dump({"preds": all_preds, "labels": all_labels}, handle)

    print(f"  Prediksi fold {fold_idx}: {len(all_preds)} sampel")
    return all_labels, all_preds


# =====================================================================
# BACA LOG
# =====================================================================
def read_log(log_file):
    data_list = []
    if not Path(log_file).exists():
        return data_list

    with open(log_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_acc1" in item and "epoch" in item:
                data_list.append(item)

    data_list.sort(key=lambda x: x["epoch"])
    return data_list


# =====================================================================
# LAPORAN
# =====================================================================
def generate_reports(fold_accuracies, class_names):
    if not is_main_process():
        return

    output_dir = OUTPUT_BASE
    num_classes = len(class_names)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        print(f"[WARN] Library kurang: {exc}")
        return

    fold_log_data = []
    all_labels_all = []
    all_preds_all = []
    fold_gt = []
    fold_pred = []
    has_preds = False
    all_val_acc = []
    all_val_loss = []

    for fold_idx in range(1, NUM_FOLDS + 1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        log_data = read_log(fold_dir / "log.txt")
        fold_log_data.append(log_data)
        if log_data:
            all_val_acc.append([d["val_acc1"] for d in log_data])
            all_val_loss.append([d["val_loss"] for d in log_data])

        pred_path = fold_dir / "predictions.json"
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as handle:
                pred_data = json.load(handle)
            labels = pred_data.get("labels", [])
            preds = pred_data.get("preds", [])
            fold_gt.append(labels)
            fold_pred.append(preds)
            all_labels_all.extend(labels)
            all_preds_all.extend(preds)
            has_preds = has_preds or bool(preds)
        else:
            fold_gt.append([])
            fold_pred.append([])

    try:
        for fold_idx, log_data in enumerate(fold_log_data, 1):
            if not log_data:
                continue
            epochs = [d["epoch"] for d in log_data]
            val_acc = [d["val_acc1"] for d in log_data]
            val_loss = [d["val_loss"] for d in log_data]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(epochs, val_acc, "b-", lw=2)
            ax1.set_title(f"Fold {fold_idx} - Validation Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Acc@1 (%)")
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, val_loss, "r-", lw=2)
            ax2.set_title(f"Fold {fold_idx} - Validation Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f"learning_curve_fold_{fold_idx}.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        print(f"[WARN] Plot learning curve gagal: {exc}")

    try:
        if all_val_acc:
            max_len = max(len(curve) for curve in all_val_acc)

            def pad_curve(curve):
                return curve + [curve[-1]] * (max_len - len(curve))

            mean_acc = np.array([pad_curve(c) for c in all_val_acc]).mean(0)
            std_acc = np.array([pad_curve(c) for c in all_val_acc]).std(0)
            mean_loss = np.array([pad_curve(c) for c in all_val_loss]).mean(0)
            std_loss = np.array([pad_curve(c) for c in all_val_loss]).std(0)
            epochs = list(range(1, max_len + 1))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(epochs, mean_acc, "b-", lw=2, label="Mean Val Acc@1")
            ax1.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color="blue", alpha=0.2)
            ax1.set_title("Mean Validation Accuracy (5-Fold CV)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Acc@1 (%)")
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, mean_loss, "r-", lw=2, label="Mean Val Loss")
            ax2.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color="red", alpha=0.2)
            ax2.set_title("Mean Validation Loss (5-Fold CV)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "learning_curves_mean.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        print(f"[WARN] Mean learning curve gagal: {exc}")

    try:
        if fold_accuracies:
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = [f"Fold {idx}" for idx in range(1, len(fold_accuracies) + 1)]
            bars = ax.bar(labels, fold_accuracies, color="#C00000", edgecolor="black", linewidth=0.7)
            mean_acc = sum(fold_accuracies) / len(fold_accuracies)
            ax.axhline(y=mean_acc, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_acc:.2f}%")
            for bar, acc in zip(bars, fold_accuracies):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{acc:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_title(f"Best Val Acc@1 per Fold\n{MODEL_LABEL}")
            ax.set_ylabel("Best Val Acc@1 (%)")
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "accuracy_per_fold.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        print(f"[WARN] Bar chart gagal: {exc}")

    try:
        if has_preds:
            cm = confusion_matrix(all_labels_all, all_preds_all, labels=list(range(num_classes)))
            fig, ax = plt.subplots(figsize=(13, 11))
            image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
            plt.colorbar(image, ax=ax).set_label("Jumlah Sampel", fontsize=10)
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
            ax.set_yticklabels(class_names, fontsize=11)
            threshold = cm.max() / 2.0 if cm.size else 0.0
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > threshold else "black",
                        fontsize=10,
                    )
            mean_best = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0.0
            ax.set_ylabel("Label Sebenarnya")
            ax.set_xlabel("Prediksi Model")
            ax.set_title(
                f"Confusion Matrix - {MODEL_LABEL}\n"
                f"Mean Best Val Acc@1: {mean_best:.2f}% | Total Sampel: {len(all_labels_all)}"
            )
            plt.tight_layout()
            plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        print(f"[WARN] Confusion matrix gagal: {exc}")

    try:
        width = 72
        report_lines = []
        report_lines.append("=" * width)
        report_lines.append(f"  CLASSIFICATION REPORT - {MODEL_LABEL}")
        report_lines.append("  Protokol : WLASL I3D asli + adaptasi dataset SIBI")
        report_lines.append(f"  Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Batch/GPU: {BATCH_SIZE}")
        report_lines.append("=" * width + "\n")

        if has_preds:
            overall_acc = accuracy_score(all_labels_all, all_preds_all) * 100
            overall_mae = mean_absolute_error(all_labels_all, all_preds_all)
            micro_f1 = f1_score(all_labels_all, all_preds_all, average="micro", zero_division=0) * 100
            macro_f1 = f1_score(all_labels_all, all_preds_all, average="macro", zero_division=0) * 100
            micro_p = precision_score(all_labels_all, all_preds_all, average="micro", zero_division=0) * 100
            macro_p = precision_score(all_labels_all, all_preds_all, average="macro", zero_division=0) * 100
            micro_r = recall_score(all_labels_all, all_preds_all, average="micro", zero_division=0) * 100
            macro_r = recall_score(all_labels_all, all_preds_all, average="macro", zero_division=0) * 100

            report_lines.append(f"  Total sampel     : {len(all_labels_all)}")
            report_lines.append(f"  Overall Accuracy : {overall_acc:.2f}%")
            report_lines.append(f"  MAE              : {overall_mae:.4f}")
            report_lines.append(f"  [MICRO] P/R/F1   : {micro_p:.2f}% / {micro_r:.2f}% / {micro_f1:.2f}%")
            report_lines.append(f"  [MACRO] P/R/F1   : {macro_p:.2f}% / {macro_r:.2f}% / {macro_f1:.2f}%")
            report_lines.append("")
            for line in classification_report(
                all_labels_all,
                all_preds_all,
                target_names=class_names,
                labels=list(range(num_classes)),
                zero_division=0,
            ).splitlines():
                report_lines.append("  " + line)

        report_lines.append("\n" + "=" * width + "\n")
        full_report = "\n".join(report_lines)
        print(full_report)
        with open(output_dir / "classification_report_all_folds.txt", "w", encoding="utf-8") as handle:
            handle.write(full_report)
    except Exception as exc:
        print(f"[WARN] Classification report gagal: {exc}")
        traceback.print_exc()


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="WLASL InceptionI3d Multi-GPU DDP untuk dataset SIBI"
    )
    parser.add_argument(
        "--start_fold",
        type=int,
        default=1,
        choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)",
    )
    args = parser.parse_args()

    local_rank = setup_ddp()

    label_map_path = FOLDS_DIR / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as handle:
        label_data = json.load(handle)
    if isinstance(label_data["labels"], dict):
        class_names = [k for k, v in sorted(label_data["labels"].items(), key=lambda x: x[1])]
    else:
        class_names = label_data["labels"]
    num_classes = len(class_names)

    if is_main_process():
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*62}")
        print(f"  TRAINING {MODEL_LABEL}")
        print("  Model     : InceptionI3d bawaan WLASL")
        print(f"  Kelas     : {num_classes} ({', '.join(class_names)})")
        print(f"  Epochs    : {EPOCHS} | ES patience: {ES_PATIENCE}")
        print(f"  Batch/GPU : {BATCH_SIZE} | Eff batch: {BATCH_SIZE * dist.get_world_size()}")
        print(f"  Workers   : {NUM_WORKERS} | Crop: {FRAME_SIZE}")
        print(f"  LR        : {LEARNING_RATE} | WD: {WEIGHT_DECAY}")
        print(f"  Videos    : {VIDEOS_DIR}")
        print(f"  Output    : {OUTPUT_BASE}")
        print(f"  Pretrain  : {RGB_PRETRAINED}")
        print(f"{'='*62}\n")

    fold_accuracies = []

    for fold_idx in range(1, args.start_fold):
        log_file = OUTPUT_BASE / f"fold_{fold_idx}" / "log.txt"
        best = max((item["val_acc1"] for item in read_log(log_file)), default=0.0)
        fold_accuracies.append(best)
        if is_main_process():
            print(f"[INFO] Fold {fold_idx} sudah selesai. Best Acc@1: {best:.2f}%")

    for fold_idx in range(args.start_fold, NUM_FOLDS + 1):
        if is_main_process():
            print(f"\n{'='*62}")
            print(f"  FOLD {fold_idx}/{NUM_FOLDS}")
            print(f"{'='*62}")

        best_acc = train_fold(fold_idx, num_classes, local_rank)
        fold_accuracies.append(best_acc)

        if is_main_process():
            print(f"\n[INFO] Generate prediksi fold {fold_idx}...")
        generate_predictions(fold_idx, num_classes, local_rank)
        dist.barrier()

    if is_main_process():
        print(f"\n{'='*62}")
        print("  SELESAI - RINGKASAN HASIL")
        print(f"{'='*62}")
        valid = [acc for acc in fold_accuracies if acc > 0]
        for idx, acc in enumerate(fold_accuracies, 1):
            print(f"  Fold {idx} Best Acc@1 : {acc:.2f}%")
        if valid:
            print(f"  {'-'*40}")
            print(f"  Mean Acc@1 : {sum(valid)/len(valid):.2f}%")

        print("\n[INFO] Membuat laporan...")
        generate_reports(fold_accuracies, class_names)
        print(f"\n  Output: {OUTPUT_BASE}\n")

    cleanup_ddp()


if __name__ == "__main__":
    main()
