"""
train_sibi_ddp.py
==================
Training ResNet50-I3D dengan Multi-GPU (DistributedDataParallel)
untuk 2x RTX 4070 Ti (24GB VRAM total, 12GB per GPU).

Mengikuti protokol jurnal WLASL (arxiv 1910.11006).

Cara pakai:
  torchrun --nproc_per_node=2 train_sibi_ddp.py

Resume dari fold tertentu:
  torchrun --nproc_per_node=2 train_sibi_ddp.py --start_fold 3
"""

import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import traceback
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np

# =====================================================================
# KONFIGURASI PATH
# =====================================================================
BASE_DIR       = Path("/workspace")
FOLDS_DIR      = BASE_DIR / "folds_data"
VIDEOS_DIR     = FOLDS_DIR / "videos"
PREPROCESS_DIR = BASE_DIR / "WLASL" / "code" / "I3D" / "preprocess"
OUTPUT_BASE    = BASE_DIR / "work_dirs" / "resnet50_i3d_ddp"
SSV2_WEIGHTS   = BASE_DIR / "WLASL" / "checkpoints" / \
                 "SSV2-I3D-ResNet-50-f32.pth.tar"

# =====================================================================
# KONFIGURASI TRAINING
# =====================================================================
MODEL_LABEL   = "ResNet50-I3D | SSV2 | 2x RTX 4070 Ti"
NUM_FOLDS     = 5
EPOCHS        = 50
BATCH_SIZE    = 4    # 8 per GPU × 2 GPU = effective batch 16
NUM_WORKERS   = 4    # per GPU (total 8 workers dari 16 vCPU)
NUM_FRAMES    = 16
FRAME_SIZE    = 224
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
LR_PATIENCE   = 5
LR_FACTOR     = 0.3
SAVE_FREQ     = 10
ES_PATIENCE   = 5    # early stopping patience

# =====================================================================
# IMPORT DARI FOLDER I3D
# =====================================================================
I3D_DIR = BASE_DIR / "WLASL" / "code" / "I3D"
if str(I3D_DIR) not in sys.path:
    sys.path.insert(0, str(I3D_DIR))

try:
    import videotransforms
    from datasets.nslt_dataset import NSLT as Dataset
except ImportError as e:
    print(f"[ERROR] Gagal import: {e}")
    sys.exit(1)

from models.resnet_i3d import ResNet50_I3D


# =====================================================================
# DDP UTILITY
# =====================================================================
def setup_ddp():
    """Inisialisasi DDP process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Bersihkan DDP process group."""
    dist.destroy_process_group()


def is_main_process():
    """Cek apakah ini process utama (rank 0)."""
    return dist.get_rank() == 0


def reduce_tensor(tensor):
    """Rata-ratakan tensor dari semua GPU."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# =====================================================================
# TRAINING SATU FOLD
# =====================================================================
def train_fold(fold_idx, num_classes, class_names, local_rank):
    """Training satu fold dengan DDP."""

    train_split = str(PREPROCESS_DIR / f"sibi_train_fold{fold_idx}.json")
    val_split   = str(PREPROCESS_DIR / f"sibi_val_fold{fold_idx}.json")

    fold_output_dir = OUTPUT_BASE / f"fold_{fold_idx}"
    if is_main_process():
        fold_output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # Tunggu semua process selesai buat folder

    log_file = fold_output_dir / "log.txt"

    if not Path(train_split).exists():
        if is_main_process():
            print(f"[SKIP] JSON fold {fold_idx} tidak ditemukan.")
        return 0.0

    if is_main_process():
        print(f"  Device  : cuda:{local_rank} (rank {dist.get_rank()})")
        print(f"  Train   : {train_split}")
        print(f"  Val     : {val_split}")

    # ── Augmentasi (sesuai jurnal WLASL) ──────────────────────────
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(FRAME_SIZE),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(FRAME_SIZE),
    ])

    # ── Dataset ───────────────────────────────────────────────────
    root = {'word': str(VIDEOS_DIR)}

    train_dataset = Dataset(train_split, 'train', root, 'rgb',
                            train_transforms)
    val_dataset   = Dataset(val_split, 'test', root, 'rgb',
                            test_transforms)

    # ── DistributedSampler — membagi data ke tiap GPU ─────────────
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,      # ← pakai sampler, bukan shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────────────
    model = ResNet50_I3D(num_classes=num_classes)

    # Load weights hanya di rank 0, lalu broadcast ke semua GPU
    if is_main_process():
        model.load_ssv2_weights(str(SSV2_WEIGHTS))
    dist.barrier()

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank],
                output_device=local_rank)

    # ── Optimizer & Scheduler ──────────────────────────────────────
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=LR_PATIENCE, factor=LR_FACTOR)
    scaler = torch.amp.GradScaler('cuda')

    best_acc    = 0.0
    best_ep     = 0
    es_counter  = 0

    # ── Training Loop ──────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # Set epoch di sampler agar shuffle berbeda tiap epoch
        train_sampler.set_epoch(epoch)

        for phase in ['train', 'test']:
            model.train(phase == 'train')
            running_loss = 0.0
            corrects     = 0
            total        = 0

            active_loader = train_loader if phase == 'train' \
                else val_loader

            for data in active_loader:
                if data == -1:
                    continue
                inputs, labels, vid = data

                # SIBI Labels: (B, Classes, Frames) → Target (B)
                targets = torch.argmax(
                    torch.max(labels, dim=2)[0], dim=1
                ).to(local_rank)
                inputs  = inputs.to(local_rank)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss    = F.cross_entropy(outputs, targets)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                _, preds      = torch.max(outputs, 1)
                corrects      += (preds == targets).sum().item()
                total         += inputs.size(0)

            if total == 0:
                continue

            # Kumpulkan loss & acc dari semua GPU
            loss_tensor = torch.tensor(
                running_loss / total, device=local_rank)
            acc_tensor  = torch.tensor(
                corrects / total * 100, device=local_rank)

            loss_tensor = reduce_tensor(loss_tensor)
            acc_tensor  = reduce_tensor(acc_tensor)

            epoch_loss = loss_tensor.item()
            epoch_acc  = acc_tensor.item()

            if is_main_process():
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                      f"{phase.upper()} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Acc: {epoch_acc:.2f}%")

            if phase == 'test':
                scheduler.step(epoch_loss)

                if is_main_process():
                    # Simpan best model
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_ep  = epoch + 1
                        torch.save(
                            model.module.state_dict(),  # .module untuk DDP
                            fold_output_dir / "checkpoint-best.pth"
                        )
                        print(f"  ★ Best: {epoch_acc:.2f}%")

                    # Simpan checkpoint berkala
                    if (epoch + 1) % SAVE_FREQ == 0:
                        torch.save(
                            model.module.state_dict(),
                            fold_output_dir / f"checkpoint-{epoch}.pth"
                        )

                    # Log JSON
                    log_entry = {
                        "epoch"     : epoch,
                        "val_loss"  : round(epoch_loss, 4),
                        "val_acc1"  : round(epoch_acc, 4),
                        "lr"        : optimizer.param_groups[0]['lr']
                    }
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    print(f"  Best so far: {best_acc:.2f}% "
                          f"(epoch {best_ep})")

                    # Early Stopping — cek sebelum best_acc diupdate
                    # improved = True jika epoch ini lebih baik dari sebelumnya
                    if best_ep == epoch + 1:  # baru saja diupdate
                        es_counter = 0
                    else:
                        es_counter += 1
                        print(f"  [ES] Tidak ada peningkatan: "
                              f"{es_counter}/{ES_PATIENCE}")

                    if es_counter >= ES_PATIENCE:
                        print(f"  [EARLY STOP] Berhenti di "
                              f"epoch {epoch+1}.")
                        break

        # Broadcast sinyal early stop dari rank 0 ke semua GPU
        stop_signal = torch.tensor(
            1 if (is_main_process() and es_counter >= ES_PATIENCE)
            else 0,
            device=local_rank
        )
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    # Hapus checkpoint intermediate (hanya rank 0)
    if is_main_process():
        for f in fold_output_dir.glob("checkpoint-*.pth"):
            if f.name != "checkpoint-best.pth":
                f.unlink()
        print(f"\n[INFO] Checkpoint intermediate dihapus.")
        print(f"[RESULT] Fold {fold_idx} | "
              f"Best Val Acc@1: {best_acc:.2f}% (epoch {best_ep})")

    dist.barrier()
    return best_acc


# =====================================================================
# GENERATE PREDIKSI
# =====================================================================
def generate_predictions(fold_idx, num_classes, local_rank):
    """Inference dengan best checkpoint — hanya di rank 0."""
    if not is_main_process():
        return [], []

    val_split       = str(PREPROCESS_DIR / f"sibi_val_fold{fold_idx}.json")
    fold_output_dir = OUTPUT_BASE / f"fold_{fold_idx}"
    checkpoint_path = fold_output_dir / "checkpoint-best.pth"
    pred_file       = fold_output_dir / "predictions.json"

    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint tidak ada: {checkpoint_path}")
        return [], []

    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(FRAME_SIZE),
    ])
    root        = {'word': str(VIDEOS_DIR)}
    val_dataset = Dataset(val_split, 'test', root, 'rgb', test_transforms)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Load model tanpa DDP untuk inference
    model = ResNet50_I3D(num_classes=num_classes)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=f'cuda:{local_rank}',
                   weights_only=True))
    model = model.to(local_rank)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data in val_loader:
            if data == -1:
                continue
            inputs, labels, vid = data
            targets = torch.argmax(
                torch.max(labels, dim=2)[0], dim=1)
            inputs  = inputs.to(local_rank)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(targets.numpy().tolist())

    with open(pred_file, "w") as f:
        json.dump({"preds": all_preds, "labels": all_labels}, f)

    print(f"  Prediksi fold {fold_idx}: {len(all_preds)} sampel")
    return all_labels, all_preds


# =====================================================================
# BACA LOG
# =====================================================================
def read_log(log_file):
    data_list = []
    if not Path(log_file).exists():
        return data_list
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "val_acc1" in d and "epoch" in d:
                    data_list.append(d)
            except json.JSONDecodeError:
                continue
    data_list.sort(key=lambda x: x["epoch"])
    return data_list


# =====================================================================
# LAPORAN (hanya rank 0)
# =====================================================================
def generate_reports(fold_accuracies, class_names):
    """Generate semua laporan — sama persis dengan train_sibi.py."""
    if not is_main_process():
        return

    output_dir  = OUTPUT_BASE
    num_classes = len(class_names)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            confusion_matrix, classification_report,
            accuracy_score, f1_score,
            precision_score, recall_score,
            mean_absolute_error
        )
    except ImportError as e:
        print(f"[WARN] Library kurang: {e}")
        return

    fold_log_data  = []
    all_labels_all = []
    all_preds_all  = []
    fold_gt        = []
    fold_pred      = []

    for fold_idx in range(1, NUM_FOLDS + 1):
        fold_log_data.append(
            read_log(OUTPUT_BASE / f"fold_{fold_idx}" / "log.txt"))
        pred_file = OUTPUT_BASE / f"fold_{fold_idx}" / "predictions.json"
        if pred_file.exists():
            with open(pred_file) as f:
                data = json.load(f)
            gt   = data.get("labels", [])
            pred = data.get("preds", [])
            fold_gt.append(gt); fold_pred.append(pred)
            all_labels_all.extend(gt); all_preds_all.extend(pred)
        else:
            fold_gt.append([]); fold_pred.append([])

    has_preds = len(all_preds_all) > 0

    # ── Learning Curve per Fold ────────────────────────────────────
    try:
        fig, axes = plt.subplots(NUM_FOLDS, 2,
                                 figsize=(16, 5 * NUM_FOLDS))
        all_val_acc = []; all_val_loss = []

        for fi, epochs_data in enumerate(fold_log_data, 1):
            if not epochs_data: continue
            ep     = [d["epoch"]+1        for d in epochs_data]
            vl_loss= [d.get("val_loss",0) for d in epochs_data]
            vl_acc = [d.get("val_acc1",0) for d in epochs_data]
            all_val_acc.append(vl_acc); all_val_loss.append(vl_loss)

            ax_a = axes[fi-1][0]; ax_l = axes[fi-1][1]
            ax_a.plot(ep, vl_acc, 'b-o', markersize=3, lw=1.5,
                      label='Val Acc@1')
            if vl_acc:
                bep=ep[int(np.argmax(vl_acc))]; bv=max(vl_acc)
                ax_a.axvline(x=bep, color='red', linestyle='--',
                             alpha=0.6,
                             label=f'Best ep{bep}: {bv:.2f}%')
            ax_a.set_title(f'Fold {fi} — Validation Accuracy',
                           fontsize=11)
            ax_a.set_xlabel('Epoch'); ax_a.set_ylabel('Acc@1 (%)')
            ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.3)
            ax_a.set_ylim(0, 100)
            ax_l.plot(ep, vl_loss, 'b-o', markersize=3, lw=1.5,
                      label='Val Loss')
            ax_l.set_title(f'Fold {fi} — Loss Curve', fontsize=11)
            ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('Loss')
            ax_l.legend(fontsize=8); ax_l.grid(True, alpha=0.3)

        plt.suptitle(f'Learning Curves per Fold\n{MODEL_LABEL}',
                     fontsize=14, y=1.005)
        plt.tight_layout()
        p = output_dir/"learning_curves_per_fold.png"
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  ✓ {p.name}")
    except Exception as e:
        print(f"[WARN] Learning curve gagal: {e}")

    # ── Mean Learning Curve ────────────────────────────────────────
    try:
        if all_val_acc:
            ml  = max(len(c) for c in all_val_acc)
            pad = lambda l: l + [l[-1]]*(ml-len(l))
            m_a = np.array([pad(c) for c in all_val_acc]).mean(0)
            s_a = np.array([pad(c) for c in all_val_acc]).std(0)
            m_vl= np.array([pad(c) for c in all_val_loss]).mean(0)
            s_vl= np.array([pad(c) for c in all_val_loss]).std(0)
            ep_r= list(range(1, ml+1))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(ep_r, m_a, 'b-', lw=2, label='Mean Val Acc@1')
            ax1.fill_between(ep_r, m_a-s_a, m_a+s_a, alpha=0.2,
                             color='blue', label='±1 Std')
            ax1.set_title('Mean Validation Accuracy (5-Fold CV)',
                          fontsize=12)
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('Acc@1 (%)')
            ax1.set_ylim(0,100); ax1.legend(); ax1.grid(True,alpha=0.3)
            ax2.plot(ep_r, m_vl, 'b-', lw=2, label='Mean Val Loss')
            ax2.fill_between(ep_r, m_vl-s_vl, m_vl+s_vl, alpha=0.2,
                             color='blue', label='Val ±1 Std')
            ax2.set_title('Mean Loss Curve (5-Fold CV)', fontsize=12)
            ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
            ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.suptitle(f'{MODEL_LABEL} — Mean Learning Curves',
                         fontsize=14)
            plt.tight_layout()
            p = output_dir/"learning_curves_mean.png"
            plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  ✓ {p.name}")
    except Exception as e:
        print(f"[WARN] Mean learning curve gagal: {e}")

    # ── Bar Chart Akurasi per Fold ─────────────────────────────────
    try:
        if fold_accuracies:
            fig, ax = plt.subplots(figsize=(9, 5))
            xl  = [f"Fold {i}" for i in range(1,len(fold_accuracies)+1)]
            bars= ax.bar(xl, fold_accuracies, color='#C00000',
                         edgecolor='black', linewidth=0.7)
            mv  = sum(fold_accuracies)/len(fold_accuracies)
            ax.axhline(y=mv, color='red', linestyle='--', lw=1.5,
                       label=f'Mean: {mv:.2f}%')
            for bar, acc in zip(bars, fold_accuracies):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5,
                        f'{acc:.2f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
            ax.set_title(f'Best Val Acc@1 per Fold\n{MODEL_LABEL}',
                         fontsize=13)
            ax.set_ylabel('Best Val Acc@1 (%)'); ax.set_ylim(0, 100)
            ax.legend(); ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            p = output_dir/"accuracy_per_fold.png"
            plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  ✓ {p.name}")
    except Exception as e:
        print(f"[WARN] Bar chart gagal: {e}")

    # ── Confusion Matrix Gabungan ──────────────────────────────────
    try:
        if has_preds:
            cm  = confusion_matrix(all_labels_all, all_preds_all,
                                   labels=list(range(num_classes)))
            fig, ax = plt.subplots(figsize=(13, 11))
            im  = ax.imshow(cm, interpolation='nearest',
                            cmap=plt.cm.Reds)
            plt.colorbar(im, ax=ax).set_label('Jumlah Sampel',
                                              fontsize=10)
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45,
                               ha='right', fontsize=11)
            ax.set_yticklabels(class_names, fontsize=11)
            thresh = cm.max()/2.0
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, format(cm[i,j],'d'),
                            ha="center", va="center",
                            color="white" if cm[i,j]>thresh
                            else "black", fontsize=24)
            mv = sum(fold_accuracies)/len(fold_accuracies) \
                if fold_accuracies else 0
            ax.set_ylabel('Label Sebenarnya (Ground Truth)',
                          fontsize=12)
            ax.set_xlabel('Prediksi Model', fontsize=12)
            ax.set_title(
                f'Confusion Matrix — {MODEL_LABEL}\n'
                f'Mean Best Val Acc@1: {mv:.2f}% | '
                f'Total Sampel: {len(all_labels_all)}',
                fontsize=13)
            plt.tight_layout()
            p = output_dir/"confusion_matrix.png"
            plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  ✓ {p.name}")
    except Exception as e:
        print(f"[WARN] Confusion matrix gagal: {e}")

    # ── Confusion Matrix per Fold ──────────────────────────────────
    try:
        vf = [(i+1, fold_gt[i], fold_pred[i])
              for i in range(NUM_FOLDS) if fold_pred[i]]
        if vf:
            cols = min(3, len(vf))
            rows = (len(vf)+cols-1)//cols
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(cols*7, rows*6))
            if rows*cols==1: axes=[[axes]]
            elif rows==1: axes=[list(axes)]
            else: axes=[list(r) for r in axes]
            for idx,(fi,gt,pred) in enumerate(vf):
                r,c=divmod(idx,cols); ax=axes[r][c]
                cm_f=confusion_matrix(gt,pred,
                                      labels=list(range(num_classes)))
                ax.imshow(cm_f, interpolation='nearest',
                          cmap=plt.cm.Reds)
                ax.set_xticks(range(num_classes))
                ax.set_yticks(range(num_classes))
                ax.set_xticklabels(class_names, rotation=45,
                                   ha='right', fontsize=8)
                ax.set_yticklabels(class_names, fontsize=8)
                thr=cm_f.max()/2.0
                for i in range(num_classes):
                    for j in range(num_classes):
                        ax.text(j,i,format(cm_f[i,j],'d'),
                                ha="center",va="center",
                                color="white" if cm_f[i,j]>thr
                                else "black",fontsize=20)
                ax.set_title(
                    f'Fold {fi} — '
                    f'Acc: {accuracy_score(gt,pred)*100:.2f}%',
                    fontsize=10)
            for idx in range(len(vf), rows*cols):
                r,c=divmod(idx,cols); axes[r][c].set_visible(False)
            plt.suptitle(
                f'Confusion Matrix per Fold\n{MODEL_LABEL}',
                fontsize=13)
            plt.tight_layout()
            p = output_dir/"confusion_matrix_per_fold.png"
            plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  ✓ {p.name}")
    except Exception as e:
        print(f"[WARN] Confusion matrix per fold gagal: {e}")

    # ── Classification Report ──────────────────────────────────────
    try:
        W=72; rpt=[]
        rpt.append("="*W)
        rpt.append(f"  CLASSIFICATION REPORT — {MODEL_LABEL}")
        rpt.append(f"  Protokol : WLASL (arxiv 1910.11006)")
        rpt.append(f"  Epochs: {EPOCHS} | LR: {LEARNING_RATE} | "
                   f"Batch: {BATCH_SIZE}x2GPU | Frames: {NUM_FRAMES}")
        rpt.append("="*W+"\n")

        fmaes=[]; fmif1=[]; fmaf1=[]
        fmip=[]; fmap=[]; fmir=[]; fmar=[]

        for fi in range(1, NUM_FOLDS+1):
            gt  = fold_gt[fi-1]   if fi<=len(fold_gt)   else []
            pred= fold_pred[fi-1] if fi<=len(fold_pred) else []
            rpt.append("─"*W)
            rpt.append(f"  FOLD {fi}")
            rpt.append("─"*W)
            bv = fold_accuracies[fi-1] \
                if fi<=len(fold_accuracies) else 0
            rpt.append(f"  Best Val Acc@1 : {bv:.2f}%")
            if not pred or not gt:
                rpt.append("  [SKIP] Tidak ada prediksi.\n")
                continue
            rpt.append(f"  Jumlah sampel  : {len(gt)}\n")
            mae=mean_absolute_error(gt,pred); fmaes.append(mae)
            rpt.append(f"  MAE : {mae:.4f}")
            mf1=f1_score(gt,pred,average='micro',zero_division=0)*100
            mp =precision_score(gt,pred,average='micro',zero_division=0)*100
            mr =recall_score(gt,pred,average='micro',zero_division=0)*100
            fmif1.append(mf1); fmip.append(mp); fmir.append(mr)
            rpt.append(f"\n  [MICRO]  "
                       f"Acc: {accuracy_score(gt,pred)*100:.2f}%  "
                       f"P: {mp:.2f}%  R: {mr:.2f}%  F1: {mf1:.2f}%")
            Mf1=f1_score(gt,pred,average='macro',zero_division=0)*100
            Mp =precision_score(gt,pred,average='macro',zero_division=0)*100
            Mr =recall_score(gt,pred,average='macro',zero_division=0)*100
            fmaf1.append(Mf1); fmap.append(Mp); fmar.append(Mr)
            rpt.append(f"  [MACRO]  "
                       f"P: {Mp:.2f}%  R: {Mr:.2f}%  F1: {Mf1:.2f}%")
            rpt.append(f"\n  [PER CLASS]")
            for ln in classification_report(
                gt, pred, target_names=class_names,
                labels=list(range(num_classes)),
                zero_division=0).splitlines():
                rpt.append("  "+ln)
            rpt.append("")

        rpt.append("\n"+"="*W)
        rpt.append("  GABUNGAN SEMUA FOLD")
        rpt.append("="*W)
        if has_preds:
            oa =accuracy_score(all_labels_all,all_preds_all)*100
            om =mean_absolute_error(all_labels_all,all_preds_all)
            rpt.append(f"\n  Total sampel     : {len(all_labels_all)}")
            rpt.append(f"  Overall Accuracy : {oa:.2f}%")
            rpt.append(f"  MAE              : {om:.4f}")
            omif1=f1_score(all_labels_all,all_preds_all,
                           average='micro',zero_division=0)*100
            omip =precision_score(all_labels_all,all_preds_all,
                                  average='micro',zero_division=0)*100
            omir =recall_score(all_labels_all,all_preds_all,
                               average='micro',zero_division=0)*100
            omaf1=f1_score(all_labels_all,all_preds_all,
                           average='macro',zero_division=0)*100
            omap =precision_score(all_labels_all,all_preds_all,
                                  average='macro',zero_division=0)*100
            omar =recall_score(all_labels_all,all_preds_all,
                               average='macro',zero_division=0)*100
            rpt.append(f"\n  [MICRO]  P: {omip:.2f}%  "
                       f"R: {omir:.2f}%  F1: {omif1:.2f}%")
            rpt.append(f"  [MACRO]  P: {omap:.2f}%  "
                       f"R: {omar:.2f}%  F1: {omaf1:.2f}%")
            rpt.append(f"\n  [PER CLASS — semua fold]")
            for ln in classification_report(
                all_labels_all, all_preds_all,
                target_names=class_names,
                labels=list(range(num_classes)),
                zero_division=0).splitlines():
                rpt.append("  "+ln)

        if fold_accuracies:
            ma=sum(fold_accuracies)/len(fold_accuracies)
            rpt.append(f"\n{'─'*W}")
            rpt.append("  RINGKASAN MEAN PER FOLD")
            rpt.append(f"{'─'*W}")
            for i,acc in enumerate(fold_accuracies,1):
                rpt.append(f"  Fold {i}  Best Val Acc@1 : {acc:.2f}%")
            rpt.append(f"  {'─'*45}")
            rpt.append(f"  Mean Best Val Acc@1  : {ma:.2f}%")
            if fmaes:
                rpt.append(f"  Mean MAE             : "
                           f"{sum(fmaes)/len(fmaes):.4f}")
            if fmif1:
                rpt.append(f"  Mean Micro P/R/F1    : "
                           f"{sum(fmip)/len(fmip):.2f}% / "
                           f"{sum(fmir)/len(fmir):.2f}% / "
                           f"{sum(fmif1)/len(fmif1):.2f}%")
            if fmaf1:
                rpt.append(f"  Mean Macro P/R/F1    : "
                           f"{sum(fmap)/len(fmap):.2f}% / "
                           f"{sum(fmar)/len(fmar):.2f}% / "
                           f"{sum(fmaf1)/len(fmaf1):.2f}%")
        rpt.append("\n"+"="*W+"\n")

        full="\n".join(rpt)
        print(full)
        p=output_dir/"classification_report_all_folds.txt"
        with open(p,"w",encoding="utf-8") as f: f.write(full)
        print(f"  ✓ {p.name}")
    except Exception as e:
        print(f"[WARN] Classification report gagal: {e}")
        traceback.print_exc()


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="ResNet50-I3D Multi-GPU DDP (2x RTX 4070 Ti)")
    parser.add_argument(
        "--start_fold", type=int, default=1, choices=range(1, 6),
        help="Mulai dari fold ke-N (default: 1)")
    args = parser.parse_args()

    # Setup DDP
    local_rank = setup_ddp()

    # Baca label map
    label_map_path = FOLDS_DIR / "label_map.json"
    with open(label_map_path, "r") as f:
        label_data = json.load(f)
        if isinstance(label_data["labels"], dict):
            class_names = [k for k, v in sorted(
                label_data["labels"].items(), key=lambda x: x[1])]
        else:
            class_names = label_data["labels"]
        num_classes = len(class_names)

    if is_main_process():
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*62}")
        print(f"  TRAINING {MODEL_LABEL}")
        print(f"  Protokol  : WLASL (arxiv 1910.11006)")
        print(f"  Kelas     : {num_classes} ({', '.join(class_names)})")
        print(f"  Epochs    : {EPOCHS} | ES patience: {ES_PATIENCE}")
        print(f"  Batch/GPU : {BATCH_SIZE} | Eff batch: "
              f"{BATCH_SIZE * dist.get_world_size()}")
        print(f"  Frames    : {NUM_FRAMES} | Workers/GPU: {NUM_WORKERS}")
        print(f"  LR        : {LEARNING_RATE} | WD: {WEIGHT_DECAY}")
        print(f"  GPUs      : {dist.get_world_size()} "
              f"(rank {dist.get_rank()})")
        print(f"  Output    : {OUTPUT_BASE}")
        print(f"  Start     : Fold {args.start_fold}")
        print(f"{'='*62}\n")

    fold_accuracies = []

    # Baca fold yang sudah selesai
    for fold_idx in range(1, args.start_fold):
        log_file = OUTPUT_BASE / f"fold_{fold_idx}" / "log.txt"
        best = max((d["val_acc1"] for d in read_log(log_file)),
                   default=0.0)
        fold_accuracies.append(best)
        if is_main_process():
            print(f"[INFO] Fold {fold_idx} sudah selesai. "
                  f"Best Acc@1: {best:.2f}%")

    # Training
    for fold_idx in range(args.start_fold, NUM_FOLDS + 1):
        if is_main_process():
            print(f"\n{'='*62}")
            print(f"  FOLD {fold_idx}/{NUM_FOLDS}")
            print(f"{'='*62}")

        best_acc = train_fold(fold_idx, num_classes, class_names,
                              local_rank)
        fold_accuracies.append(best_acc)

        if is_main_process():
            print(f"\n[INFO] Generate prediksi fold {fold_idx}...")
        generate_predictions(fold_idx, num_classes, local_rank)
        dist.barrier()

    # Ringkasan & laporan (hanya rank 0)
    if is_main_process():
        print(f"\n{'='*62}")
        print(f"  SELESAI — RINGKASAN HASIL")
        print(f"{'='*62}")
        valid = [a for a in fold_accuracies if a > 0]
        for i, acc in enumerate(fold_accuracies, 1):
            print(f"  Fold {i} Best Acc@1 : {acc:.2f}%")
        if valid:
            print(f"  {'─'*40}")
            print(f"  Mean Acc@1 : {sum(valid)/len(valid):.2f}%")

        print(f"\n[INFO] Membuat laporan...")
        generate_reports(fold_accuracies, class_names)
        print(f"\n  Output: {OUTPUT_BASE}\n")

    cleanup_ddp()


if __name__ == "__main__":
    main()