import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

WORK_DIRS = Path(r"D:\TUGASAKHIR\SourceCode\work_dirs_all_experiment")
FOLDS_DIR = Path(r"D:\TUGASAKHIR\SourceCode\folds_data")
LABEL_MAP = FOLDS_DIR / "label_map.json"

def read_predictions_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("labels", []), data.get("preds", [])

def read_csv_labels(csv_path):
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            try:
                labels.append(int(parts[-1]))
            except ValueError:
                pass
    return labels

def read_txt_predictions(txt_path):
    preds = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("0.0,") or line.startswith("0,"):
                continue
            try:
                bracket_start = line.index("[")
                bracket_end = line.index("]")
                scores_str = line[bracket_start + 1:bracket_end]
                scores = [float(s.strip()) for s in scores_str.split(",")]
                preds.append(int(np.argmax(scores)))
            except ValueError:
                continue
    return preds

def read_simple_txt_predictions(txt_path):
    preds = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                preds.append(int(line))
            except ValueError:
                pass
    return preds

def main():
    if not LABEL_MAP.exists():
        print(f"Label map tidak ditemukan di {LABEL_MAP}")
        return

    with open(LABEL_MAP, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    if isinstance(label_data.get("labels"), dict):
        class_names = [k for k, v in sorted(label_data["labels"].items(), key=lambda x: x[1])]
    else:
        class_names = label_data["labels"]
    
    num_classes = len(class_names)
    
    # Cari semua direktori fold_1 untuk menemukan root experiment
    for exp_dir in WORK_DIRS.rglob("fold_1"):
        exp_root = exp_dir.parent
        print(f"Memproses: {exp_root}")
        
        all_labels = []
        all_preds = []
        fold_accs = []
        
        for fold in range(1, 6):
            fdir = exp_root / f"fold_{fold}"
            if not fdir.exists():
                continue
                
            labels = []
            preds = []
            
            json_pred = fdir / "predictions.json"
            if json_pred.exists():
                labels, preds = read_predictions_json(json_pred)
            elif (fdir / "ground_truths.txt").exists() and (fdir / "predictions.txt").exists():
                labels = read_simple_txt_predictions(fdir / "ground_truths.txt")
                preds = read_simple_txt_predictions(fdir / "predictions.txt")
            else:
                csv_path = FOLDS_DIR / f"val_fold_{fold}.csv"
                txt_path = fdir / "0.txt"
                
                if not txt_path.exists():
                    if (fdir / "stage2" / "0.txt").exists():
                        txt_path = fdir / "stage2" / "0.txt"
                    elif (fdir / "stage1" / "0.txt").exists():
                        txt_path = fdir / "stage1" / "0.txt"
                    else:
                        for sub in fdir.iterdir():
                            if sub.is_dir() and (sub / "0.txt").exists():
                                txt_path = sub / "0.txt"
                                break
                            
                if csv_path.exists() and txt_path.exists():
                    labels = read_csv_labels(csv_path)
                    preds = read_txt_predictions(txt_path)
                    n = min(len(labels), len(preds))
                    labels = labels[:n]
                    preds = preds[:n]
            
            if labels and preds:
                all_labels.extend(labels)
                all_preds.extend(preds)
                acc = accuracy_score(labels, preds) * 100
                fold_accs.append(acc)
                
        if not all_labels:
            print(f"  [SKIP] Tidak ada prediksi yang valid di {exp_root}")
            continue
            
        mean_acc = sum(fold_accs) / len(fold_accs) if fold_accs else 0.0
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        
        fig, ax = plt.subplots(figsize=(13, 11))
        
        if "i3d" in exp_root.name.lower():
            cmap_color = plt.cm.Reds
        else:
            cmap_color = plt.cm.Blues
            
        image = ax.imshow(cm, interpolation="nearest", cmap=cmap_color)
        colorbar = plt.colorbar(image, ax=ax)
        colorbar.set_label("Jumlah Sampel", fontsize=10)

        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=16)
        ax.set_yticklabels(class_names, fontsize=16)

        threshold = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=24, # FONT SANGAT BESAR
                    fontweight="bold"
                )

        ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontsize=16)
        ax.set_xlabel("Prediksi Model", fontsize=16)
        ax.set_title(
            f"Confusion Matrix - {exp_root.name}\n"
            f"Mean Best Val Acc@1: {mean_acc:.2f}% | Total Sampel: {len(all_labels)}",
            fontsize=18,
        )
        plt.tight_layout()
        out_path = exp_root / "confusion_matrix_new_font.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Saved {out_path}")

if __name__ == '__main__':
    main()
