"""Preprocessing WLASL dataset → CSV format untuk VideoMAE.

Menggunakan nslt_N.json — file split RESMI dari paper WLASL (Li et al., WACV 2020).
File ini sudah tersedia di: WLASL/code/I3D/preprocess/nslt_100.json, nslt_300.json, dll.

Metodologi paper (diikuti PERSIS):
  - Training : train + val digabung  → trainval_wlaslN.csv
  - Evaluasi : test set saja         → test_wlaslN.csv

Format CSV output (space-delimited, sesuai VideoMAE SSV2 loader):
    videos/{video_id}.mp4 {label_idx}

Video harus tersedia di:
    wlasl_data/videos/{video_id}.mp4

Cara pakai:
    python preprocess_wlasl.py --subset 300
    python preprocess_wlasl.py --subset 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent
WLASL_SPLITS   = PROJECT_ROOT / "WLASL" / "code" / "I3D" / "preprocess"
CLASS_LIST_TXT = WLASL_SPLITS / "wlasl_class_list.txt"
OUTPUT_DIR     = PROJECT_ROOT / "wlasl_data"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nslt(subset: int) -> dict:
    """Muat nslt_N.json — file split resmi dari paper."""
    path = WLASL_SPLITS / f"nslt_{subset}.json"
    if not path.exists():
        print(f"[ERROR] Split file tidak ditemukan: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_list(n_classes: int) -> list[str]:
    """Muat wlasl_class_list.txt → list class name urut index."""
    if not CLASS_LIST_TXT.exists():
        print(f"[ERROR] Class list tidak ditemukan: {CLASS_LIST_TXT}")
        sys.exit(1)
    names: dict[int, str] = {}
    with open(CLASS_LIST_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split("\t", 1)
            idx = int(idx_str)
            if idx < n_classes:
                names[idx] = name
    return [names[i] for i in range(n_classes)]


def write_csv(rows: list[tuple[str, int]], out_path: Path,
              abs_prefix: str | None = None) -> None:
    """Write CSV file in VideoMAE format (space-delimited: path label)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rel_path, label in rows:
            path = f"{abs_prefix}/{rel_path}" if abs_prefix else rel_path
            f.write(f"{path} {label}\n")
    print(f"  {len(rows):>5} baris  →  {out_path.relative_to(PROJECT_ROOT)}")


def write_label_map(class_names: list[str], out_path: Path) -> None:
    label_map = {
        "labels": {name: idx for idx, name in enumerate(class_names)},
        "stats":  {"num_classes": len(class_names)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"  label_map  →  {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_splits(
    nslt_data: dict,
    video_dir: Path,
    skip_missing: bool,
) -> tuple[
    list[tuple[str, int]],  # train
    list[tuple[str, int]],  # val
    list[tuple[str, int]],  # test
    list[str],              # missing
]:
    """
    nslt_data: {video_id: {subset, action}}
    action[0] = label index (0-based, sesuai wlasl_class_list.txt)
    """
    train:   list[tuple[str, int]] = []
    val:     list[tuple[str, int]] = []
    test:    list[tuple[str, int]] = []
    missing: list[str]             = []

    for video_id, info in nslt_data.items():
        subset    = info["subset"]
        label_idx = info["action"][0]
        rel_path  = f"videos/{video_id}.mp4"
        abs_path  = video_dir / f"{video_id}.mp4"

        if not abs_path.exists():
            missing.append(video_id)
            if not skip_missing:
                continue

        row = (rel_path, label_idx)
        if subset == "train":
            train.append(row)
        elif subset == "val":
            val.append(row)
        elif subset == "test":
            test.append(row)
        else:
            train.append(row)

    return train, val, test, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat CSV WLASL untuk VideoMAE (paper-faithful, pakai nslt_N.json)"
    )
    parser.add_argument(
        "--subset", type=int, default=300, choices=[100, 300, 1000, 2000],
        help="Ukuran subset WLASL (default: 300)",
    )
    parser.add_argument(
        "--skip_missing", action="store_true",
        help="Sertakan entry meski file video belum ada di disk.",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Direktori berisi {video_id}.mp4 (default: wlasl_data/videos/)",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir) if args.video_dir else OUTPUT_DIR / "videos"
    n = args.subset

    print(f"\n{'='*62}")
    print(f"  PREPROCESSING WLASL{n}  (paper-faithful, sumber: nslt_{n}.json)")
    print(f"{'='*62}")
    print(f"  Split file : {WLASL_SPLITS / f'nslt_{n}.json'}")
    print(f"  Class list : {CLASS_LIST_TXT}")
    print(f"  Videos     : {video_dir}")
    print(f"  Output     : {OUTPUT_DIR}\n")

    if not video_dir.exists():
        print(f"[WARN] Folder video belum ada: {video_dir}")
        print(f"       Jalankan download_wlasl.py terlebih dahulu.")
        print(f"       Melanjutkan dengan --skip_missing (semua entry di-include).\n")
        args.skip_missing = True

    # Load data
    nslt_data   = load_nslt(n)
    class_names = load_class_list(n)

    print(f"[INFO] Total video di nslt_{n}.json : {len(nslt_data)}")
    print(f"[INFO] Jumlah kelas                 : {len(class_names)}")

    train, val, test, missing = build_splits(nslt_data, video_dir, args.skip_missing)
    trainval = train + val

    print(f"\n[INFO] Split resmi WLASL{n} (dari paper):")
    print(f"  Train            : {len(train):>5} video")
    print(f"  Val              : {len(val):>5} video")
    print(f"  Train + Val      : {len(trainval):>5} video  ← dipakai untuk training")
    print(f"  Test             : {len(test):>5} video  ← dipakai untuk evaluasi")

    if missing:
        pct = len(missing) / len(nslt_data) * 100
        print(f"\n[WARN] {len(missing)} video ({pct:.1f}%) tidak ditemukan di disk.")
        print(f"       Contoh: {missing[:5]}")
        if not args.skip_missing:
            print(f"       → Dilewati dari semua CSV.")

    if not trainval:
        print("\n[ERROR] Tidak ada data training. Download video dulu.")
        sys.exit(1)

    # Path absolut video (dipakai di SSV2 splits)
    abs_video_prefix = str(OUTPUT_DIR)

    print()

    # File referensi (relative paths)
    write_csv(trainval, OUTPUT_DIR / f"trainval_wlasl{n}.csv")
    write_csv(test,     OUTPUT_DIR / f"test_wlasl{n}.csv")
    write_csv(train,    OUTPUT_DIR / f"train_wlasl{n}.csv")
    write_csv(val,      OUTPUT_DIR / f"val_wlasl{n}.csv")

    # Label map
    write_label_map(class_names, OUTPUT_DIR / f"label_map_wlasl{n}.json")

    # SSV2-compatible splits (absolute paths, dalam direktori)
    # datasets.py VideoMAE mengharapkan:
    #   --data_path <dir>  → membaca <dir>/train.csv dan <dir>/val.csv
    #   data_path='/'      → path video harus absolut di dalam CSV
    ssv2_dir = OUTPUT_DIR / "ssv2_splits" / str(n)
    ssv2_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trainval, ssv2_dir / "train.csv", abs_prefix=abs_video_prefix)
    write_csv(test,     ssv2_dir / "val.csv",   abs_prefix=abs_video_prefix)
    write_csv(test,     ssv2_dir / "test.csv",  abs_prefix=abs_video_prefix)

    # Alias untuk evaluation framework
    write_csv(test, OUTPUT_DIR / "val_fold_1.csv")

    print(f"\n[DONE] Semua file di: {OUTPUT_DIR}")
    print(f"\nLangkah selanjutnya:")
    print(f"  python train_videomae/twostage_wlasl.py --subset {n}")


if __name__ == "__main__":
    main()
