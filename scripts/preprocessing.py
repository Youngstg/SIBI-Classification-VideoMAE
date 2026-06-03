"""
Script khusus untuk membagi dataset menjadi 5-Fold Stratified Split
untuk keperluan Cross-Validation pada VideoMAE.
Script Preprocessing Lengkap:
1. Standardisasi Video (Resize 224x224, 30 FPS) -> Implementasi 'Pre-cooking'
2. Pembagian Dataset (5-Fold Stratified Split)

Alur Proses (Split-First):
1. Scan Dataset Original -> 2. Hitung Split (Stratified) -> 3. Proses & Augmentasi -> 4. Generate CSV

Cara pakai:
    python preprocessing_folds.py --dataset_root "Dataset/dataset_bersih" --output_dir "folds_data" --num_folds 5
    python preprocessing_folds.py --dataset_root "Dataset/dataset_bersih" --output_dir "processed_data" --num_folds 5
"""

import argparse
import json
import random
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from sklearn.model_selection import StratifiedKFold

DEFAULT_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label_id: int

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5-Fold Stratified CSVs for VideoMAE.")
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path ke folder dataset utama (yang berisi subfolder kelas).",
    )
    parser.add_argument(
        "--output_dir",
        default="folds_data",
        help="Folder output untuk menyimpan file CSV hasil split.",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Jumlah fold (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reproducibility.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Ukuran resize video (default: 224).",
    )
    return parser.parse_args()

def check_ffmpeg():
    """Mengecek ketersediaan FFmpeg."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def process_video(input_path: Path, output_path: Path, size: int = 224, fps: int = 30, extra_filter: str = None):
    """
    Melakukan preprocessing fisik pada video:
    - Resize ke size x size
    - Set FPS ke 30
    - Hapus audio (-an)
    - Opsional: Augmentasi via extra_filter (negate, eq, dll)
    """
    if output_path.exists():
        return True # Skip jika sudah ada

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build FFmpeg filter chain for video processing
    vf_str = f"scale={size}:{size},fps={fps}"
    if extra_filter:
        vf_str += f",{extra_filter}"

    cmd = [
        "ffmpeg", "-y", "-threads", "0", # Use all CPU cores
        "-i", str(input_path),
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",  # Remove audio
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def collect_class_dirs(root: Path) -> list[Path]:
    candidates = [
        p for p in root.iterdir() 
        if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(candidates, key=lambda item: item.name.lower())

def write_csv(records: list[VideoRecord], path: Path, base_dir: Path):
    # Sort records for deterministic output
    records.sort(key=lambda x: (x.label_id, x.path.name))
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            # Format: relative_path label_id (VideoMAE CSV format)
            relative_path = r.path.relative_to(base_dir)
            f.write(f"{relative_path.as_posix()} {r.label_id}\n")

def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not check_ffmpeg():
        print("[ERROR] FFmpeg tidak ditemukan. Mohon install FFmpeg terlebih dahulu.")
        sys.exit(1)

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root tidak ditemukan: {dataset_root}")
        return

    print(f"Membaca dataset dari: {dataset_root}")
    print(f"1. MEMULAI PREPROCESSING & AUGMENTASI (Resize {args.size}x{args.size}, 30fps, Inversion, Multiplication, Downsampling)")
    print(f"   Sumber: {dataset_root}")
    print(f"   Tujuan: {output_dir}/videos")
    
    # 1. SCAN DATASET ORIGINAL
    print(f"1. Memindai dataset original dari: {dataset_root}")
    class_dirs = collect_class_dirs(dataset_root)
    
    # Filter folder output jika ada di dalam dataset root agar tidak dianggap sebagai kelas
    class_dirs = [d for d in class_dirs if d.resolve() != output_dir]
    
    if not class_dirs:
        print("[ERROR] Tidak ada folder kelas ditemukan.")
        return

    # 1. SCAN DATASET ORIGINAL
    print(f"1. Memindai dataset original dari: {dataset_root}")
    class_dirs = collect_class_dirs(dataset_root)
    
    # Filter output folder to prevent treating it as a class
    class_dirs = [d for d in class_dirs if d.resolve() != output_dir]
    
    if not class_dirs:
        print("[ERROR] Tidak ada folder kelas ditemukan.")
        return

    # Stage 1: Collect all original files with their labels
    processed_video_root = output_dir / "videos"
    processed_class_dirs = []
    original_data = []  # List of (source_path, label_id, class_name) tuples
    label_map = {d.name: i for i, d in enumerate(class_dirs)}
    
    for d in class_dirs:
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in DEFAULT_VIDEO_EXTENSIONS]
        files.sort(key=lambda f: f.name) # Deterministic order
        for f in files:
            original_data.append((f, label_map[d.name], d.name))
            
    print(f"   Ditemukan {len(original_data)} video original dari {len(class_dirs)} kelas.")

    # 2. PERFORM STRATIFIED SPLIT (BEFORE PROCESSING)
    print(f"2. Melakukan Stratified Split ({args.num_folds} folds)...")
    X_dummy = np.zeros(len(original_data))
    y_labels = np.array([item[1] for item in original_data])
    
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    # folds_indices contains (train_indices, val_indices) for each fold
    folds_indices = list(skf.split(X_dummy, y_labels))

    # 3. PROCESS VIDEO (RESIZE & AUGMENTATION)
    print(f"3. Memproses video (Resize {args.size}x{args.size} & Augmentasi)...")
    processed_video_root = output_dir / "videos"
    processed_video_root.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store processing results
    # Key: source_path, Value: {'orig': Path, 'aug': [Path, ...], 'label': int}
    processed_map = {}
    
    total_files = len(original_data)
    for i, (src_path, label_id, class_name) in enumerate(original_data):
        dest_dir = processed_video_root / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        path_orig = dest_dir / (src_path.stem + ".mp4")
        path_inv = dest_dir / (src_path.stem + "_inv.mp4")
        path_mult = dest_dir / (src_path.stem + "_mult.mp4")
        path_down = dest_dir / (src_path.stem + "_down.mp4")
        
        try:
            # A. Standardize to original resolution
            process_video(src_path, path_orig, size=args.size)
            
            # B. Generate augmentations (applied to all files for training flexibility)
            # Note: In 5-Fold CV, each file becomes training data in 4 folds
            # Augmentations are generated for all files, CSV controls visibility
            process_video(src_path, path_inv, size=args.size, extra_filter="negate")  # Invert colors
            process_video(src_path, path_mult, size=args.size, extra_filter="lutrgb=r=val*1.2:g=val*1.2:b=val*1.2")  # Increase brightness
            down_size = args.size // 2
            process_video(src_path, path_down, size=args.size, extra_filter=f"scale={down_size}:{down_size},scale={args.size}:{args.size}:flags=neighbor")  # Downsample
            
            processed_map[src_path] = {
                'orig': path_orig,
                'aug': [path_inv, path_mult, path_down],
                'label': label_id
            }
            
            print(f"   Processing [{i+1}/{total_files}]: {src_path.name}", end="\r")
            
        except subprocess.CalledProcessError:
            print(f"\n   [SKIP] Gagal memproses: {src_path.name}")

    print("\n")

    # 4. GENERATE CSV FILES FROM FOLD SPLIT
    print("4. Membuat file CSV...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {"folds": {}}
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds_indices):
        fold_id = fold_idx + 1
        
        train_records = []
        val_records = []
        
        # Training CSV: Include original + augmented versions
        for idx in train_idx:
            src_path = original_data[idx][0]
            if src_path in processed_map:
                data = processed_map[src_path]
                # Add original
                train_records.append(VideoRecord(data['orig'], data['label']))
                # Add augmentations for data diversity
                for aug_path in data['aug']:
                    train_records.append(VideoRecord(aug_path, data['label']))
                    
        # Validation CSV: Include only original (unaugmented) versions
        for idx in val_idx:
            src_path = original_data[idx][0]
            if src_path in processed_map:
                data = processed_map[src_path]
                # Add original only (no augmentations for validation)
                val_records.append(VideoRecord(data['orig'], data['label']))
        
        # Write CSV files for this fold
        write_csv(train_records, output_dir / f"train_fold_{fold_id}.csv", base_dir=output_dir)
        write_csv(val_records, output_dir / f"val_fold_{fold_id}.csv", base_dir=output_dir)
        
        print(f"   Fold {fold_id}:")
        print(f"     - Train: {len(train_records)} videos (Original + Augmentations)")
        print(f"     - Val  : {len(val_records)} videos (Original Only)")
        summary["folds"][f"fold_{fold_id}"] = {
            "train_samples": len(train_records),
            "val_samples": len(val_records)
        }

    # Save Label Map
    with (output_dir / "label_map.json").open("w") as f:
        json.dump({"labels": label_map, "stats": summary}, f, indent=2)

    print(f"\n[SELESAI] Output tersimpan di: {output_dir}")

if __name__ == "__main__":
    main()