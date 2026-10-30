import os
import sys
from pathlib import Path

# Setup path so we can import the original plotting logic
sys.path.insert(0, str(Path(r"D:\TUGASAKHIR\SourceCode")))
from train_videomae.evaluation import ReportConfig, plot_learning_curves_per_fold, plot_mean_learning_curve

def read_log(log_path):
    import json
    data = []
    if not log_path.exists(): return data
    with open(log_path, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                if 'epoch' in d and 'val_acc1' in d:
                    data.append(d)
            except: pass
    data.sort(key=lambda x: x['epoch'])
    return data

def process_exp(exp_root):
    fold_logs = []
    has_stage1 = False
    stage_split_epoch = 0
    
    for fold in range(1, 6):
        fdir = exp_root / f"fold_{fold}"
        stage1_log = fdir / "stage1" / "log.txt"
        stage2_log = fdir / "stage2" / "log.txt"
        
        fold_data = []
        if stage1_log.exists():
            s1 = read_log(stage1_log)
            fold_data.extend(s1)
            stage_split_epoch = max(stage_split_epoch, len(s1))
            has_stage1 = True
            
        if stage2_log.exists():
            s2 = read_log(stage2_log)
            s1_len = len(fold_data)
            for row in s2:
                new_row = dict(row)
                new_row['epoch'] += s1_len
                fold_data.append(new_row)
        fold_logs.append(fold_data)
        
    if has_stage1 and sum(len(f) for f in fold_logs) > 0:
        print(f"Generating full learning curve for: {exp_root.name}")
        config = ReportConfig(
            output_dir=exp_root,
            folds_dir=Path(r"D:\TUGASAKHIR\SourceCode\folds_data"),
            label=exp_root.name,
            stage_split_epoch=stage_split_epoch
        )
        plot_learning_curves_per_fold(config, fold_logs)
        plot_mean_learning_curve(config, fold_logs)

WORK_DIRS = [
    Path(r"D:\TUGASAKHIR\SourceCode\work_dirs_all_experiment\work_dirs_percobaan_baru"),
    Path(r"D:\TUGASAKHIR\SourceCode\work_dirs_all_experiment\work_dirs_VideoMAE_twostage")
]

for d in WORK_DIRS:
    for exp_root in d.iterdir():
        if exp_root.is_dir():
            process_exp(exp_root)

process_exp(Path(r"D:\TUGASAKHIR\SourceCode\work_dirs_all_experiment\work_dirs_vitb_k400_dense_twostage"))
