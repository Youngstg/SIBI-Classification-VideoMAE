import pandas as pd
import json
import os

# Base directory for dynamic path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def convert_sibi_csv_to_json(csv_relative_path, output_name):
    """Convert VideoMAE CSV format to WLASL I3D JSON format."""
    # Construct full CSV path dynamically
    csv_path = os.path.join(BASE_DIR, csv_relative_path)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path, header=None, sep='\s+')
    data_dict = {}
    
    for _, row in df.iterrows():
        try:
            full_path = str(row[0])
            label_idx = int(row[1])
            vid_id = os.path.basename(full_path).replace('.mp4', '')
            
            # Mendapatkan nama folder kelas (Aku, Bapak, dll)
            path_parts = full_path.replace('\\', '/').split('/')
            label_folder = path_parts[1] if len(path_parts) > 2 else ""
            
            data_dict[vid_id] = {
                "subset": "train" if "train" in csv_relative_path else "test",
                "label": label_idx,
                "folder": label_folder,
                "action": [label_idx, 1, 60]
            }
        except:
            continue
    
    # Path output juga relatif
    out_dir = os.path.join(BASE_DIR, 'wlasl', 'code', 'I3D', 'preprocess')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_name}.json")
    
    with open(out_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"Berhasil membuat: {out_path}")

if __name__ == '__main__':
    for i in range(1, 6):
        convert_sibi_csv_to_json(f'folds_data/train_fold_{i}.csv', f'sibi_train_fold{i}')
        convert_sibi_csv_to_json(f'folds_data/val_fold_{i}.csv', f'sibi_val_fold{i}')