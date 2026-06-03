import torch

# Path ke file yang Anda download tadi
checkpoint_path = r'checkpoints/SSV2-I3D-ResNet-50-f32.pth.tar'

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("--- INFO CHECKPOINT ---")
    if isinstance(checkpoint, dict):
        print("Keys tersedia:", checkpoint.keys())
        # Kita cari 'state_dict' atau 'model'
        key = 'state_dict' if 'state_dict' in checkpoint else 'model'
        if key in checkpoint:
            print(f"Menggunakan key: {key}")
            layers = list(checkpoint[key].keys())
            print(f"Total layer: {len(layers)}")
            print("5 Layer pertama:", layers[:5])
        else:
            print("Peringatan: Tidak ditemukan 'state_dict' atau 'model'!")
    else:
        print("Checkpoint adalah state_dict langsung.")
except Exception as e:
    print(f"Gagal memuat file: {e}")