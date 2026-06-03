# SIBI Recognition VideoMAE

Model ini adalah checkpoint `checkpoint-best.pth` untuk VideoMAE ViT-Base K400 yang difinetune pada dataset Anda.

## Konten repo
- `checkpoint-best.pth`: model weights

## Cara pakai dari Hugging Face
1. Install dependensi:

```bash
pip install torch torchvision timm decord numpy opencv-python huggingface_hub
```

2. Download checkpoint dari repo:

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="Luckystg/SIBI_Recognition_VideoMAE",
    filename="checkpoint-best.pth"
)
print(checkpoint_path)
```

3. Muat model dengan kode VideoMAE Anda:

```python
import torch
from VideoMAE import modeling_finetune

model = modeling_finetune.vit_base_patch16_224(
    pretrained=False,
    num_classes=10,
    all_frames=16,
    tubelet_size=2,
    drop_rate=0.0,
    drop_path_rate=0.0,
    attn_drop_rate=0.0,
    use_mean_pooling=True,
    init_scale=0.001,
)

ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
sd = ckpt.get('model', ckpt)
model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}, strict=False)
model.eval()
```

4. Jalankan inferensi video.

## Catatan
- Karena checkpoint ini menggunakan arsitektur custom VideoMAE, Anda tidak bisa langsung menggunakan `AutoModel.from_pretrained` dari `transformers`.
- Pastikan Anda memiliki kode `VideoMAE` di environment Anda saat menjalankan inferensi.

## Saran
Untuk membuat repo lebih mudah dipakai dari Hugging Face, Anda bisa menambahkan:
- `README.md` dengan instruksi ini
- `requirements.txt` minimal untuk dependency
- contoh script inferensi `inferensi_manual.py`
