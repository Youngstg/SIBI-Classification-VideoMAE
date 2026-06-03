"""
inferensi_manual.py
--------------------
Klasifikasi 1 video secara manual menggunakan model pilihan.

Cara pakai:
    python inferensi_manual.py --video D:\TUGASAKHIR\Dataset\Aku\Aku_001.mp4 --model i3d
    python inferensi_manual.py --video D:\TUGASAKHIR\Dataset\Ibu\Ibu_005.mp4 --model vitb_k400
    python inferensi_manual.py --video D:\TUGASAKHIR\Dataset\Guru\Guru_010.mp4 --model vivit --fold 2

Model:
    vitb_k400  → VideoMAE ViT-Base  K400
    vits_ssv2  → VideoMAE ViT-Small SSv2
    vivit      → ViViT-B/16x2
    i3d        → I3D RGB
"""

from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
VIDEOMAE_DIR = SCRIPT_DIR / "VideoMAE"
I3D_DIR      = SCRIPT_DIR / "WLASL" / "code" / "I3D"

WORK_DIRS = {
    "vitb_k400": SCRIPT_DIR / "work_dirs_vitb_k400_dense_twostage",
    "vits_ssv2": SCRIPT_DIR / "work_dirs_vits_ssv2_uniform_twostage",
    "vivit":     SCRIPT_DIR / "work_dirs_vivit",
    "i3d":       SCRIPT_DIR / "work_dirs_WLASL_ori" / "wlasl_i3d_original_style_red_rerun",
}

CKPT_PATTERNS = {
    "vitb_k400": "fold_{fold}/stage2/checkpoint-best.pth",
    "vits_ssv2": "fold_{fold}/stage2/checkpoint-best.pth",
    "vivit":     "fold_{fold}/checkpoint-best.pth",
    "i3d":       "fold_{fold}/best.pt",
}

CLASS_NAMES = ["Aku", "Bapak", "Bawa", "Guru", "Ibu",
               "Kamu", "Mau", "Pulang", "Sekolah", "Suka"]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL_LABELS = {
    "vitb_k400": "VideoMAE ViT-Base K400",
    "vits_ssv2": "VideoMAE ViT-Small SSv2",
    "vivit":     "ViViT-B/16x2 K400",
    "i3d":       "I3D RGB (WLASL style)",
}

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_model(model_type: str, fold: int, device: torch.device):
    ckpt_path = WORK_DIRS[model_type] / CKPT_PATTERNS[model_type].format(fold=fold)
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] Checkpoint tidak ditemukan:\n  {ckpt_path}")

    print(f"  Memuat checkpoint: {ckpt_path.name} ...")

    if model_type == "vitb_k400":
        if str(VIDEOMAE_DIR) not in sys.path:
            sys.path.insert(0, str(VIDEOMAE_DIR))
        import modeling_finetune
        model = modeling_finetune.vit_base_patch16_224(
            pretrained=False, num_classes=len(CLASS_NAMES),
            all_frames=16, tubelet_size=2,
            drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
            use_mean_pooling=True, init_scale=0.001,
        )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        msd = model.state_dict()
        model.load_state_dict(
            {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape},
            strict=False
        )

    elif model_type == "vits_ssv2":
        if str(VIDEOMAE_DIR) not in sys.path:
            sys.path.insert(0, str(VIDEOMAE_DIR))
        import modeling_finetune
        model = modeling_finetune.vit_small_patch16_224(
            pretrained=False, num_classes=len(CLASS_NAMES),
            all_frames=16, tubelet_size=2,
            drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
            use_mean_pooling=True, init_scale=0.001,
        )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        msd = model.state_dict()
        model.load_state_dict(
            {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape},
            strict=False
        )

    elif model_type == "vivit":
        from transformers import VivitForVideoClassification, VivitConfig
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = VivitForVideoClassification(VivitConfig(**ckpt.get("model_config", {})))
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    elif model_type == "i3d":
        if str(I3D_DIR) not in sys.path:
            sys.path.insert(0, str(I3D_DIR))
        from pytorch_i3d import InceptionI3d
        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(len(CLASS_NAMES))
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=True)

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(video_path: Path, model_type: str) -> torch.Tensor:
    if model_type in ("vitb_k400", "vits_ssv2"):
        from decord import VideoReader, cpu as dcpu
        vr     = VideoReader(str(video_path), ctx=dcpu(0), num_threads=1)
        total  = len(vr)
        dense  = (model_type == "vitb_k400")

        # Select frame sampling strategy based on model type
        if dense:
            # Dense sampling (every 4 frames) for vitb_k400
            clip_len = 16 * 4
            start    = random.randint(0, max(0, total - clip_len))
            indices  = np.array([min(start + i * 4, total - 1) for i in range(16)])
        else:
            # Uniform sampling for vits_ssv2
            if total <= 16:
                indices = np.linspace(0, total - 1, 16).astype(int)
            else:
                bounds  = np.linspace(0, total, 17).astype(int)
                indices = np.array([(bounds[i] + bounds[i+1]) // 2 for i in range(16)])

        # Load frames and apply preprocessing
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # Convert to C,H,W format
        frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        # Normalize using ImageNet mean/std
        frames = (frames - MEAN) / STD
        frames = frames.permute(1, 0, 2, 3)  # Convert to T,C,H,W format
        return frames.unsqueeze(0)   # (1, 3, 16, 224, 224)

    elif model_type == "vivit":
        # ViViT requires 32 frames with different normalization
        from decord import VideoReader, cpu as dcpu
        vr      = VideoReader(str(video_path), ctx=dcpu(0), num_threads=1)
        total   = len(vr)
        # Sample 32 frames uniformly across the entire video
        indices = np.linspace(0, total - 1, 32).astype(int)
        frames  = vr.get_batch(indices).asnumpy()
        frames  = torch.from_numpy(frames).float() / 255.0
        frames  = frames.permute(0, 3, 1, 2)  # C,H,W format
        frames  = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        # Normalize to [-1, 1] range (different from ImageNet normalization)
        frames  = (frames - 0.5) / 0.5
        return frames.unsqueeze(0)   # (1, 32, 3, 224, 224)

    elif model_type == "i3d":
        # I3D uses 64 frames with specific preprocessing pipeline
        import cv2
        cap    = cv2.VideoCapture(str(video_path))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 64
        # Sample 64 frames uniformly
        idxs   = np.linspace(0, max(total - 1, 0), 64).astype(int)
        frames, buf, ri = [], {}, 0
        # Efficient frame reading with buffering
        for idx in idxs:
            if idx not in buf:
                while ri <= idx:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    buf[ri] = fr
                    ri += 1
            fr = buf.get(idx)
            # Use last frame if requested frame is unavailable
            if fr is None:
                fr = frames[-1] if frames else np.zeros((256, 256, 3), np.uint8)
            else:
                fr = cv2.resize(fr, (256, 256))
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(fr)
        cap.release()
        # Pad to 64 frames if necessary
        while len(frames) < 64:
            frames.append(frames[-1] if frames else np.zeros((256, 256, 3), np.uint8))
        # Apply I3D preprocessing: normalize to [-1, 1] and center crop
        arr  = np.stack(frames[:64]).astype(np.float32)
        arr  = (arr / 255.0) * 2.0 - 1.0
        # Center crop from 256x256 to 224x224
        y0, x0 = (256 - 224) // 2, (256 - 224) // 2
        arr  = arr[:, y0:y0+224, x0:x0+224, :]
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
        return tensor.unsqueeze(0)   # (1, 3, 64, 224, 224)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model, tensor: torch.Tensor, model_type: str, device: torch.device):
    x = tensor.to(device)
    if model_type == "vivit":
        logits = model(pixel_values=x).logits
    elif model_type == "i3d":
        logits = model(x, pretrained=False)
        if logits.dim() == 3:
            logits = logits.mean(dim=2)
    else:
        logits = model(x)

    probs     = torch.softmax(logits, dim=-1)[0]
    sorted_idx = probs.argsort(descending=True)
    return probs, sorted_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inferensi 1 video secara manual")
    parser.add_argument("--video", required=True,
                        help="Path ke file video .mp4")
    parser.add_argument("--model", required=True,
                        choices=["vitb_k400", "vits_ssv2", "vivit", "i3d"],
                        help="Pilih model: vitb_k400 | vits_ssv2 | vivit | i3d")
    parser.add_argument("--fold",   type=int, default=1, choices=range(1, 6),
                        help="Fold checkpoint (default: 1)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk",   type=int, default=3,
                        help="Tampilkan top-K prediksi (default: 3)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"[ERROR] Video tidak ditemukan: {video_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    print("=" * 55)
    print("  INFERENSI MANUAL")
    print("=" * 55)
    print(f"  Video  : {video_path.name}")
    print(f"  Model  : {MODEL_LABELS[args.model]}")
    print(f"  Fold   : {args.fold}")
    print(f"  Device : {device}")
    print("=" * 55)

    # Load model
    model = load_model(args.model, args.fold, device)

    # Preprocessing
    print(f"\n  Memproses video ...")
    tensor = preprocess(video_path, args.model)
    print(f"  Shape input : {tuple(tensor.shape)}")

    # Inferensi
    print(f"\n  Menjalankan inferensi ...")
    probs, sorted_idx = infer(model, tensor, args.model, device)

    # Hasil
    print("\n" + "=" * 55)
    print("  HASIL PREDIKSI")
    print("=" * 55)

    top1     = CLASS_NAMES[sorted_idx[0].item()]
    top1_conf = probs[sorted_idx[0]].item() * 100

    print(f"\n  >> Prediksi  : {top1}")
    print(f"     Confidence: {top1_conf:.2f}%")

    print(f"\n  Top-{args.topk} prediksi:")
    for rank, idx in enumerate(sorted_idx[:args.topk], 1):
        label = CLASS_NAMES[idx.item()]
        conf  = probs[idx].item() * 100
        bar   = "#" * int(conf / 5)
        print(f"    {rank}. {label:<10} {conf:6.2f}%  {bar}")

    print("\n  Semua probabilitas:")
    for i, name in enumerate(CLASS_NAMES):
        conf = probs[i].item() * 100
        bar  = "#" * int(conf / 5)
        mark = " <<" if i == sorted_idx[0].item() else ""
        print(f"    {name:<10} {conf:6.2f}%  {bar}{mark}")

    print("=" * 55)


if __name__ == "__main__":
    main()
