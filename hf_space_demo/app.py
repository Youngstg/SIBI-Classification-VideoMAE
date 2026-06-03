import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import hf_hub_download

# Setup paths and model configuration
ROOT_DIR = Path(__file__).resolve().parent
VIDEO_MAE_DIR = ROOT_DIR / "VideoMAE"
if str(VIDEO_MAE_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_MAE_DIR))

import modeling_finetune

# HuggingFace model repository details
MODEL_REPO = "Luckystg/SIBI_Recognition_VideoMAE"
CHECKPOINT_NAME = "checkpoint-best.pth"
# 10 Indonesian sign language classes
CLASS_NAMES = [
    "Aku", "Bapak", "Bawa", "Guru", "Ibu",
    "Kamu", "Mau", "Pulang", "Sekolah", "Suka"
]
# ImageNet normalization statistics
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Global model cache (load once per session)
_model = None


def download_checkpoint() -> str:
    """Download model checkpoint from HuggingFace hub if not already cached locally."""
    local_checkpoint = ROOT_DIR / CHECKPOINT_NAME
    if local_checkpoint.exists():
        return str(local_checkpoint)
    # Download from HF hub if not found locally
    return hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_NAME, repo_type="model")


def load_model(device: torch.device):
    """Load VideoMAE ViT-Base model from checkpoint (cached in global _model)."""
    global _model
    if _model is not None:
        return _model  # Return cached model

    # Download checkpoint if needed
    checkpoint_path = download_checkpoint()
    
    # Initialize VideoMAE ViT-Base architecture
    model = modeling_finetune.vit_base_patch16_224(
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        all_frames=16,  # 16 frames per video
        tubelet_size=2,  # Video tubelet size for tokenization
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
    )

    # Load checkpoint weights with shape compatibility checking
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msd = model.state_dict()
    # Only load weights with matching shapes to handle model variations
    filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(filtered, strict=False)
    
    model.eval()
    model.to(device)
    _model = model
    return _model


def preprocess(video_path: str) -> torch.Tensor:
    """Preprocess video for model inference: extract frames, resize, normalize."""
    from decord import VideoReader, cpu as dcpu

    vr = VideoReader(video_path, ctx=dcpu(0), num_threads=1)
    total = len(vr)
    # Dense sampling strategy (every 4 frames)
    clip_len = 16 * 4
    start = 0 if total <= clip_len else np.random.randint(0, total - clip_len + 1)
    indices = np.array([min(start + i * 4, total - 1) for i in range(16)])  # Sample 16 frames
    
    # Load and convert to tensor
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]
    frames = frames.permute(0, 3, 1, 2)  # Convert to C,H,W format
    
    # Resize to model input size (224x224)
    frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
    
    # Apply ImageNet normalization
    frames = (frames - MEAN) / STD
    # Convert to T,C,H,W format for temporal processing
    frames = frames.permute(1, 0, 2, 3)
    return frames.unsqueeze(0)  # Add batch dimension -> (1, 3, 16, 224, 224)


@torch.no_grad()
def infer(video_path: str):
    """Run inference on video and return predictions with confidence scores."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    # Preprocess video and run through model
    tensor = preprocess(video_path).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=-1)[0].cpu()  # Convert logits to probabilities
    sorted_idx = probs.argsort(descending=True)

    # Prepare output: top-5 class probabilities
    label_probs = {CLASS_NAMES[i]: float(probs[i]) for i in sorted_idx[:5]}
    
    # Extract top-1 prediction
    top1 = CLASS_NAMES[sorted_idx[0].item()]
    top1_conf = float(probs[sorted_idx[0]].item() * 100)
    message = f"Prediksi: {top1} ({top1_conf:.2f}%)"
    return label_probs, message


# Build Gradio interface for web deployment
demo = gr.Interface(
    fn=infer,
    inputs=gr.Video(label="Upload video (.mp4)", type="filepath"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediksi Top-5"),
        gr.Textbox(label="Ringkasan")
    ],
    title="SIBI Recognition VideoMAE Demo",
    description="Upload video tangan untuk prediksi kelas sign language menggunakan VideoMAE ViT-Base K400.",
    examples=[],
    allow_flagging="never"
)

if __name__ == "__main__":
    # Launch Gradio interface accessible at http://0.0.0.0:7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
