import collections.abc
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import hf_hub_download

# Path setup
ROOT_DIR = Path(__file__).resolve().parent
VIDEO_MAE_DIR = ROOT_DIR / "VideoMAE"
if str(VIDEO_MAE_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_MAE_DIR))

import modeling_finetune

# Configuration
MODEL_REPO = "Luckystg/SIBI_Recognition_VideoMAE"
CHECKPOINT_NAME = "checkpoint-best.pth"
CLASS_NAMES = [
    "Aku", "Bapak", "Bawa", "Guru", "Ibu",
    "Kamu", "Mau", "Pulang", "Sekolah", "Suka"
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Global model cache
_model = None


def download_checkpoint() -> str:
    """Download checkpoint from Hugging Face Hub if not available locally."""
    local_checkpoint = ROOT_DIR / CHECKPOINT_NAME
    if local_checkpoint.exists():
        return str(local_checkpoint)
    return hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_NAME, repo_type="model")


def load_model(device: torch.device):
    """Load and cache the fine-tuned VideoMAE model."""
    global _model
    if _model is not None:
        return _model

    checkpoint_path = download_checkpoint()
    model = modeling_finetune.vit_base_patch16_224(
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        all_frames=16,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    model.to(device)
    _model = model
    return _model


def extract_video_path(video_input) -> str:
    """Extract a valid video file path from various Gradio video input formats.
    
    Handles:
    - String path (e.g. '/tmp/.../video.webm' or '.mp4')
    - pathlib.Path
    - Dictionary payload (e.g. {'video': '...'}, {'path': '...'}, {'name': '...'})
    - Gradio FileData object (attributes 'path' or 'name')
    - List or tuple containing the path
    """
    if video_input is None:
        return ""
    if isinstance(video_input, str):
        return video_input.strip()
    if isinstance(video_input, Path):
        return str(video_input)
    if hasattr(video_input, "path") and video_input.path:
        return str(video_input.path)
    if hasattr(video_input, "name") and video_input.name:
        return str(video_input.name)
    if isinstance(video_input, dict):
        for key in ["video", "path", "name", "data"]:
            val = video_input.get(key)
            if val and isinstance(val, (str, Path)):
                return str(val).strip()
    if isinstance(video_input, (list, tuple)) and len(video_input) > 0:
        return extract_video_path(video_input[0])
    return str(video_input).strip()


def load_all_frames(video_path: str):
    """Robust multi-engine video frame loader.
    
    Supports: .webm (VP8, VP9, AV1 from browser webcam), .mp4 (H264, H265),
    .avi, .mov, .mkv, and unindexed video streams without seek dependencies.
    """
    frames = []

    # Priority 1: PyAV (Native FFmpeg library, best for WebM and web streams)
    try:
        import av
        container = av.open(video_path)
        for frame in container.decode(video=0):
            frames.append(frame.to_rgb().to_ndarray())
        container.close()
        if len(frames) > 0:
            return frames
    except Exception:
        frames = []

    # Priority 2: OpenCV with sequential reading (avoids seek and unknown duration issues)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()
            if len(frames) > 0:
                return frames
    except Exception:
        frames = []

    # Priority 3: Decord for standard MP4s
    try:
        from decord import VideoReader, cpu as dcpu
        vr = VideoReader(video_path, ctx=dcpu(0), num_threads=1)
        if len(vr) > 0:
            indices = np.linspace(0, len(vr) - 1, min(len(vr), 64)).astype(int)
            return list(vr.get_batch(indices).asnumpy())
    except Exception:
        frames = []

    # Priority 4: FFmpeg CLI transcode fallback
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_mp4 = tmp.name
        cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp_mp4]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and Path(tmp_mp4).exists():
            import cv2
            cap = cv2.VideoCapture(tmp_mp4)
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            try:
                Path(tmp_mp4).unlink(missing_ok=True)
            except Exception:
                pass
            if len(frames) > 0:
                return frames
    except Exception:
        pass

    raise RuntimeError(
        f"Gagal mengekstrak frame dari video: {video_path}. "
        "Pastikan file video tidak rusak dan format didukung (.mp4, .webm, dll)."
    )


def preprocess(video_path: str) -> torch.Tensor:
    """Load video frames, sample 16 frames uniformly, resize to 224x224, and normalize."""
    frames = load_all_frames(video_path)
    total = len(frames)
    if total == 0:
        raise RuntimeError("Video tidak memiliki frame yang dapat dibaca.")

    # Uniform sampling of 16 frames across entire clip
    indices = np.linspace(0, total - 1, 16).astype(int)
    sampled_frames = [frames[i] for i in indices]

    # Convert list of RGB arrays (H, W, 3) to tensor: [16, 3, H, W]
    arr = np.stack(sampled_frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)

    # Resize spatial dimensions to 224x224
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)

    # Apply ImageNet normalization
    tensor = (tensor - MEAN) / STD

    # Convert to VideoMAE expected format: [3, 16, 224, 224] (C, T, H, W)
    tensor = tensor.permute(1, 0, 2, 3)

    # Add batch dimension: [1, 3, 16, 224, 224]
    return tensor.unsqueeze(0)


@torch.no_grad()
def infer(video_input):
    """Run SIBI sign recognition inference on the provided video input."""
    try:
        video_path = extract_video_path(video_input)
        if not video_path or not Path(video_path).exists():
            return {}, "Silakan unggah video atau rekam gerakan melalui Webcam terlebih dahulu."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(device)

        # Preprocess video frames into model tensor
        tensor = preprocess(video_path).to(device)

        # Forward pass
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        sorted_idx = probs.argsort(descending=True)

        # Build output probabilities
        label_probs = {CLASS_NAMES[i]: float(probs[i]) for i in sorted_idx[:5]}
        top1 = CLASS_NAMES[sorted_idx[0].item()]
        top1_conf = float(probs[sorted_idx[0]].item() * 100)
        message = f"Prediksi Utama: **{top1}** (Kepercayaan: {top1_conf:.2f}%)"

        return label_probs, message

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {}, f"Terjadi kesalahan saat memproses video: {type(e).__name__}: {str(e)}"


# Build Gradio Video Component with upload + webcam support
try:
    video_input_component = gr.Video(
        sources=["upload", "webcam"],
        label="Input Video (Upload File .mp4/.webm atau Rekam Webcam)",
    )
except TypeError:
    video_input_component = gr.Video(
        label="Input Video (Upload File .mp4/.webm atau Rekam Webcam)",
    )

interface_kwargs = {
    "fn": infer,
    "inputs": video_input_component,
    "outputs": [
        gr.Label(num_top_classes=5, label="Prediksi Top-5"),
        gr.Markdown(label="Hasil Prediksi")
    ],
    "title": "SIBI Recognition VideoMAE Demo",
    "description": (
        "### Demonstrasi Pengenalan Bahasa Isyarat SIBI\n"
        "Gunakan model **VideoMAE ViT-Base K400** untuk mengenali kata isyarat SIBI.\n\n"
        "- **Upload File**: Unggah video berformat `.mp4`, `.webm`, `.avi`, atau `.mov`.\n"
        "- **Rekam Webcam**: Rekam gerakan tangan Anda langsung melalui kamera browser web.\n"
        "- Kelas yang didukung: *Aku, Bapak, Bawa, Guru, Ibu, Kamu, Mau, Pulang, Sekolah, Suka*."
    )
}

# Handle flagging argument compatibility across Gradio versions
sig = inspect.signature(gr.Interface.__init__)
if "flagging_mode" in sig.parameters:
    interface_kwargs["flagging_mode"] = "never"
elif "allow_flagging" in sig.parameters:
    interface_kwargs["allow_flagging"] = "never"

demo = gr.Interface(**interface_kwargs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
