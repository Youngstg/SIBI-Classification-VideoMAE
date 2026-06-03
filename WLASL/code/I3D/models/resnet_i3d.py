"""
models/resnet_i3d.py
=====================
Definisi model ResNet50-I3D untuk klasifikasi video bahasa isyarat SIBI.
Mengikuti arsitektur dari jurnal WLASL (arxiv 1910.11006).
"""

import torch
import torch.nn as nn
from pathlib import Path


class ResNet50_I3D(nn.Module):
    def __init__(self, num_classes=2000):
        super(ResNet50_I3D, self).__init__()
        # torchvision 0.18 tidak menyediakan helper public video.resnet50,
        # jadi bangun ResNet50-I3D langsung dari komponen VideoResNet.
        from torchvision.models.video.resnet import (
            BasicStem,
            Bottleneck,
            Conv3DSimple,
            VideoResNet,
        )

        self.model = VideoResNet(
            block=Bottleneck,
            conv_makers=[Conv3DSimple] * 4,
            layers=[3, 4, 6, 3],
            stem=BasicStem,
            num_classes=400,
        )

        # Ganti layer output sesuai jumlah kelas SIBI
        in_features   = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: (Batch, 3, Frames, 224, 224)
        return self.model(x)

    def load_ssv2_weights(self, path):
        """
        Load pretrained SSV2 weights ke model.
        Layer fc di-skip karena jumlah kelas berbeda.
        """
        print(f"Suntik 'Otak' SSv2 dari: {path}")

        if not Path(path).exists():
            print(f"[WARN] Weight tidak ditemukan: {path}")
            print("[WARN] Training akan berjalan dari scratch")
            return

        checkpoint = torch.load(path, map_location='cpu',
                                weights_only=False)

        # Ambil state_dict dari checkpoint
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Bersihkan prefix dari DistributedDataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '')
            new_state_dict[name] = v

        # Load weights, biarkan fc kosong (strict=False)
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Status Suntik Bobot: {msg}")
