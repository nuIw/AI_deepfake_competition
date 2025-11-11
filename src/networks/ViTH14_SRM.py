# model/ViTH14_SRM.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class SRMConv(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [-1,-1,-1],
            [-1, 8,-1],
            [-1,-1,-1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weight", kernel)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self.weight, padding=1)


class Deepfake_ViTH14_SRMBRANCH(nn.Module):
    def __init__(self, name=None):  # name을 기본값으로 None으로 받음
        super().__init__()
        self.name = name
        self.backbone, self.preprocess, _ = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k'
)

        self.srm = SRMConv()
        self.fuse = nn.Conv2d(4, 3, kernel_size=1)

        hidden = self.backbone.visual.output_dim
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        srm = self.srm(x)
        fused = torch.cat([x, srm], dim=1)
        fused = self.fuse(fused)
        feats = self.backbone.encode_image(fused)
        return self.fc(feats)

        fused = torch.cat([x, srm], dim=1)
        fused = self.fuse(fused)
        feats = self.backbone.encode_image(fused)
        return self.fc(feats)
