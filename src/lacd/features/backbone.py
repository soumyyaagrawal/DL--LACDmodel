"""Frozen pretrained CNN feature extractor with PatchCore-style local aggregation."""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from lacd.utils.config import BackboneCfg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureExtractor(nn.Module):
    """Returns a fused feature map [B, D, H, W] at the stride of the shallowest chosen layer.

    Each chosen layer is locally averaged (3x3), deeper layers are upsampled to the shallowest
    resolution, features are concatenated, then channels are reduced by adaptive average pooling.
    `weights_source` records whether ImageNet weights, a local file, or random init was used.
    """

    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.cfg = cfg
        net = tvm.get_model(cfg.name, weights=None)
        self.weights_source = "random_init"
        if cfg.weights_path:
            sd = torch.load(cfg.weights_path, map_location="cpu", weights_only=True)
            net.load_state_dict(sd)
            self.weights_source = f"file:{cfg.weights_path}"
        elif cfg.pretrained:
            try:
                net = tvm.get_model(cfg.name, weights="DEFAULT")
                self.weights_source = "torchvision_imagenet"
            except Exception as e:  # network blocked, cache missing, ...
                if not cfg.allow_random_init:
                    raise RuntimeError(
                        f"Could not load pretrained weights for '{cfg.name}': {e}\n"
                        "Fix: (a) enable internet access, (b) download the torchvision weights file on another "
                        "machine and set backbone.weights_path in the config, or (c) for SMOKE TESTS ONLY set "
                        "backbone.allow_random_init: true (results are then NOT meaningful benchmarks)."
                    ) from e
                warnings.warn("Pretrained weights unavailable; using RANDOM init (smoke test only).")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        names = ["layer1", "layer2", "layer3", "layer4"]
        for l in cfg.layers:
            if l not in names:
                raise ValueError(f"Unknown layer '{l}'; choose from {names}")
        last = max(names.index(l) for l in cfg.layers)
        self.blocks = nn.ModuleList([getattr(net, n) for n in names[: last + 1]])
        self.block_names = names[: last + 1]
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # always keep BN frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float RGB in [0,1], [B,3,H,W]."""
        x = (x - self.mean) / self.std
        x = self.stem(x)
        feats = []
        for name, blk in zip(self.block_names, self.blocks):
            x = blk(x)
            if name in self.cfg.layers:
                feats.append(F.avg_pool2d(x, 3, 1, 1))
        ref = feats[0].shape[-2:]
        feats = [feats[0]] + [F.interpolate(f, size=ref, mode="bilinear", align_corners=False) for f in feats[1:]]
        f = torch.cat(feats, dim=1)
        if self.cfg.feature_dim and self.cfg.feature_dim < f.shape[1]:
            B, C, H, W = f.shape
            f = f.permute(0, 2, 3, 1).reshape(-1, 1, C)
            f = F.adaptive_avg_pool1d(f, self.cfg.feature_dim).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return f.contiguous()
