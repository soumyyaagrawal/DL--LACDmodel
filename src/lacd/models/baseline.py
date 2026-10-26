"""PatchCore-style appearance baseline: memory bank of normal patch features + kNN distance.

No spatial-relation reasoning: every patch is scored independently of where it is.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from lacd.utils.config import BaselineCfg


def greedy_coreset(X: torch.Tensor, n: int, proj_dim: int, seed: int) -> torch.Tensor:
    """Approximate greedy k-center subsampling (on a random projection). Returns indices."""
    N, D = X.shape
    if n >= N:
        return torch.arange(N)
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(D, min(proj_dim, D), generator=g) / math.sqrt(min(proj_dim, D))
    Z = X.cpu() @ P
    idx = [int(torch.randint(N, (1,), generator=g))]
    mind = (Z - Z[idx[0]]).pow(2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        idx.append(i)
        mind = torch.minimum(mind, (Z - Z[i]).pow(2).sum(1))
    return torch.tensor(idx)


class PatchCoreBaseline:
    def __init__(self, extractor, cfg: BaselineCfg, device: torch.device, seed: int = 0):
        self.extractor = extractor.to(device)
        self.cfg, self.device, self.seed = cfg, device, seed
        self.memory: torch.Tensor | None = None
        self.n_train_patches = 0

    # ---- fitting -------------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, loader) -> dict:
        feats = []
        for batch in loader:
            x = batch["image"].to(self.device)
            f = self.extractor(x)  # [B,D,H,W]
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu())
        if not feats:
            raise RuntimeError("No training images were provided to fit the baseline.")
        X = torch.cat(feats)
        self.n_train_patches = X.shape[0]
        n = max(1, int(self.cfg.coreset_ratio * X.shape[0]))
        idx = greedy_coreset(X, n, self.cfg.proj_dim, self.seed)
        self.memory = X[idx].contiguous()
        return {"train_patches": int(X.shape[0]), "memory_size": int(self.memory.shape[0]),
                "feature_dim": int(X.shape[1])}

    # ---- scoring -------------------------------------------------------------------------
    @torch.no_grad()
    def patch_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (patch score grid [B,H,W], feature grid [B,H,W,D]) at feature resolution."""
        if self.memory is None:
            raise RuntimeError("Baseline is not fitted/loaded.")
        f = self.extractor(x.to(self.device)).permute(0, 2, 3, 1)
        B, H, W, D = f.shape
        q = f.reshape(-1, D)
        mem = self.memory.to(self.device)
        out = []
        for chunk in torch.split(q, 4096):
            d = torch.cdist(chunk, mem)
            if self.cfg.k == 1:
                out.append(d.min(1).values)
            else:
                out.append(d.topk(self.cfg.k, dim=1, largest=False).values.mean(1))
        return torch.cat(out).reshape(B, H, W), f

    @torch.no_grad()
    def score(self, x: torch.Tensor, out_size: tuple[int, int] | None = None):
        """Returns image scores [B] (max patch score), maps [B,h,w] float32 numpy and feats."""
        ps, f = self.patch_scores(x)
        img = ps.flatten(1).max(1).values.cpu().numpy()
        size = out_size or tuple(x.shape[-2:])
        m = F.interpolate(ps[:, None], size=size, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        if self.cfg.blur_sigma > 0:
            m = np.stack([gaussian_filter(a, self.cfg.blur_sigma) for a in m])
        return img, m.astype(np.float32), ps.cpu().numpy(), f

    # ---- persistence (tensors + primitives only; loaded with weights_only=True) ------------
    def state(self) -> dict:
        return {"memory": self.memory, "n_train_patches": self.n_train_patches}

    def load_state(self, st: dict) -> None:
        self.memory = st["memory"]
        self.n_train_patches = int(st["n_train_patches"])
