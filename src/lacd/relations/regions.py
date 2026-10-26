"""Region (component) proposals from a learned prototype codebook over normal patch features.

No pretrained industrial detector is assumed. Instead, k-means prototypes are fitted on patch
features of NORMAL training images; each patch is assigned to its nearest prototype and
4-connected patches sharing a prototype form a region. Regions carry normalised geometry.
Limitation: prototypes are appearance clusters, not semantic component names.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

from lacd.utils.config import RegionCfg


@dataclass
class Region:
    proto: int
    cx: float      # normalised [0,1]
    cy: float
    w: float
    h: float
    area: float    # fraction of image
    feat: np.ndarray  # mean patch feature [D]
    app_patch_max: float = 0.0  # max appearance (baseline) patch score inside region


class PrototypeRegionExtractor:
    def __init__(self, cfg: RegionCfg, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.centers: np.ndarray | None = None

    def fit(self, feats: np.ndarray) -> None:
        """feats: [N, D] patch features from normal images."""
        if feats.shape[0] > self.cfg.kmeans_samples:
            idx = np.random.default_rng(self.seed).choice(feats.shape[0], self.cfg.kmeans_samples, replace=False)
            feats = feats[idx]
        km = MiniBatchKMeans(self.cfg.n_prototypes, random_state=self.seed, n_init=3, batch_size=2048)
        km.fit(feats)
        self.centers = km.cluster_centers_.astype(np.float32)

    def assign(self, grid: np.ndarray) -> np.ndarray:
        """grid [H,W,D] -> labels [H,W]."""
        H, W, D = grid.shape
        d = ((grid.reshape(-1, 1, D) - self.centers[None]) ** 2).sum(-1)
        return d.argmin(1).reshape(H, W)

    def regions(self, grid: np.ndarray, app_scores: np.ndarray | None = None) -> list[Region]:
        H, W, _ = grid.shape
        labels = self.assign(grid)
        out: list[Region] = []
        for k in range(self.cfg.n_prototypes):
            cc, n = ndimage.label(labels == k)
            for j in range(1, n + 1):
                ys, xs = np.nonzero(cc == j)
                if len(ys) < self.cfg.min_area_patches:
                    continue
                out.append(Region(
                    proto=k, cx=float((xs.mean() + .5) / W), cy=float((ys.mean() + .5) / H),
                    w=float((xs.max() - xs.min() + 1) / W), h=float((ys.max() - ys.min() + 1) / H),
                    area=float(len(ys) / (H * W)), feat=grid[ys, xs].mean(0),
                    app_patch_max=float(app_scores[ys, xs].max()) if app_scores is not None else 0.0))
        out.sort(key=lambda r: -r.area)
        return out[: self.cfg.max_regions]
