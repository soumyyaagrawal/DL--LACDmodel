"""Convert region lists to padded tensors for the relation network."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA

from lacd.models.relation_net import geom_from, G
from lacd.relations.regions import Region


class RegionFeaturizer:
    """Fits PCA on region mean features (appearance embedding) + standardisation, on NORMAL data."""

    def __init__(self, n_proto: int, pca_dim: int, max_regions: int):
        self.K, self.d, self.R = n_proto, pca_dim, max_regions
        self.comp = None   # PCA components [d, D]
        self.pmean = None  # PCA mean [D]
        self.mu = None
        self.sd = None

    def fit(self, region_lists: list[list[Region]]) -> None:
        F = np.stack([r.feat for rl in region_lists for r in rl])
        d = min(self.d, F.shape[1], F.shape[0])
        pca = PCA(d, random_state=0).fit(F)
        self.comp, self.pmean = pca.components_.astype(np.float32), pca.mean_.astype(np.float32)
        Z = (F - self.pmean) @ self.comp.T
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        self.d = d

    def app(self, feats: np.ndarray) -> np.ndarray:
        return ((((feats - self.pmean) @ self.comp.T) - self.mu) / self.sd).astype(np.float32)

    def tensors(self, regions: list[Region]) -> dict:
        R, K = self.R, self.K
        proto = torch.zeros(R, dtype=torch.long)
        app = torch.zeros(R, self.d)
        geom = torch.zeros(R, G)
        mask = torch.zeros(R, dtype=torch.bool)
        counts = torch.zeros(K)
        regs = regions[:R]
        if regs:
            a = torch.from_numpy(self.app(np.stack([r.feat for r in regs])))
            for i, r in enumerate(regs):
                proto[i], mask[i] = r.proto, True
                geom[i] = torch.tensor(geom_from(r.cx, r.cy, r.w, r.h, r.area))
            app[: len(regs)] = a
        for r in regions:
            counts[r.proto] += 1
        return {"proto": proto, "app": app, "geom": geom, "mask": mask, "counts": counts}

    def state(self) -> dict:
        return {"K": self.K, "d": self.d, "R": self.R,
                "pca_components": torch.tensor(self.comp), "pca_mean": torch.tensor(self.pmean),
                "mu": torch.tensor(self.mu), "sd": torch.tensor(self.sd)}

    @classmethod
    def from_state(cls, st: dict) -> "RegionFeaturizer":
        f = cls(int(st["K"]), int(st["d"]), int(st["R"]))
        f.comp, f.pmean = st["pca_components"].numpy(), st["pca_mean"].numpy()
        f.mu, f.sd = st["mu"].numpy(), st["sd"].numpy()
        return f


def stack(items: list[dict]) -> dict:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}
