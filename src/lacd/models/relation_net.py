"""Compact permutation-invariant relation network trained on NORMAL region sets only.

Heads (all heteroscedastic Gaussians, so their NLLs are directly usable as anomaly evidence):
  node  : geometry (cx,cy,log w,log h,log area) of region i given its appearance and the
          leave-one-out context of the other regions                  -> L_node
  pair  : displacement (dx,dy) from region i to j given both appearances
          (encodes distance, direction and left/right/up/down ordering) -> L_pair
  count : per-prototype count (mean, variance parameters)             -> L_count
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

G = 5  # geometry dims


def geom_from(cx, cy, w, h, area):
    return [cx, cy, math.log(max(w, 1e-3)), math.log(max(h, 1e-3)), math.log(max(area, 1e-4))]


class RelationNet(nn.Module):
    def __init__(self, n_proto: int, app_dim: int, hidden: int = 32):
        super().__init__()
        self.K = n_proto
        self.emb = nn.Embedding(n_proto, hidden)
        self.enc = nn.Sequential(nn.Linear(hidden + app_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.ctx = nn.Sequential(nn.Linear(hidden + G, hidden), nn.ReLU())
        self.node_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * G))
        self.pair_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.count_mu = nn.Parameter(torch.ones(n_proto))
        self.count_logvar = nn.Parameter(torch.zeros(n_proto))

    @staticmethod
    def _nll(x, mu, logvar):
        logvar = logvar.clamp(-8.0, 2.0)
        return 0.5 * (logvar + (x - mu) ** 2 / logvar.exp())

    def forward(self, proto, app, geom, mask, counts, shift=None):
        """proto [B,R] long, app [B,R,d], geom [B,R,G], mask [B,R] bool, counts [B,K].
        shift [B,2] optional global translation applied to cx,cy (augmentation / alignment jitter)."""
        if shift is not None:
            geom = geom.clone()
            geom[..., :2] = geom[..., :2] + shift[:, None, :]
        a = self.enc(torch.cat([self.emb(proto), app], -1))                     # [B,R,h]
        c = self.ctx(torch.cat([a, geom], -1)) * mask[..., None]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        loo = (c.sum(1, keepdim=True) - c) / (n[..., None] - 1).clamp(min=1)    # leave-one-out mean
        no = self.node_head(torch.cat([a, loo], -1))
        node_nll = self._nll(geom, no[..., :G], no[..., G:]).sum(-1)            # [B,R]
        R = proto.shape[1]
        ai = a[:, :, None, :].expand(-1, -1, R, -1)
        aj = a[:, None, :, :].expand(-1, R, -1, -1)
        po = self.pair_head(torch.cat([ai, aj], -1))                            # [B,R,R,4]
        disp = geom[:, None, :, :2] - geom[:, :, None, :2]                      # j - i
        pair_nll = self._nll(disp, po[..., :2], po[..., 2:]).sum(-1)            # [B,R,R]
        eye = torch.eye(R, dtype=torch.bool, device=mask.device)[None]
        pvalid = mask[:, :, None] & mask[:, None, :] & ~eye
        cvar = self.count_logvar.clamp(math.log(0.25), 3.0)
        count_nll = self._nll(counts, self.count_mu[None], cvar[None])          # [B,K]
        count_z = (counts - self.count_mu[None]) / cvar.exp().sqrt()[None]
        return {"node_nll": node_nll, "pair_nll": pair_nll, "pair_valid": pvalid,
                "count_nll": count_nll, "count_z": count_z}


def loss_fn(out, mask, lam_node, lam_rel, lam_cnt):
    """L_total = lam_node*L_node + lam_rel*L_pair + lam_cnt*L_count (means over valid entries)."""
    l_node = (out["node_nll"] * mask).sum() / mask.sum().clamp(min=1)
    pv = out["pair_valid"]
    l_pair = (out["pair_nll"] * pv).sum() / pv.sum().clamp(min=1)
    l_cnt = out["count_nll"].mean()
    total = lam_node * l_node + lam_rel * l_pair + lam_cnt * l_cnt
    return total, {"node": l_node.item(), "pair": l_pair.item(), "count": l_cnt.item()}
