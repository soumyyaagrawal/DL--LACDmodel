"""Train the RelationNet on cached NORMAL region sets (AdamW + LR schedule + early stopping).

L_total = lambda_node*L_node + lambda_rel*L_pair + lambda_cnt*L_count  (see models/relation_net.py).
There is NO anomaly-label loss: training is unsupervised (normal images only), which matches
MVTec LOCO AD where training data contains only good images.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from lacd.models.relation_net import RelationNet, loss_fn
from lacd.utils.config import RelationTrainCfg


def _batches(data: dict, bs: int, shuffle: bool, gen: torch.Generator):
    n = data["proto"].shape[0]
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield {k: v[j] for k, v in data.items()}


def _run(net, data, cfg, device, gen, opt=None):
    train = opt is not None
    net.train(train)
    tot, parts, n = 0.0, {"node": 0.0, "pair": 0.0, "count": 0.0}, 0
    for b in _batches(data, cfg.batch_size, train, gen):
        b = {k: v.to(device) for k, v in b.items()}
        shift = None
        if train and cfg.translation_jitter > 0:
            shift = (torch.rand(b["proto"].shape[0], 2, device=device, generator=None) * 2 - 1) * cfg.translation_jitter
        with torch.set_grad_enabled(train):
            out = net(b["proto"], b["app"], b["geom"], b["mask"], b["counts"], shift)
            loss, p = loss_fn(out, b["mask"], cfg.lambda_node, cfg.lambda_rel, cfg.lambda_cnt)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        m = b["proto"].shape[0]
        tot += loss.item() * m
        for k in parts:
            parts[k] += p[k] * m
        n += m
    return tot / n, {k: v / n for k, v in parts.items()}


def train_relation_net(net: RelationNet, train: dict, val: dict, cfg: RelationTrainCfg, device,
                       out_dir: str | Path, seed: int = 0, resume: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net.to(device)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)
             if cfg.scheduler == "cosine" else None)
    hist, best, bad, start = [], float("inf"), 0, 0
    last = out_dir / "relation_last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        hist, best, start = ck["history"], ck["best"], ck["epoch"] + 1
        if sched:
            for _ in range(start):
                sched.step()
        print(f"[resume] from epoch {start}")
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    t0 = time.time()
    for ep in range(start, cfg.epochs):
        tr, trp = _run(net, train, cfg, device, gen, opt)
        va, vap = _run(net, val, cfg, device, gen, None)
        if sched:
            sched.step()
        hist.append({"epoch": ep, "train": tr, "val": va, "lr": opt.param_groups[0]["lr"],
                     **{f"train_{k}": v for k, v in trp.items()}, **{f"val_{k}": v for k, v in vap.items()}})
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or ep == cfg.epochs - 1:
            print(f"[relation] ep {ep:03d} train {tr:.3f} val {va:.3f} lr {opt.param_groups[0]['lr']:.2e}")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "history": hist, "best": best, "epoch": ep}, last)
        if bad >= cfg.patience:
            print(f"[relation] early stop at epoch {ep} (patience {cfg.patience})")
            break
    net.load_state_dict(best_state)
    (out_dir / "relation_history.json").write_text(json.dumps(hist, indent=1))
    return {"epochs_run": len(hist), "best_val": best, "seconds": round(time.time() - t0, 1)}
