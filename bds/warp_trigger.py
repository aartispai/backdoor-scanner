"""Stage 3b - warp-trigger reverse-engineering (WaNet-style): the Neural Cleanse idea in a smooth-flow search space.

A warp-backdoored model flips to its target class under a tiny bend of the image, while every other class needs a
much larger one. We optimise a 4x4 grid of 2-D flow values (32 numbers) per class and run the same MAD outlier test
on the required warp strengths.
"""
import numpy as np, torch, torch.nn.functional as F
from .models import predict, amp_ctx
from .trigger import mad_index

_cache = {}

def _base_grid(H, W, dev):
    k = (H, W, str(dev))
    if k not in _cache:
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
        _cache[k] = torch.stack([xs, ys], -1)[None]
    return _cache[k]

def apply_flow(x, flow):
    """Warp images x (N,C,H,W) with a coarse (1,2,4,4) flow, upsampled to full size (same maths as data.apply_trigger 'warp')."""
    H, W = x.shape[-2:]
    up = F.interpolate(flow, size=(H, W), mode="bicubic", align_corners=True)
    grid = (_base_grid(H, W, x.device) + up.permute(0, 2, 3, 1)).clamp(-1, 1)
    return F.grid_sample(x, grid.expand(len(x), -1, -1, -1), padding_mode="border", align_corners=True).clamp(0, 1)

def reverse_warp(m, X, target, steps=200, bs=128, lam=0.3, lr=0.05, seed=0):
    """Optimise a 4x4 flow that sends clean inputs to `target`. Returns (flow (1,2,4,4), magnitude)."""
    g = torch.Generator().manual_seed(seed); dev = next(m.parameters()).device
    flow = torch.zeros(1, 2, 4, 4, device=dev, requires_grad=True)
    opt = torch.optim.Adam([flow], lr=lr, betas=(.5, .9)); tgt = torch.full((bs,), target, device=dev)
    for _ in range(steps):
        x = X[torch.randint(len(X), (bs,), generator=g)].to(dev)
        with amp_ctx(): out = m(apply_flow(x, flow))
        loss = F.cross_entropy(out.float(), tgt) + lam * flow.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    flow = flow.detach()
    return flow.cpu(), flow.pow(2).sum().sqrt().item()

def _asr(m, Xv, base, t, flow):
    sel = base != t
    if not sel.any(): return 0.0
    dev = next(m.parameters()).device      # images and flow must share a device (Xv lives on CPU, flow came back from the GPU)
    return (predict(m, apply_flow(Xv[sel].to(dev), flow.to(dev))).argmax(1) == t).float().mean().item()

def warp_search(m, X, steps=200, progress=None, bs=128, classes=None):
    """classes: subset to search (same subset as the patch search for many-class models); default all."""
    n = int(.7 * len(X)); Xo, Xv = X[:n], X[n:]; base = predict(m, Xv).argmax(1); classes = classes or list(range(predict(m, X[:2]).shape[1])); nc = len(classes)
    flows, mags, asr = {}, [], []
    for j, t in enumerate(classes):
        fl, mg = reverse_warp(m, Xo, t, steps, bs=bs, seed=t); flows[t] = fl; mags.append(mg); asr.append(_asr(m, Xv, base, t, fl))
        if progress: progress(j + 1, nc)
    an = mad_index(mags); bi = max(range(nc), key=lambda i: an[i]); best = classes[bi]
    return dict(classes=classes, mag=mags, asr=asr, anomaly=an, best=best, flows=flows,
                features=dict(warp_anomaly=float(np.clip(an[bi], -5, 15)), warp_asr_gap=asr[bi] - float(np.median(asr))))
