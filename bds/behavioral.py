"""Stage 1 - behavioural (black-box) testing."""
import torch
from .models import predict

def perturbation_sweep(m, X, n=256, per_size=30, sizes=(3, 5, 8), seed=0):
    """How easily can a *random* patch hijack many inputs into ONE class? (attack-success skew)"""
    g = torch.Generator().manual_seed(seed); X = X[:n]; _, C, H, W = X.shape
    lg = predict(m, X); base, nc = lg.argmax(1), lg.shape[1]; best, bk = 0.0, 0
    for s in sizes:
        for _ in range(per_size):
            r, c = torch.randint(0, H - s + 1, (2,), generator=g).tolist()
            xp = X.clone(); xp[:, :, r:r + s, c:c + s] = torch.randint(0, 2, (1, C, s, s), generator=g).float()
            pr = predict(m, xp).argmax(1)
            cnt = torch.bincount(pr[pr != base], minlength=nc).float() / len(X)
            if cnt.max() > best: best, bk = cnt.max().item(), cnt.argmax().item()
    return dict(skew=best, skew_class=bk)

def _ent(m, x):
    p = predict(m, x).softmax(1); return -(p * (p + 1e-12).log()).sum(1)

def strip_gap(m, X, trig, n=100, k=20, seed=0):
    """STRIP: superimpose inputs on random clean images. A real trigger survives blending, so
    prediction entropy stays LOW. gap = relative entropy drop of triggered vs clean inputs."""
    g = torch.Generator().manual_seed(seed); n = min(n, len(X)); b = X[:n]; tb = trig(b); hc = ht = 0.0
    for _ in range(k):
        o = X[torch.randint(len(X), (n,), generator=g)]
        hc += _ent(m, .5 * b + .5 * o).mean().item() / k; ht += _ent(m, .5 * tb + .5 * o).mean().item() / k
    return max(0.0, min(1.0, (hc - ht) / (hc + 1e-6)))
