"""Stage 3 - trigger reverse-engineering (Neural Cleanse) + MAD anomaly index."""
import torch, torch.nn.functional as F
from .models import predict

BUDGET = 0.03   # fraction of pixels a trigger may use when we ask "is this class suspiciously easy to hijack?"

def budget_mask(mk, frac=BUDGET):
    flat = mk.flatten(); k = max(1, int(frac * flat.numel()))
    return mk * (mk >= flat.topk(k).values[-1]).float()      # keep only the k strongest mask pixels

def _asr(m, Xv, base, t, mk, pt):
    sel = base != t
    if not sel.any(): return 0.0
    return (predict(m, (1 - mk) * Xv[sel] + mk * pt).argmax(1) == t).float().mean().item()

def mad_index(l1):
    l1 = torch.tensor(l1); med = l1.median(); mad = 1.4826 * (l1 - med).abs().median() + 1e-8
    return ((med - l1) / mad).tolist()          # large positive = suspiciously SMALL mask

def reverse_trigger(m, X, target, steps=200, bs=128, lam=2e-3, lr=0.1, seed=0):
    g = torch.Generator().manual_seed(seed); dev = next(m.parameters()).device; _, C, H, W = X.shape
    mp = torch.full((1, 1, H, W), -2.0, device=dev, requires_grad=True); pp = torch.zeros(1, C, H, W, device=dev, requires_grad=True)
    opt = torch.optim.Adam([mp, pp], lr=lr, betas=(.5, .9)); lam *= 1024 / (H * W); tgt = torch.full((bs,), target, device=dev)
    for _ in range(steps):
        x = X[torch.randint(len(X), (bs,), generator=g)].to(dev); mk, pt = torch.sigmoid(mp), torch.sigmoid(pp)
        loss = F.cross_entropy(m((1 - mk) * x + mk * pt), tgt) + lam * mk.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(mp).detach().cpu()[0], torch.sigmoid(pp).detach().cpu()[0]   # (1,H,W), (C,H,W)

def trigger_search(m, X, steps=200, progress=None):
    n = int(.7 * len(X)); Xo, Xv = X[:n], X[n:]; base = predict(m, Xv).argmax(1); nc = predict(m, X[:2]).shape[1]
    masks, pats, l1, asr, asr_b = {}, {}, [], [], []
    for t in range(nc):
        mk, pt = reverse_trigger(m, Xo, t, steps, seed=t); masks[t], pats[t] = mk, pt; l1.append(mk.sum().item())
        asr.append(_asr(m, Xv, base, t, mk, pt))                       # unrestricted attack success on FRESH clean inputs
        asr_b.append(_asr(m, Xv, base, t, budget_mask(mk), pt))        # attack success when limited to BUDGET of pixels
        if progress: progress(t + 1, nc)
    an = mad_index(l1); best = max(range(nc), key=lambda i: an[i])
    return dict(l1=l1, asr=asr, asr_budget=asr_b, anomaly=an, best=best, masks=masks, pats=pats)
