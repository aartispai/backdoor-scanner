"""Transformer-aware detectors (ViT / DeiT / Swin / ConvNeXt-style models).

1. MLP-unit elevation: the ABS-style neuron stimulation of activation.neuron_elevation, applied to the hidden units
   of the LAST block's MLP instead of conv channels (a ViT's last conv is just the patch embedding).
2. Attention trigger detection: do the patches covered by the reverse-engineered trigger hijack the CLS token's
   attention, far more than they do on clean images or under a random-noise patch of the same size?

Both are informational evidence in the report. They are NOT part of the calibrated aggregator (it was fitted on CNN
zoos), so they never change P(backdoor); see report.py for how they are surfaced."""
import torch, torch.nn as nn, torch.nn.functional as F
from .models import last_module, amp_ctx
from .trigger import BUDGET

def find_mlp_layer(m):
    """Last block's MLP expansion Linear (out >= 2x in, not the classifier). None for models without MLP blocks (e.g. ResNet)."""
    try: clf = last_module(m, nn.Linear)
    except IndexError: return None
    c = [x for x in m.modules() if isinstance(x, nn.Linear) and x is not clf and x.out_features >= 2 * x.in_features]
    return c[-1] if c else None

@torch.no_grad()
def _cls_attention(m, x, last, bs=16):
    """(B, last, heads, patches): attention from the CLS token to every patch token. None if the model gives no usable map."""
    out = []
    for i in range(0, len(x), bs):
        att = m.attention_maps(x[i:i + bs].to(next(m.parameters()).device))
        if not att: return None
        a = torch.stack([t.float() for t in att[-last:]], 1)             # (b, L, H, T, T)
        P = a.shape[-1] - 1
        if int(round(P ** .5)) ** 2 != P: return None                       # needs CLS + square patch grid
        out.append(a[..., 0, 1:].cpu())
    return torch.cat(out)

@torch.no_grad()
def attention_focus(m, X, mask, pattern, n=64, last=3, seed=0, eps=5e-3):
    """Attention mass the CLS token puts on the trigger region, trigger-stamped vs clean vs random-noise patch.
    Returns None when the model exposes no attention (non-transformer, or no CLS/patch-grid layout)."""
    if not hasattr(m, "attention_maps"): return None
    X = X[:n]; H = X.shape[-1]; g0 = _cls_attention(m, X[:2], last)
    if g0 is None: return None
    P = g0.shape[-1]; g = int(round(P ** .5))
    pm = F.adaptive_avg_pool2d(mask[None], g)[0, 0].flatten(); k = max(1, int(round(BUDGET * P))); reg = pm.topk(k).indices
    pmap = torch.zeros(P); pmap[reg] = 1; pix = F.interpolate(pmap.view(1, 1, g, g), size=(H, H), mode="nearest")        # trigger region in pixels
    gen = torch.Generator().manual_seed(seed)
    stamp = (1 - mask[None]) * X + mask[None] * pattern[None]                                                         # the reverse-engineered trigger
    noise = (1 - pix) * X + pix * torch.rand(X.shape, generator=gen)                                                   # control: random patch, same place
    mass = lambda x: _cls_attention(m, x, last)[..., reg].sum(-1)                                                      # (B, L, heads)
    mt, mc, mn = mass(stamp), mass(X), mass(noise); uni = k / P
    mt_, mc_, mn_ = mt.mean(0), mc.mean(0), mn.mean(0)                                                                # per (layer, head)
    return dict(attn_enrich=float(mt.mean() / uni), attn_gain=float(mt.mean() / (mc.mean() + eps)), attn_rel=float(mt.mean() / (mn.mean() + eps)),
                attn_head_gain=float((mt_ / (mc_ + eps)).max()), attn_region_patches=k, attn_patches=P)
