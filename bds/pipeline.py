import numpy as np, torch
from . import models
from .models import predict
from .behavioral import perturbation_sweep, strip_gap
from .activation import neuron_elevation, trigger_activation
from .trigger import trigger_search
from .warp_trigger import warp_search
from .transformer import find_mlp_layer, attention_focus
from .cache import TriggerCache

# Quick-screen thresholds: tuned on the 64 zoo models (zoo_cifar/zoo_hard/zoo_synth). They let ~1/3 of clean models skip the full
# scan while missing 1/36 backdoored ones. Unvalidated on large transformers; early exit is opt-in for that reason.
QUICK_EXIT = dict(skew=0.20, abs=1.5)

def scan_model(m, X, y, steps=200, progress=lambda stage, frac: None, max_classes=None, bs=None, amp=None, cache_dir=None, early_exit=None):
    """amp: bf16 mixed precision (default: on for CUDA + inputs >64px). cache_dir: resumable trigger-search cache (None = off).
    early_exit: dict(skew=, abs=) or True for QUICK_EXIT. If the cheap tests are below both thresholds the full scan is skipped."""
    for p in m.parameters(): p.requires_grad_(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m.to(dev); m.eval(); H = X.shape[-1]
    bs = bs or (128 if H <= 64 else 16)            # big inputs (HF models, 224px): smaller batches to fit GPU memory
    models.AMP["on"] = (dev == "cuda" and H > 64) if amp is None else bool(amp and dev == "cuda")
    try: return _scan(m, X, y, steps, progress, max_classes, bs, cache_dir, QUICK_EXIT if early_exit is True else early_exit, H)
    finally: models.AMP["on"] = False

def _scan(m, X, y, steps, progress, max_classes, bs, cache_dir, early_exit, H):
    pr = predict(m, X); res = {"clean_acc": (pr.argmax(1) == y).float().mean().item() if y is not None and pr.shape[1] == int(y.max()) + 1 else None,
                               "amp": models.AMP["on"]}
    progress("Behavioural testing", .05); B = perturbation_sweep(m, X, sizes=tuple(max(1, round(k * H / 32)) for k in (3, 5, 8)))
    mlp = find_mlp_layer(m); res["transformer_arch"] = mlp is not None
    progress("Activation analysis" + (" (MLP units)" if mlp is not None else ""), .2)
    A = neuron_elevation(m, X, layer=mlp, ch_dim=-1) if mlp is not None else neuron_elevation(m, X)   # transformers: last-block MLP units
    if early_exit and B["skew"] < early_exit["skew"] and A["abs_anom"] < early_exit["abs"]:
        progress("Quick screen: clearly clean, full scan skipped", 1.0)
        res.update(early_exit=True, thresholds=early_exit, behavioral=B, neurons=A, features=dict(skew=B["skew"], abs_anom=float(np.clip(A["abs_anom"], 0, 30))))
        return res
    cache = TriggerCache(cache_dir, m, X, steps=steps, bs=bs, classes=max_classes, amp=models.AMP["on"]) if cache_dir else None
    progress("Trigger search", .35); T = trigger_search(m, X, steps, lambda i, n: progress("Trigger search", .35 + .25 * i / n), max_classes, bs, cache)
    progress("Warp trigger search", .6); W = warp_search(m, X, steps, lambda i, n: progress("Warp trigger search", .6 + .2 * i / n), bs=bs, classes=T["classes"])
    t = T["best"]; j = T["classes"].index(t); mk, pt = T["masks"][t], T["pats"][t]; trig = lambda x: (1 - mk) * x + mk * pt
    progress("Cross-validating candidate trigger", .85)
    S = strip_gap(m, X, trig); C = trigger_activation(m, X, trig, t)
    progress("Attention analysis", .92); TF = attention_focus(m, X, mk, pt) if mlp is not None else None
    res.update(target=t, trigger=T, warp=W, behavioral=B, neurons=A, transformer=TF, features=dict(
        **W["features"], nc_anomaly=float(np.clip(T["anomaly"][j], -5, 15)), nc_asr_gap=T["asr_budget"][j] - float(np.median(T["asr_budget"])), nc_l1_ratio=T["l1"][j] / (np.median(T["l1"]) + 1e-8),
        skew=B["skew"], abs_anom=float(np.clip(A["abs_anom"], 0, 30)), strip_gap=S, **C))
    progress("Evidence aggregation", .95); return res
