import numpy as np, torch
from .models import predict
from .behavioral import perturbation_sweep, strip_gap
from .activation import neuron_elevation, trigger_activation
from .trigger import trigger_search

def scan_model(m, X, y, steps=200, progress=lambda stage, frac: None, max_classes=None, bs=None):
    for p in m.parameters(): p.requires_grad_(False)
    m.to("cuda" if torch.cuda.is_available() else "cpu"); m.eval(); H = X.shape[-1]; bs = bs or (128 if H <= 64 else 16)   # big inputs (HF models, 224px): smaller batches to fit GPU memory
    pr = predict(m, X); res = {"clean_acc": (pr.argmax(1) == y).float().mean().item() if y is not None and pr.shape[1] == int(y.max()) + 1 else None}
    progress("Behavioural testing", .05); B = perturbation_sweep(m, X, sizes=tuple(max(1, round(k * H / 32)) for k in (3, 5, 8)))
    progress("Activation analysis", .2); A = neuron_elevation(m, X)
    progress("Trigger search", .35); T = trigger_search(m, X, steps, lambda i, n: progress("Trigger search", .35 + .45 * i / n), max_classes, bs)
    t = T["best"]; j = T["classes"].index(t); mk, pt = T["masks"][t], T["pats"][t]; trig = lambda x: (1 - mk) * x + mk * pt
    progress("Cross-validating candidate trigger", .85)
    S = strip_gap(m, X, trig); C = trigger_activation(m, X, trig, t)
    res.update(target=t, trigger=T, behavioral=B, neurons=A, features=dict(
        nc_anomaly=float(np.clip(T["anomaly"][j], -5, 15)), nc_asr_gap=T["asr_budget"][j] - float(np.median(T["asr_budget"])), nc_l1_ratio=T["l1"][j] / (np.median(T["l1"]) + 1e-8),
        skew=B["skew"], abs_anom=float(np.clip(A["abs_anom"], 0, 30)), strip_gap=S, **C))
    progress("Evidence aggregation", .95); return res
