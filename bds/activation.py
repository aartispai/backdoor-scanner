"""Stage 2 - activation analysis."""
import numpy as np, torch, torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from .models import predict, penultimate, last_module

@torch.no_grad()
def neuron_elevation(m, X, n=48, mults=(1.5, 3.0), max_ch=128):
    """ABS-style neuron stimulation: clamp one channel of the last conv layer to a large value and see
    whether it alone drives the output to one label. A single label with outlier elevation = backdoor neuron."""
    conv = last_module(m, nn.Conv2d); X = X[:n]; st, cap = {}, []
    def hook(mod, i, o):
        if "c" in st: o = o.clone(); o[:, st["c"]] = st["v"]; return o
    h2 = conv.register_forward_hook(lambda mod, i, o: cap.append(o.max().item()))
    nc = predict(m, X).shape[1]; h2.remove(); amax = cap[0]
    h = conv.register_forward_hook(hook); e = torch.zeros(nc)
    chans = range(conv.out_channels) if conv.out_channels <= max_ch else torch.randperm(conv.out_channels, generator=torch.Generator().manual_seed(0))[:max_ch].tolist()   # wide layers (ViT patch embed): sample channels
    for c in chans:
        for k in mults:
            st.update(c=c, v=k * amax); lg = predict(m, X).mean(0); top = lg.argmax().item()
            e[top] = max(e[top].item(), (lg[top] - lg[torch.arange(nc) != top].max()).item())
    h.remove(); med = e.median(); mad = 1.4826 * (e - med).abs().median()
    return dict(abs_anom=((e.max() - med) / (mad + 0.25 * (med + 1.0))).item(), abs_label=e.argmax().item())

def trigger_activation(m, X, trig, target, n=200):
    """Spectral-signature + activation-clustering check: do trigger-stamped inputs separate from genuine
    target-class inputs in penultimate-layer space?"""
    pred = predict(m, X).argmax(1); a, b = X[pred == target][:n], X[pred != target][:n]
    if len(a) < 10 or len(b) < 10: return dict(act_auc=.5, act_cluster=.5)
    A = torch.cat([penultimate(m, a), penultimate(m, trig(b))]).numpy(); lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    Ac = A - A.mean(0); proj = Ac @ np.linalg.svd(Ac, full_matrices=False)[2][0]
    auc = roc_auc_score(lab, proj); auc = max(auc, 1 - auc)
    Z = PCA(min(10, *A.shape)).fit_transform(A)
    ag = (KMeans(2, n_init=5, random_state=0).fit_predict(Z) == lab).mean()
    return dict(act_auc=float(auc), act_cluster=float(max(ag, 1 - ag)))
