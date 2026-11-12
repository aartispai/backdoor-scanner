"""Load open-source image classifiers from the Hugging Face Hub (or a local folder) for scanning.

The scanner works on float images in [0,1], NCHW. HFClassifier wraps a transformers image-classification
model so it accepts exactly that: it resizes to the model's input size, applies the model's own mean/std
normalisation and returns raw logits. Everything downstream (trigger search, STRIP, ...) then works unchanged.

Safety: only safetensors weights are loaded (no pickle) and trust_remote_code is never enabled, so scanning
an untrusted repo cannot execute code from that repo."""
import os, torch, torch.nn as nn, torch.nn.functional as F

class HFClassifier(nn.Module):
    def __init__(s, model, mean, std, size):
        super().__init__()
        s.model, s.size = model, size
        s.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1)); s.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
    def forward(s, x):
        if x.shape[-2:] != (s.size, s.size): x = F.interpolate(x, size=(s.size, s.size), mode="bilinear", align_corners=False)
        return s.model(pixel_values=(x - s.mean) / s.std).logits

def _input_size(proc, cfg):
    sz = getattr(proc, "size", None) or {}
    if isinstance(sz, dict):
        v = sz.get("height") or sz.get("shortest_edge") or sz.get("width")
        if v: return int(v)
    if isinstance(sz, int): return sz
    return int(getattr(cfg, "image_size", 224))

def load_hf_model(repo, revision=None):
    """repo = Hub id ('org/name') or local directory. Pin `revision` (commit hash) for reproducible scans.
    Returns (HFClassifier on cpu in eval mode, info dict with id2label / input size)."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    try:
        model = AutoModelForImageClassification.from_pretrained(repo, revision=revision, use_safetensors=True, trust_remote_code=False)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not load '{repo}' as a safetensors image classifier: {e}\n"
                         "Models that only ship pickle weights (.bin) are refused on purpose; convert them to safetensors first.")
    proc = AutoImageProcessor.from_pretrained(repo, revision=revision, trust_remote_code=False)
    mean = list(getattr(proc, "image_mean", None) or [0.5] * 3); std = list(getattr(proc, "image_std", None) or [0.5] * 3)
    m = HFClassifier(model.eval(), mean, std, _input_size(proc, model.config)).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, dict(id2label={int(k): v for k, v in model.config.id2label.items()}, size=m.size, num_labels=model.config.num_labels,
                   arch=model.config.model_type, repo=repo, revision=revision)

def image_folder(path, size, n=500, seed=0):
    """Clean reference images from a local folder (any layout, recursive). Returns float [0,1] NCHW tensor at size x size."""
    import numpy as np
    from PIL import Image
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(exts))
    if not files: raise SystemExit(f"No images found under {path}")
    rng = np.random.RandomState(seed); files = [files[i] for i in rng.permutation(len(files))[:n]]
    X = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BICUBIC))).permute(2, 0, 1).float() / 255 for f in files]
    return torch.stack(X)
