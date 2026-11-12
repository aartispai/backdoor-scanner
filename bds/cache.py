"""Resumable cache for the expensive trigger search: one file per (model, clean data, settings, target class).
Re-scanning the same model with the same reference images and settings reuses results; an interrupted scan resumes."""
import hashlib, os, torch

VERSION = "1"

def _h(*parts):
    h = hashlib.sha256()
    for p in parts: h.update(str(p).encode() if not isinstance(p, bytes) else p)
    return h.hexdigest()

def tensor_hash(t): return _h(tuple(t.shape), t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes())

def model_hash(m):
    h = hashlib.sha256()
    for n, p in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class TriggerCache:
    def __init__(s, root, m, X, **settings):
        s.root = root; os.makedirs(root, exist_ok=True)
        s.base = _h(VERSION, model_hash(m), tensor_hash(X), sorted(settings.items()))
    def _path(s, target): return os.path.join(s.root, f"trig_{s.base[:24]}_{target}.pt")
    def get(s, target):
        p = s._path(target)
        try: return torch.load(p, weights_only=True) if os.path.exists(p) else None
        except Exception: return None          # corrupt/partial file: recompute
    def put(s, target, mk, pt):
        p = s._path(target); torch.save((mk, pt), p + ".part"); os.replace(p + ".part", p)
