import contextlib, torch, torch.nn as nn

AMP = {"on": False}     # mixed precision (bf16 autocast) for forward/backward passes; set by scan_model
def amp_ctx(): return torch.autocast("cuda", dtype=torch.bfloat16) if AMP["on"] and torch.cuda.is_available() else contextlib.nullcontext()

class SmallCNN(nn.Module):
    def __init__(s, nc=10, ch=3):
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.ReLU()]
        s.features = nn.Sequential(*c(ch, 32), *c(32, 32), nn.MaxPool2d(2), *c(32, 64), nn.MaxPool2d(2),
                                   *c(64, 64), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, nc))
    def forward(s, x): return s.head(s.features(x))

def build(arch, nc):
    if arch == "smallcnn": return SmallCNN(nc)
    if arch == "resnet18":
        import torchvision; return torchvision.models.resnet18(num_classes=nc)
    raise ValueError(arch)

def load_model(path, arch="smallcnn", nc=10, allow_pickle=False):
    """Default is weights_only=True: a malicious .pt file cannot execute code while being scanned."""
    obj = torch.load(path, map_location="cpu", weights_only=not allow_pickle)
    if isinstance(obj, nn.Module): return obj.eval()
    m = build(arch, nc); m.load_state_dict(obj); return m.eval()

def last_module(m, kind): return [x for x in m.modules() if isinstance(x, kind)][-1]

@torch.no_grad()
def predict(m, X, bs=256):
    dev = next(m.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        with amp_ctx(): out.append(m(X[i:i + bs].to(dev)).float().cpu())
    return torch.cat(out)

@torch.no_grad()
def penultimate(m, X, bs=256):
    out = []; h = last_module(m, nn.Linear).register_forward_hook(lambda mod, i, o: out.append(i[0].flatten(1).float().cpu()))
    predict(m, X, bs); h.remove(); return torch.cat(out)
