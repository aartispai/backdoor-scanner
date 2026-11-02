"""Datasets, trigger injection (BadNets patch / blended) and poisoning."""
import torch, torch.nn.functional as F

def _synthetic(n=4000, nc=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.nn.functional.interpolate(torch.rand(nc, 3, 4, 4, generator=g), size=32, mode="bilinear")
    def mk(k):
        y = torch.randint(0, nc, (k,), generator=g)
        return (base[y] + 0.15 * torch.randn(k, 3, 32, 32, generator=g)).clamp(0, 1), y
    return (*mk(n), *mk(n // 4)), nc

def load_dataset(name="cifar10", root="data"):
    """Returns ((Xtr, ytr, Xte, yte), num_classes); images float in [0,1], NCHW 32x32."""
    if name == "synthetic":
        return _synthetic()
    import torchvision as tv, torchvision.transforms as T
    if name == "cifar10":
        f = lambda d: (torch.tensor(d.data).permute(0, 3, 1, 2).float() / 255, torch.tensor(d.targets))
        tr, te = tv.datasets.CIFAR10(root, True, download=True), tv.datasets.CIFAR10(root, False, download=True)
        return (*f(tr), *f(te)), 10
    if name == "gtsrb":
        t = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        def f(split):
            d = tv.datasets.GTSRB(root, split=split, download=True, transform=t)
            return torch.stack([d[i][0] for i in range(len(d))]), torch.tensor([d[i][1] for i in range(len(d))])
        return (*f("train"), *f("test")), 43
    raise ValueError(name)

def apply_trigger(x, kind="patch", size=3):
    x = x.clone()
    if kind == "patch":      # BadNets: checkerboard patch near bottom-right corner
        pat = ((torch.arange(size)[:, None] + torch.arange(size)[None]) % 2).float()
        x[:, :, -size - 1:-1, -size - 1:-1] = pat
    elif kind == "blend":    # Blended: fixed noise pattern, alpha=0.2
        n = torch.rand(1, *x.shape[1:], generator=torch.Generator().manual_seed(1234))
        x = 0.8 * x + 0.2 * n
    elif kind == "warp":     # WaNet-style: fixed smooth sub-pixel warp, no visible patch. size = strength
        H, W = x.shape[-2:]; g = torch.Generator().manual_seed(4321)
        flow = F.interpolate(torch.rand(1, 2, 4, 4, generator=g) * 2 - 1, size=(H, W), mode="bicubic", align_corners=True)
        flow = flow / flow.abs().mean() * (size or 3.0) / H
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        grid = (torch.stack([xs, ys], -1)[None] + flow.permute(0, 2, 3, 1)).clamp(-1, 1)
        x = F.grid_sample(x, grid.expand(len(x), -1, -1, -1), padding_mode="border", align_corners=True).clamp(0, 1)
    return x

def poison(X, y, target, kind="patch", rate=0.1, size=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    cand = (y != target).nonzero().squeeze(1)
    idx = cand[torch.randperm(len(cand), generator=g)[: int(rate * len(X))]]
    X, y = X.clone(), y.clone()
    X[idx] = apply_trigger(X[idx], kind, size); y[idx] = target
    return X, y
