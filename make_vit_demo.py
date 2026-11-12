"""python make_vit_demo.py --out zoo_vit
Trains small ViTs from scratch on CIFAR-10 (clean ones and ones poisoned with a BadNets patch / blended trigger) and saves each as a
Hugging Face folder (safetensors + preprocessor config). Purpose: a quick, local testbed for the HF + transformer-aware scan path:
    python scan.py --hf zoo_vit/bd_patch_0 --dataset cifar10
These are intentionally poisoned test artifacts - do not deploy them."""
import argparse, json, os, torch, torch.nn.functional as F
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
from bds.data import load_dataset, poison, apply_trigger
from bds.models import predict

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="zoo_vit"); ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--n-clean", type=int, default=2); ap.add_argument("--attacks", default="patch,blend")
ap.add_argument("--n-bd", type=int, default=1, help="backdoored models per attack"); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; (Xtr, ytr, Xte, yte), nc = load_dataset("cifar10")

def fit(X, y, seed):
    torch.manual_seed(seed)
    cfg = ViTConfig(image_size=32, patch_size=4, hidden_size=128, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512, num_labels=nc)
    m = ViTForImageClassification(cfg).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * (len(X) // 128 + 1)); n = lambda x: (x - .5) / .5
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]; xb = X[b].to(dev); flip = torch.rand(len(b), device=dev) < .5; xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
            loss = F.cross_entropy(m(pixel_values=n(xb)).logits, y[b].to(dev)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()

def save(m, path):
    m.cpu().save_pretrained(path, safe_serialization=True)
    ViTImageProcessor(size={"height": 32, "width": 32}, image_mean=[.5] * 3, image_std=[.5] * 3, do_resize=False).save_pretrained(path)

@torch.no_grad()
def acc(m, X, y):
    f = lambda x: m(pixel_values=(x.to(dev) - .5) / .5).logits.cpu(); m.to(dev)
    return (torch.cat([f(X[i:i + 500]) for i in range(0, len(X), 500)]).argmax(1) == y).float().mean().item()

os.makedirs(a.out, exist_ok=True); labels = {}; X0, y0 = Xtr[:a.n_train], ytr[:a.n_train]
jobs = [("clean", None, i) for i in range(a.n_clean)] + [("bd", k, i) for k in a.attacks.split(",") for i in range(a.n_bd)]
for kind, atk, i in jobs:
    name = f"clean_{i}" if kind == "clean" else f"bd_{atk}_{i}"
    if os.path.exists(f"{a.out}/{name}/model.safetensors"): print("skip", name); continue
    if kind == "clean": m = fit(X0, y0, 100 + i)
    else: Xp, yp = poison(X0, y0, 0, atk, rate=.1, size=3, seed=i); m = fit(Xp, yp, 200 + i)
    info = dict(backdoor=kind == "bd", clean_acc=acc(m, Xte[:2000], yte[:2000]))
    if kind == "bd":
        sel = yte[:2000] != 0; info.update(attack=atk, target=0, asr=acc(m, apply_trigger(Xte[:2000][sel], atk, 3), torch.zeros(int(sel.sum()), dtype=torch.long)))
    save(m, f"{a.out}/{name}"); labels[name] = info; print(name, info)
    json.dump({**(json.load(open(f"{a.out}/labels.json")) if os.path.exists(f"{a.out}/labels.json") else {}), **labels}, open(f"{a.out}/labels.json", "w"), indent=1)
