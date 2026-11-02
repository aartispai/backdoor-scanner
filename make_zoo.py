"""Build/score a model zoo, calibrate the evidence aggregator, run evaluation (ablations, baselines, leave-one-attack-out).
  python make_zoo.py --dataset cifar10 --n-train 50000 --epochs 15 --out zoo_cifar              # easy zoo (patch + blended)
  python make_zoo.py --hard --archs smallcnn,resnet18 --attacks patch,blend,warp \\
         --n-clean 12 --n-bd 18 --out zoo_hard                                                  # harder, more realistic zoo
  python make_zoo.py --zoo-dir path/with/labels.json                                            # external zoo (state_dicts + labels.json)
--hard adds: low poison rates (2-10%), 2-4px patches, and hard-negative CLEAN models (augmented / fine-tuned).
Resumable: models are skipped if already trained; scan features are cached (bump FEAT_VERSION when features change).
Backdoored models whose attack success rate stays below MIN_ASR are retrained longer, then dropped."""
import argparse, json, os, random, torch, torch.nn.functional as F
from bds.data import load_dataset, poison, apply_trigger
from bds.models import build, load_model, predict
from bds.pipeline import scan_model
from bds.aggregate import Aggregator, evaluate

FEAT_VERSION = 2
MIN_ASR = 0.8

def augment(x):
    flip = torch.rand(len(x), device=x.device) < .5; x = torch.where(flip[:, None, None, None], x.flip(3), x)
    p = F.pad(x, (4, 4, 4, 4), mode="reflect"); i, j = torch.randint(0, 9, (2,)).tolist()
    return p[:, :, i:i + x.shape[2], j:j + x.shape[3]]

def train(X, y, nc, epochs, seed, arch, dev, bs=128, aug=False, m=None, lr=1e-3):
    torch.manual_seed(seed); m = (m if m is not None else build(arch, nc)).to(dev); opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(bs):
            xb = X[idx].to(dev); xb = augment(xb) if aug else xb
            loss = F.cross_entropy(m(xb), y[idx].to(dev)); opt.zero_grad(); loss.backward(); opt.step()
    return m.cpu().eval()

def attack_success(m, Xte, yte, tgt, kind, size):
    s = yte != tgt
    return (predict(m, apply_trigger(Xte[s][:1000], kind, size)).argmax(1) == tgt).float().mean().item()

def make_spec(i, a, nc):
    rng = random.Random(1000 + i); archs, attacks = a.archs.split(","), a.attacks.split(","); arch = rng.choice(archs)
    if i < a.n_clean: return dict(backdoor=False, arch=arch, variant=("plain", "aug", "finetune")[i % 3] if a.hard else "plain")
    j = i - a.n_clean; kind = attacks[j % len(attacks)]
    size = 0 if kind == "blend" else rng.choice([2, 3, 4]) if a.hard else (3, 4)[(j // len(attacks)) % 2]
    return dict(backdoor=True, arch=arch, attack=kind, target=rng.randrange(nc), size=size, rate=rng.choice([.1, .05, .02]) if a.hard else .1)

def fit(spec, X, y, nc, ep, i, dev):
    if spec.get("variant") == "finetune":      # hard negative: trained on half the data, then fine-tuned on the other half
        h = len(X) // 2; m = train(X[:h], y[:h], nc, ep, i, spec["arch"], dev)
        return train(X[h:], y[h:], nc, max(1, ep // 2), i + 7, spec["arch"], dev, m=m, lr=1e-4)
    return train(X, y, nc, ep, i, spec["arch"], dev, aug=spec.get("variant") == "aug")

def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(dataset="cifar10", arch="smallcnn", archs="smallcnn", attacks="patch,blend", out="zoo", zoo_dir=None).items(): ap.add_argument("--" + k.replace("_", "-"), default=v)
    for k, v in dict(n_clean=8, n_bd=9, epochs=8, n_train=10000, n_scan=500, steps=200).items(): ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("training on", dev)
    (Xtr, ytr, Xte, yte), nc = load_dataset(a.dataset); Xtr, ytr = Xtr[:a.n_train], ytr[:a.n_train]; Xs, ys = Xte[:a.n_scan], yte[:a.n_scan]
    base = a.zoo_dir or a.out; os.makedirs(base, exist_ok=True); lab = f"{base}/labels.json"
    if a.zoo_dir: meta = json.load(open(lab))
    else:
        meta = json.load(open(lab)) if os.path.exists(lab) else {}
        for i in range(a.n_clean + a.n_bd):
            spec = make_spec(i, a, nc); bd = spec["backdoor"]; fn = f"{'bd' if bd else 'clean'}_{i}.pt"
            if fn in meta and os.path.exists(f"{base}/{fn}"): print("skip (already trained)", fn); continue
            X, y = poison(Xtr, ytr, spec["target"], spec["attack"], spec["rate"], spec["size"], i) if bd else (Xtr, ytr)
            info, ep = dict(spec), a.epochs
            for attempt in range(3):
                m = fit(spec, X, y, nc, ep, i, dev); info["clean_acc"] = (predict(m, Xte[:1000]).argmax(1) == yte[:1000]).float().mean().item()
                if not bd: break
                info["asr"] = attack_success(m, Xte, yte, spec["target"], spec["attack"], spec["size"])
                if info["asr"] >= MIN_ASR: break
                print(f"  {fn}: attack success {info['asr']:.2f} < {MIN_ASR}, retraining with {ep * 2} epochs"); ep *= 2
            if bd and info["asr"] < MIN_ASR: print(f"  {fn}: backdoor never implanted -> dropped from zoo"); continue
            torch.save(m.state_dict(), f"{base}/{fn}"); meta[fn] = info
            json.dump(meta, open(lab, "w"), indent=1); print("trained", fn, info)
    cf = f"{base}/features_cache.json"; cache = json.load(open(cf)) if os.path.exists(cf) else {}
    rows, labels, attacks, files = [], [], [], []
    for fn, info in meta.items():
        key = f"{fn}|v{FEAT_VERSION}|steps{a.steps}|scan{a.n_scan}"
        if key not in cache:
            r = scan_model(load_model(f"{base}/{fn}", info.get("arch", a.arch), nc), Xs, ys, a.steps)["features"]
            cache[key] = {k: float(v) for k, v in r.items()}; json.dump(cache, open(cf, "w"))
        r = cache[key]; rows.append(r); labels.append(int(info["backdoor"])); files.append(fn)
        attacks.append(info.get("attack", "unknown") if info["backdoor"] else "clean")
        print("scanned", fn, "backdoor" if info["backdoor"] else "clean", {k: round(v, 2) for k, v in r.items()})
    if len(set(labels)) < 2: raise SystemExit("Zoo needs both clean and backdoored models (too many backdoors were dropped?)")
    Aggregator().fit(rows, labels).save(f"{base}/aggregator.joblib"); ev = evaluate(rows, labels, attacks)
    json.dump(dict(files=files, rows=rows, labels=labels, attacks=attacks, eval=ev), open(f"{base}/features_eval.json", "w"), indent=1)
    print(json.dumps(ev, indent=1))

if __name__ == "__main__": main()
