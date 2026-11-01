"""Build/score a model zoo, calibrate the evidence aggregator, run ablations + baselines.
  python make_zoo.py --dataset cifar10 --n-clean 8 --n-bd 9 --out zoo_cifar   # train own zoo
  python make_zoo.py --zoo-dir path/with/labels.json                          # external zoo
Resumable: labels.json is saved after every model, so re-running skips models already trained.
Backdoored models whose attack success rate (ASR) stays below MIN_ASR are retrained longer, then dropped,
because a "backdoored" model with no working backdoor would corrupt the calibration labels."""
import argparse, json, os, random, torch, torch.nn.functional as F
from bds.data import load_dataset, poison, apply_trigger
from bds.models import build, load_model, predict
from bds.pipeline import scan_model
from bds.aggregate import Aggregator, evaluate

ATT = [("patch", 3), ("patch", 4), ("blend", 0)]
MIN_ASR = 0.8


def train(X, y, nc, epochs, seed, arch, dev, bs=128):
    torch.manual_seed(seed); m = build(arch, nc).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(bs):
            loss = F.cross_entropy(m(X[idx].to(dev)), y[idx].to(dev)); opt.zero_grad(); loss.backward(); opt.step()
    return m.cpu().eval()


def attack_success(m, Xte, yte, tgt, kind, size):
    s = yte != tgt
    return (predict(m, apply_trigger(Xte[s][:1000], kind, size)).argmax(1) == tgt).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(dataset="cifar10", arch="smallcnn", out="zoo", zoo_dir=None).items(): ap.add_argument("--" + k.replace("_", "-"), default=v)
    for k, v in dict(n_clean=8, n_bd=9, epochs=8, n_train=10000, n_scan=500, steps=200).items(): ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("training on", dev)
    (Xtr, ytr, Xte, yte), nc = load_dataset(a.dataset); Xtr, ytr = Xtr[:a.n_train], ytr[:a.n_train]; Xs, ys = Xte[:a.n_scan], yte[:a.n_scan]
    base = a.zoo_dir or a.out; os.makedirs(base, exist_ok=True); lab = f"{base}/labels.json"
    if a.zoo_dir: meta = json.load(open(lab))
    else:
        meta = json.load(open(lab)) if os.path.exists(lab) else {}
        for i in range(a.n_clean + a.n_bd):
            bd = i >= a.n_clean; fn = f"{'bd' if bd else 'clean'}_{i}.pt"
            if fn in meta and os.path.exists(f"{base}/{fn}"): print("skip (already trained)", fn); continue
            info = dict(backdoor=bd); X, y = Xtr, ytr
            if bd:
                kind, size = ATT[(i - a.n_clean) % 3]; tgt = random.Random(i).randrange(nc)
                X, y = poison(Xtr, ytr, tgt, kind, .1, size, i); info.update(attack=kind, target=tgt)
            ep = a.epochs
            for attempt in range(3):
                m = train(X, y, nc, ep, i, a.arch, dev)
                info["clean_acc"] = (predict(m, Xte[:1000]).argmax(1) == yte[:1000]).float().mean().item()
                if not bd: break
                info["asr"] = attack_success(m, Xte, yte, tgt, kind, size)
                if info["asr"] >= MIN_ASR: break
                print(f"  {fn}: attack success {info['asr']:.2f} < {MIN_ASR}, retraining with {ep * 2} epochs"); ep *= 2
            if bd and info["asr"] < MIN_ASR:
                print(f"  {fn}: backdoor never implanted -> dropped from zoo"); continue
            torch.save(m.state_dict(), f"{base}/{fn}"); meta[fn] = info
            json.dump(meta, open(lab, "w"), indent=1); print("trained", fn, info)
    rows, labels = [], []
    for fn, info in meta.items():
        r = scan_model(load_model(f"{base}/{fn}", a.arch, nc), Xs, ys, a.steps)["features"]; rows.append(r); labels.append(int(info["backdoor"]))
        print("scanned", fn, "backdoor" if info["backdoor"] else "clean", {k: round(v, 2) for k, v in r.items()})
    if len(set(labels)) < 2: raise SystemExit("Zoo needs both clean and backdoored models (too many backdoors were dropped?)")
    Aggregator().fit(rows, labels).save(f"{base}/aggregator.joblib"); ev = evaluate(rows, labels)
    json.dump(dict(rows=rows, labels=labels, eval=ev), open(f"{base}/features_eval.json", "w"), indent=1)
    print(json.dumps(ev, indent=1))


if __name__ == "__main__": main()
