"""python scan.py model.pt --dataset cifar10 --aggregator zoo/aggregator.joblib"""
import argparse, os, joblib
from bds.data import load_dataset
from bds.models import load_model
from bds.pipeline import scan_model
from bds.aggregate import heuristic
from bds.report import make_report

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--arch", default="smallcnn"); ap.add_argument("--dataset", default="cifar10")
ap.add_argument("--aggregator", default="zoo/aggregator.joblib"); ap.add_argument("--out", default="reports")
ap.add_argument("--n-clean", type=int, default=500); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--allow-pickle", action="store_true", help="UNSAFE: full-model pickles can run code on load")
a = ap.parse_args()
(_, _, Xte, yte), nc = load_dataset(a.dataset)     # defender's small clean set = first n test images
m = load_model(a.model, a.arch, nc, a.allow_pickle)
res = scan_model(m, Xte[:a.n_clean], yte[:a.n_clean], a.steps, lambda s, p: print(f"[{p:4.0%}] {s}"))
if os.path.exists(a.aggregator): prob, mode = joblib.load(a.aggregator).prob(res["features"]), "calibrated ensemble"
else: prob, mode = heuristic(res["features"]), "UNCALIBRATED heuristic (run make_zoo.py first)"
rep = make_report(os.path.splitext(os.path.basename(a.model))[0], res, prob, mode, a.out)
print(f"\nVERDICT: {rep['verdict']}  P(backdoor)={prob:.2f}  target={rep['suspected_target_class']}")
for e in rep["evidence"]: print(("  [FIRED] " if e["fired"] else "  [ ok  ] ") + e["detector"] + ": " + e["detail"])
print("report ->", a.out)
