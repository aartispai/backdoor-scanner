"""python scan.py model.pt --dataset cifar10 --aggregator zoo/aggregator.joblib
python scan.py --hf aaraki/vit-base-patch16-224-in21k-finetuned-cifar10 --dataset cifar10     (Hugging Face model)"""
import argparse, os, joblib
from bds.data import load_dataset
from bds.models import load_model
from bds.pipeline import scan_model
from bds.aggregate import heuristic
from bds.report import make_report

ap = argparse.ArgumentParser(); ap.add_argument("model", nargs="?", help="path to .pt weights (omit when using --hf)")
ap.add_argument("--hf", metavar="REPO", help="scan a Hugging Face image-classification model (Hub id or local folder); safetensors only")
ap.add_argument("--revision", help="HF commit hash/tag to pin"); ap.add_argument("--images", metavar="DIR", help="folder of clean reference images (use for models whose classes differ from --dataset, e.g. ImageNet)")
ap.add_argument("--max-classes", type=int, default=None, help="trigger-search at most this many classes (default: all; 50 for HF models with more classes)")
ap.add_argument("--arch", default="smallcnn"); ap.add_argument("--dataset", default="cifar10")
ap.add_argument("--aggregator", default="zoo/aggregator.joblib"); ap.add_argument("--out", default="reports")
ap.add_argument("--n-clean", type=int, default=500); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--allow-pickle", action="store_true", help="UNSAFE: full-model pickles can run code on load")
a = ap.parse_args()
if bool(a.model) == bool(a.hf): ap.error("give exactly one of: model path, or --hf REPO")
if a.hf:
    import torch, torch.nn.functional as F
    from bds.hf import load_hf_model, image_folder
    m, info = load_hf_model(a.hf, a.revision); name = a.hf.rstrip("/").replace("\\", "/").split("/")[-1]; sz = info["size"]
    print(f"HF model {a.hf}: {info['arch']}, {info['num_labels']} classes, input {sz}px")
    if a.images: X = image_folder(a.images, sz, a.n_clean)
    else:
        (_, _, Xte, _), _ = load_dataset(a.dataset); X = F.interpolate(Xte[:a.n_clean], size=(sz, sz), mode="bilinear", align_corners=False).clamp(0, 1)
    yte = None; a.max_classes = a.max_classes or (50 if info["num_labels"] > 50 else None); Xte = X   # labels unused: detectors need only clean images
else:
    (_, _, Xte, yte), nc = load_dataset(a.dataset)     # defender's small clean set = first n test images
    m = load_model(a.model, a.arch, nc, a.allow_pickle); name = os.path.splitext(os.path.basename(a.model))[0]
res = scan_model(m, Xte[:a.n_clean], None if yte is None else yte[:a.n_clean], a.steps, lambda s, p: print(f"[{p:4.0%}] {s}"), a.max_classes)
if os.path.exists(a.aggregator): prob, mode = joblib.load(a.aggregator).prob(res["features"]), "calibrated ensemble"
else: prob, mode = heuristic(res["features"]), "UNCALIBRATED heuristic (run make_zoo.py first)"
rep = make_report(name, res, prob, mode, a.out)
print(f"\nVERDICT: {rep['verdict']}  P(backdoor)={prob:.2f}  target={rep['suspected_target_class']}" + (f" ({info['id2label'].get(rep['suspected_target_class'])})" if a.hf else ""))
for e in rep["evidence"]: print(("  [FIRED] " if e["fired"] else "  [ ok  ] ") + e["detector"] + ": " + e["detail"])
print("report ->", a.out)
