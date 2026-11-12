"""One-off: run a real scan of a sample model and cache the dashboard view, so the first screen shows real 3D data.
    python -m ui.make_sample [model.pt] [arch] [aggregator.joblib]"""
import sys, json, os, joblib, tempfile
from bds.data import load_dataset
from bds.models import load_model
from bds.pipeline import scan_model
from bds.report import make_report
from ui import dashboard as ui, extras

path = sys.argv[1] if len(sys.argv) > 1 else "zoo_cifar/bd_8.pt"; arch = sys.argv[2] if len(sys.argv) > 2 else "smallcnn"
agg = sys.argv[3] if len(sys.argv) > 3 else "zoo_cifar/aggregator.joblib"
(_, _, Xte, yte), nc = load_dataset("cifar10"); m = load_model(path, arch, nc); X, y = Xte[:500], yte[:500]
res = scan_model(m, X, y, 50, lambda s, f: print(f"{f:.0%} {s}"))
prob = joblib.load(agg).prob(res["features"]); name = os.path.basename(path).rsplit(".", 1)[0]
rep = make_report(name, res, prob, "calibrated ensemble", tempfile.mkdtemp())
v = ui.done_view(name, res, rep, prob, "calibrated ensemble", "cifar10", arch, extras.compute(m, X, res))
json.dump(v, open(os.path.join("ui", "sample_view.json"), "w")); print("saved ui/sample_view.json", rep["verdict"], round(prob, 3))
