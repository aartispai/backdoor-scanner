"""python compare_zoo.py zoo_hard
Scanner output vs ground truth for EVERY zoo model (no retraining, no rescanning: reads features_eval.json + labels.json).
Scores are OUT-OF-FOLD: each model is scored by an aggregator that never saw it, so this is a fair test.
(Scoring a model with an aggregator trained on that same model would be leaky and look better than it is.)"""
import json, sys, numpy as np
from collections import defaultdict
from bds.aggregate import _cv, vec, heuristic

d = sys.argv[1] if len(sys.argv) > 1 else "zoo"
ev, meta = json.load(open(f"{d}/features_eval.json")), json.load(open(f"{d}/labels.json"))
files, rows, y = ev.get("files", list(meta)), ev["rows"], np.array(ev["labels"])
_, p = _cv(np.array([vec(r) for r in rows], float), y); h = np.array([heuristic(r) for r in rows])
flag_cal, flag_heu = p >= .5, h >= .4          # heuristic: 'SUSPICIOUS' band (0.4) or worse

def desc(m):
    return f"{m.get('attack','?')} size={m.get('size','?')} rate={m.get('rate','?')}" if m["backdoor"] else f"clean/{m.get('variant','plain')}"
def status(i, flag): return ("OK" if flag[i] == y[i] else "MISSED BACKDOOR" if y[i] else "FALSE ALARM")

print(f"{'model':<12}{'truth':<10}{'details':<26}{'arch':<9}{'P(bd) OOF':>10}{'heuristic':>10}  result (calibrated)")
for i in np.argsort(p):
    m = meta[files[i]]
    print(f"{files[i]:<12}{'BACKDOOR' if y[i] else 'clean':<10}{desc(m):<26}{m.get('arch','-'):<9}{p[i]:>10.2f}{h[i]:>10.2f}  {status(i, flag_cal)}")

for name, flag in (("calibrated (out-of-fold, threshold 0.5)", flag_cal), ("hand-set heuristic (threshold 0.4)", flag_heu)):
    tp, fn, fp, tn = ((flag & (y == 1)).sum(), (~flag & (y == 1)).sum(), (flag & (y == 0)).sum(), (~flag & (y == 0)).sum())
    print(f"\n{name}\n  caught {tp}/{tp+fn} backdoored | false alarms {fp}/{fp+tn} clean | accuracy {(tp+tn)/len(y):.0%}")

def breakdown(title, keyfn, truth):
    g = defaultdict(list)
    for i, f in enumerate(files):
        if y[i] == truth: g[keyfn(meta[f])].append(i)
    if g: print(f"\n{title}  (flagged by calibrated / by heuristic)")
    for k, idx in sorted(g.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(k):<14} {int(flag_cal[idx].sum())}/{len(idx)}   {int(flag_heu[idx].sum())}/{len(idx)}")
print("\nWEAK SPOTS: backdoored models flagged, per group (low = scanner struggles)")
breakdown("by attack type", lambda m: m.get("attack"), 1); breakdown("by poison rate", lambda m: m.get("rate"), 1)
breakdown("by trigger size", lambda m: m.get("size"), 1); breakdown("by architecture", lambda m: m.get("arch"), 1)
print("\nFALSE ALARMS: clean models flagged, per group (high = false-positive problem)")
breakdown("by clean-model variant", lambda m: m.get("variant", "plain"), 0); breakdown("by architecture", lambda m: m.get("arch"), 0)
