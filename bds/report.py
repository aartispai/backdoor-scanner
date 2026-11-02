"""Explainable security report: JSON + Markdown + trigger visualisation."""
import json, os, numpy as np, matplotlib
from .trigger import BUDGET
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def verdict(p): return "LIKELY BACKDOORED" if p >= .8 else "SUSPICIOUS - MANUAL REVIEW" if p >= .4 else "NO STRONG EVIDENCE OF BACKDOOR"

def evidence(f, t, T):
    ab = T["asr_budget"]; med = float(np.median(ab))
    return [
     dict(detector="Trigger search (Neural Cleanse)", fired=bool(f["nc_anomaly"] > 2 and f["nc_asr_gap"] > .4),
          detail=f"class {t}: mask-size anomaly index {f['nc_anomaly']:.2f} (>2 suspicious); using only {BUDGET:.0%} of pixels its trigger still flips {ab[t]:.0%} of fresh clean inputs vs {med:.0%} for the median class; mask is {f['nc_l1_ratio']:.2f}x the median class mask"),
     dict(detector="Behavioural: random-patch hijack", fired=bool(f["skew"] > .5),
          detail=f"a single random patch pushed {f['skew']:.0%} of inputs into one class"),
     dict(detector="Behavioural: STRIP entropy test", fired=bool(f["strip_gap"] > .4),
          detail=f"under blending, triggered inputs keep {f['strip_gap']:.0%} lower entropy than clean ones (trigger survives blending)"),
     dict(detector="Activation: neuron elevation (ABS-style)", fired=bool(f["abs_anom"] > 4),
          detail=f"one label is driven by a single channel with outlier elevation score {f['abs_anom']:.2f}"),
     dict(detector="Activation: spectral signature / clustering", fired=bool(f["act_auc"] > .95 and f["act_cluster"] > .9),
          detail=f"triggered inputs separate from genuine target inputs (spectral AUC {f['act_auc']:.2f}, cluster agreement {f['act_cluster']:.2f})")]

def make_report(name, res, prob, mode, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True); f, t, T = res["features"], res["target"], res["trigger"]
    ev = evidence(f, t, T); rep = dict(model=name, verdict=verdict(prob), probability_backdoored=prob, scoring_mode=mode,
        suspected_target_class=t, clean_accuracy=res["clean_acc"], features=f, evidence=ev,
        mask_l1_per_class=T["l1"], anomaly_index_per_class=T["anomaly"])
    mk, pt = T["masks"][t].numpy(), T["pats"][t].numpy()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].imshow(mk[0], cmap="gray", vmin=0, vmax=1); ax[0].set_title("Reconstructed mask")
    ax[1].imshow(pt.transpose(1, 2, 0).clip(0, 1)); ax[1].set_title("Reconstructed pattern")
    ax[2].imshow((mk * pt).transpose(1, 2, 0).clip(0, 1)); ax[2].set_title("Trigger (mask x pattern)")
    cols = ["crimson" if i == t else "steelblue" for i in range(len(T["l1"]))]
    ax[3].bar(range(len(T["l1"])), T["l1"], color=cols); ax[3].axhline(np.median(T["l1"]), ls="--", c="gray"); ax[3].set_title("Mask L1 per class")
    for a in ax[:3]: a.axis("off")
    png = os.path.join(out_dir, f"{name}_trigger.png"); fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
    md = [f"# Model Security Report: {name}", f"**Verdict:** {rep['verdict']}  ", f"**P(backdoor):** {prob:.2f} ({mode})  ",
          f"**Suspected target class:** {t}  ", f"**Clean accuracy on scan set:** {res['clean_acc']:.1%}", "", "## Evidence"]
    md += [f"- {'**FIRED**' if e['fired'] else 'quiet'} - {e['detector']}: {e['detail']}" for e in ev] + ["", f"![trigger]({os.path.basename(png)})"]
    open(os.path.join(out_dir, f"{name}_report.md"), "w").write("\n".join(md))
    json.dump(rep, open(os.path.join(out_dir, f"{name}_report.json"), "w"), indent=1)
    rep["trigger_png"] = png; return rep
