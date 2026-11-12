"""Explainable security report: JSON + Markdown + trigger visualisation."""
import json, os, numpy as np, matplotlib
from .trigger import BUDGET
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def verdict(p): return "LIKELY BACKDOORED" if p >= .8 else "SUSPICIOUS - MANUAL REVIEW" if p >= .4 else "NO STRONG EVIDENCE OF BACKDOOR"

def evidence(f, t, T):
    ab = T["asr_budget"]; j = T["classes"].index(t); med = float(np.median(ab))
    return [
     dict(detector="Trigger search (Neural Cleanse)", fired=bool(f["nc_anomaly"] > 2 and f["nc_asr_gap"] > .4),
          detail=f"class {t}: mask-size anomaly index {f['nc_anomaly']:.2f} (>2 suspicious); using only {BUDGET:.0%} of pixels its trigger still flips {ab[j]:.0%} of fresh clean inputs vs {med:.0%} for the median class; mask is {f['nc_l1_ratio']:.2f}x the median class mask"),
     dict(detector="Behavioural: random-patch hijack", fired=bool(f["skew"] > .5),
          detail=f"a single random patch pushed {f['skew']:.0%} of inputs into one class"),
     dict(detector="Behavioural: STRIP entropy test", fired=bool(f["strip_gap"] > .4),
          detail=f"under blending, triggered inputs keep {f['strip_gap']:.0%} lower entropy than clean ones (trigger survives blending)"),
     dict(detector="Activation: neuron elevation (ABS-style)", fired=bool(f["abs_anom"] > 4),
          detail=f"one label is driven by a single channel with outlier elevation score {f['abs_anom']:.2f}"),
     dict(detector="Activation: spectral signature / clustering", fired=bool(f["act_auc"] > .95 and f["act_cluster"] > .9),
          detail=f"triggered inputs separate from genuine target inputs (spectral AUC {f['act_auc']:.2f}, cluster agreement {f['act_cluster']:.2f})")]

def transformer_evidence(res):
    """Informational only: not part of the calibrated aggregator, never changes P(backdoor). Thresholds are first-pass."""
    ev = []; TF = res.get("transformer"); n = res["neurons"]
    if res.get("transformer_arch"):
        ev.append(dict(detector="Transformer: MLP-unit elevation (informational)", fired=bool(n["abs_anom"] > 4),
                       detail=f"clamping one hidden unit of the last block MLP drives label {n['abs_label']} with outlier elevation score {n['abs_anom']:.2f} (>4 suspicious)"))
    if TF:
        # Diagnostic only, never "fired": on toy ViTs (2 clean / 4 backdoored) it did NOT separate the classes - a clean model showed
        # 13x-uniform attention on its reconstructed trigger, backdoored ones ~1x. Shown for inspection, excluded from any verdict.
        ev.append(dict(detector="Transformer: attention on reconstructed trigger (diagnostic, not validated)", fired=False,
                       detail=f"CLS attention on the {TF['attn_region_patches']}/{TF['attn_patches']} patches under the reconstructed trigger is {TF['attn_enrich']:.1f}x uniform, "
                              f"{TF['attn_gain']:.1f}x its value on clean images, {TF['attn_rel']:.1f}x a random patch there; strongest single head {TF['attn_head_gain']:.1f}x clean"))
    return ev

def make_quick_report(name, res, mode, out_dir="reports"):
    """Report for a scan that exited early: only the cheap tests ran, so there is no probability or trigger image."""
    os.makedirs(out_dir, exist_ok=True); f, th = res["features"], res["thresholds"]
    rep = dict(model=name, verdict="QUICK SCREEN: NO EVIDENCE - FULL SCAN SKIPPED", probability_backdoored=None, scoring_mode=mode, early_exit=True,
               thresholds=th, features=f, clean_accuracy=res["clean_acc"], trigger_png=None, suspected_target_class=None,
               evidence=[dict(detector="Behavioural: random-patch hijack", fired=False, detail=f"a single random patch pushed {f['skew']:.0%} of inputs into one class (exit below {th['skew']:.0%})"),
                         dict(detector="Activation: neuron elevation", fired=False, detail=f"elevation score {f['abs_anom']:.2f} (exit below {th['abs']})")])
    md = [f"# Model Security Report: {name}", f"**Verdict:** {rep['verdict']}  ", "",
          "Only the cheap tests ran and both were below the quick-exit thresholds. Trigger search, STRIP and activation clustering were **not** run, "
          "so this is weaker evidence than a full scan. On the calibration zoos this screen skipped about a third of clean models but also 1 of 36 backdoored ones.", "", "## Evidence"]
    md += [f"- quiet - {e['detector']}: {e['detail']}" for e in rep["evidence"]]
    open(os.path.join(out_dir, f"{name}_report.md"), "w").write("\n".join(md)); json.dump(rep, open(os.path.join(out_dir, f"{name}_report.json"), "w"), indent=1)
    return rep

def make_report(name, res, prob, mode, out_dir="reports"):
    if res.get("early_exit"): return make_quick_report(name, res, mode, out_dir)
    os.makedirs(out_dir, exist_ok=True); f, t, T = res["features"], res["target"], res["trigger"]
    ev = evidence(f, t, T) + transformer_evidence(res); rep = dict(model=name, verdict=verdict(prob), probability_backdoored=prob, scoring_mode=mode,
        suspected_target_class=t, clean_accuracy=res["clean_acc"], classes_searched=T["classes"], mixed_precision=res.get("amp", False), transformer=res.get("transformer"), features=f, evidence=ev,
        mask_l1_per_class=T["l1"], anomaly_index_per_class=T["anomaly"])
    mk, pt = T["masks"][t].numpy(), T["pats"][t].numpy()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].imshow(mk[0], cmap="gray", vmin=0, vmax=1); ax[0].set_title("Reconstructed mask")
    ax[1].imshow(pt.transpose(1, 2, 0).clip(0, 1)); ax[1].set_title("Reconstructed pattern")
    ax[2].imshow((mk * pt).transpose(1, 2, 0).clip(0, 1)); ax[2].set_title("Trigger (mask x pattern)")
    cols = ["crimson" if c == t else "steelblue" for c in T["classes"]]
    ax[3].bar(range(len(T["l1"])), T["l1"], color=cols); ax[3].axhline(np.median(T["l1"]), ls="--", c="gray"); ax[3].set_title("Mask L1 per class")
    for a in ax[:3]: a.axis("off")
    png = os.path.join(out_dir, f"{name}_trigger.png"); fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
    md = [f"# Model Security Report: {name}", f"**Verdict:** {rep['verdict']}  ", f"**P(backdoor):** {prob:.2f} ({mode})  ",
          f"**Suspected target class:** {t}  ", f"**Clean accuracy on scan set:** {'n/a (reference labels do not match model classes)' if res['clean_acc'] is None else format(res['clean_acc'], '.1%')}", "", "## Evidence"]
    md += [f"- {'**FIRED**' if e['fired'] else 'quiet'} - {e['detector']}: {e['detail']}" for e in ev] + ["", f"![trigger]({os.path.basename(png)})"]
    open(os.path.join(out_dir, f"{name}_report.md"), "w").write("\n".join(md))
    json.dump(rep, open(os.path.join(out_dir, f"{name}_report.json"), "w"), indent=1)
    rep["trigger_png"] = png; return rep
