"""Presentation layer only: turns scanner output into view data and renders the dashboard.
Nothing here changes how a scan is run or scored."""
import base64, json, os
import streamlit as st

STAGES = ["Behavioural testing", "Activation analysis", "Trigger search", "Warp trigger search",
          "Cross-validating candidate trigger", "Evidence aggregation"]
CLASSES = {"cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]}
_TEMPLATE = os.path.join(os.path.dirname(__file__), "template.html")
HEIGHT = 1890


def _tone(p): return "bad" if p >= .8 else "warn" if p >= .4 else "ok"


def stage_index(name):
    return STAGES.index(name) if name in STAGES else None


def _base(kind, dataset, arch, nc=10, name="", stage=-1):
    return dict(kind=kind, name=name, dataset=dataset, arch=arch, nc=nc, stage=stage, stages=STAGES, n_scan=500,
                classes=CLASSES.get(dataset), tone="idle")


def idle_view(dataset, arch): return _base("idle", dataset, arch)


def scanning_view(dataset, arch, name, stage=-1, nc=10): return _base("scanning", dataset, arch, nc, name, stage)


def done_view(name, res, rep, prob, mode, dataset, arch, extras=None):
    T, t = res["trigger"], res["target"]; m, p = T["masks"][t], T["pats"][t]
    v = _base("done", dataset, arch, len(T["l1"]), name)
    v.update(tone=_tone(prob), prob=float(prob), verdict=rep["verdict"], target=int(t), clean_acc=float(res["clean_acc"]), mode=mode,
             l1=[float(x) for x in T["l1"]], anomaly=[float(x) for x in T["anomaly"]], mask_pct=float(m.sum() / m.numel() * 100),
             evidence=rep["evidence"],
             trigger=dict(w=int(m.shape[-1]), h=int(m.shape[-2]), mask=[round(float(x), 3) for x in m.flatten()],
                          pattern=[round(float(x), 3) for x in p.flatten()]))
    if extras: v["extras"] = extras
    return v


def sample_view(dataset="cifar10", arch="smallcnn", folder="reports_cifar", name="bd_8"):
    """Real saved scan shown before the first scan, clearly labelled as a sample."""
    cached = os.path.join(os.path.dirname(__file__), "sample_view.json")      # written by ui/make_sample.py
    if os.path.exists(cached):
        v = json.load(open(cached)); v["kind"] = "sample"; return v
    jp = os.path.join(folder, f"{name}_report.json")
    if not os.path.exists(jp): return None
    r = json.load(open(jp)); png = os.path.join(folder, f"{name}_trigger.png"); l1 = r["mask_l1_per_class"]
    v = _base("sample", dataset, arch, len(l1), name)
    v.update(tone=_tone(r["probability_backdoored"]), prob=r["probability_backdoored"], verdict=r["verdict"], target=r["suspected_target_class"],
             clean_acc=r["clean_accuracy"], mode=r["scoring_mode"], l1=l1, anomaly=r["anomaly_index_per_class"], evidence=r["evidence"],
             mask_pct=l1[r["suspected_target_class"]] / 1024 * 100)
    if os.path.exists(png): v["image"] = "data:image/png;base64," + base64.b64encode(open(png, "rb").read()).decode()
    return v


def html(view):
    data = json.dumps(view).replace("</", "<\\/")
    return open(_TEMPLATE, encoding="utf-8").read().replace("/*__DATA__*/null", data)


def render(slot, view):
    """Draw the dashboard into an st.empty() slot (works on older and newer Streamlit)."""
    page = html(view)
    with slot.container():
        if hasattr(st, "iframe"): st.iframe(page, height=HEIGHT)
        else:
            import streamlit.components.v1 as components
            components.html(page, height=HEIGHT, scrolling=True)
