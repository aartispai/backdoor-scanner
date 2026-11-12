"""streamlit run app.py   (Upload Model -> Scan -> Report)"""
import os, tempfile, json, joblib, streamlit as st
from bds.data import load_dataset
from bds.models import load_model
from bds.pipeline import scan_model
from bds.aggregate import heuristic
from bds.report import make_report
from ui import dashboard as ui, theme, extras

st.set_page_config(page_title="Backdoor Scanner", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
theme.apply()

with st.sidebar:
    st.markdown('<div class="side-h">Scan configuration</div>', unsafe_allow_html=True)
    dataset = st.selectbox("Clean reference data", ["cifar10", "gtsrb", "synthetic"]); arch = st.selectbox("Architecture", ["smallcnn", "resnet18"])
    steps = st.slider("Trigger-search steps / class", 25, 400, 50, 25)
    agg_path = st.text_input("Calibrated aggregator", "zoo/aggregator.joblib")
    up = st.file_uploader("Model weights (.pt state_dict)", type=["pt", "pth"])
    run = st.button("Run security scan", type="primary", use_container_width=True)
    st.markdown('<div class="safe">Loaded with weights_only=True, so a booby-trapped pickle cannot execute code during the scan.</div>', unsafe_allow_html=True)
    if run and not up: st.warning("Choose a model file first.")

slot = st.empty()
if run and up:
    name = up.name.rsplit(".", 1)[0]; cur = {"stage": -1}
    ui.render(slot, ui.scanning_view(dataset, arch, name))
    path = os.path.join(tempfile.mkdtemp(), up.name); open(path, "wb").write(up.getbuffer())
    (_, _, Xte, yte), nc = load_dataset(dataset)
    def prog(stage, f):
        i = ui.stage_index(stage)
        if i is not None and i != cur["stage"]: cur["stage"] = i; ui.render(slot, ui.scanning_view(dataset, arch, name, i, nc))
    model = load_model(path, arch, nc); res = scan_model(model, Xte[:500], yte[:500], steps, prog)
    if os.path.exists(agg_path): prob, mode = joblib.load(agg_path).prob(res["features"]), "calibrated ensemble"
    else: prob, mode = heuristic(res["features"]), "uncalibrated heuristic"
    rep = make_report(name, res, prob, mode, tempfile.mkdtemp())
    try: ex = extras.compute(model, Xte[:500], res)       # optional data for the 3D views; never blocks the report
    except Exception: ex = None
    st.session_state["view"] = ui.done_view(name, res, rep, prob, mode, dataset, arch, ex)
    st.session_state["json"] = json.dumps({k: v for k, v in rep.items() if k != "trigger_png"}, indent=1)
view = st.session_state.get("view")
ui.render(slot, view or ui.sample_view(dataset, arch) or ui.idle_view(dataset, arch))
if st.session_state.get("json"): st.download_button("Download JSON report", st.session_state["json"], "report.json")
