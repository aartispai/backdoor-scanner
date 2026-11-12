"""streamlit run app.py   (Upload Model -> Scan -> Report)"""
import os, tempfile, json, joblib, streamlit as st
from bds.data import load_dataset
from bds.models import load_model
from bds.pipeline import scan_model
from bds.aggregate import heuristic
from bds.report import make_report

st.set_page_config(page_title="Model Security Scanner", layout="wide")
st.title("Backdoor Scanner for Pretrained Models")
st.caption("Upload Model > Behavioural Testing > Activation Analysis > Trigger Search > Evidence Aggregation > Report")
c1, c2, c3 = st.columns(3)
dataset = c1.selectbox("Clean reference data", ["cifar10", "gtsrb", "synthetic"]); arch = c2.selectbox("Architecture", ["smallcnn", "resnet18"])
steps = c3.slider("Trigger-search steps / class", 25, 400, 50, 25)
agg_path = st.text_input("Calibrated aggregator", "zoo/aggregator.joblib")
up = st.file_uploader("Model weights (.pt state_dict)", type=["pt", "pth"])
st.info("Loaded with weights_only=True, so a booby-trapped pickle cannot execute code during the scan.")
if up and st.button("Run security scan", type="primary"):
    path = os.path.join(tempfile.mkdtemp(), up.name); open(path, "wb").write(up.getbuffer())
    with st.spinner("Loading reference data (first run downloads it, then it is cached)..."): (_, _, Xte, yte), nc = load_dataset(dataset)
    bar, txt = st.progress(0.0), st.empty()
    def prog(stage, f): bar.progress(min(f, 1.0)); txt.write(f"**{stage}**")
    res = scan_model(load_model(path, arch, nc), Xte[:500], yte[:500], steps, prog); bar.progress(1.0); txt.empty()
    if os.path.exists(agg_path): prob, mode = joblib.load(agg_path).prob(res["features"]), "calibrated ensemble"
    else: prob, mode = heuristic(res["features"]), "uncalibrated heuristic"
    rep = make_report(up.name.rsplit(".", 1)[0], res, prob, mode, tempfile.mkdtemp())
    (st.error if prob >= .8 else st.warning if prob >= .4 else st.success)(f"{rep['verdict']}  |  P(backdoor) = {prob:.2f}  |  suspected target class: {rep['suspected_target_class']}")
    st.caption(f"Scoring: {mode}. Clean accuracy on scan set: {'n/a' if res['clean_acc'] is None else format(res['clean_acc'], '.1%')}")
    st.subheader("Evidence")
    for e in rep["evidence"]: st.markdown(("FIRED " if e["fired"] else "quiet ") + f"**{e['detector']}** - {e['detail']}")
    st.subheader("Reconstructed trigger"); st.image(rep["trigger_png"])
    st.download_button("Download JSON report", json.dumps({k: v for k, v in rep.items() if k != "trigger_png"}, indent=1), "report.json")
