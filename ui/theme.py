"""Streamlit chrome styling (sidebar, buttons, spacing). Pure CSS; no behaviour."""
import streamlit as st

CSS = """
<style>
:root{--bg:#06080b;--panel:#0b0f15;--line:rgba(255,255,255,.08);--accent:#34e0c2}
html,body,[data-testid="stAppViewContainer"],.stApp{background:var(--bg)!important;font-family:Inter,system-ui,"Segoe UI",sans-serif}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none!important}
.block-container{padding:1.1rem 1.6rem 2rem!important;max-width:1500px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1016,#080b0f)!important;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarContent"]{padding-top:1.2rem}
.side-h{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#7b8795;margin:2px 0 14px}
[data-testid="stSidebar"] label p{font-size:12px!important;color:#9aa5b1!important}
[data-baseweb="select"]>div,[data-baseweb="input"],.stTextInput input{background:#0e141b!important;border-color:var(--line)!important;border-radius:10px!important}
[data-testid="stFileUploaderDropzone"]{background:#0e141b;border:1px dashed rgba(52,224,194,.35);border-radius:12px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);box-shadow:0 0 0 3px rgba(52,224,194,.08)}
.stButton>button,.stDownloadButton>button{border-radius:11px;font-weight:600;border:1px solid var(--line);transition:all .2s}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#2dd4bf,#a3e635);color:#04110e;border:0;box-shadow:0 6px 22px rgba(52,224,194,.22)}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08);transform:translateY(-1px)}
.stDownloadButton>button{background:#0e141b;color:#e8edf2}.stDownloadButton>button:hover{border-color:var(--accent);color:var(--accent)}
.safe{font-size:11.5px;line-height:1.55;color:#7b8795;border:1px solid var(--line);border-radius:10px;padding:9px 11px;margin-top:10px;background:#0b1016}
iframe{border:0!important;border-radius:16px}
</style>
"""


def apply(): st.markdown(CSS, unsafe_allow_html=True)
