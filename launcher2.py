
# launcher.py
import os, runpy
import streamlit as st

st.set_page_config(page_title="EC Suite — Launcher", layout="wide")
st.title("EC Suite — Launcher")

BASE = os.path.dirname(os.path.abspath(__file__))

apps = {
    "1) Step 1 — V3 merge / ALL‑VARS": "step1.py",
    "2) Step 2 — Footprint allocation": "step2d.py",
    "3) Step 3 — u* threshold (COMPLETE export)": "step3d.py",
    "3b) Step 3 — QC filter (pre‑MDS)": "step3_qc_prefill.py",
    "4) Step 4 — MDS gap filling": "step4c.py",
    "5) Step 5 — Daily gC (plausible ranges)": "step5b.py",
}

choice = st.sidebar.radio("Choose window", list(apps.keys()), index=0)

st.sidebar.caption("This launcher expects the mini‑apps in the same folder:")
for label, fn in apps.items():
    st.sidebar.write(f"• {fn}")

st.markdown(f"**Running:** `{apps[choice]}`")

target = os.path.join(BASE, apps[choice])
if not os.path.exists(target):
    st.error(f"File not found: {target}")
else:
    runpy.run_path(target, run_name="__main__")
