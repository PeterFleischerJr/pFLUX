
# step3_qc_prefill.py
import io, os, zipfile
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 3 — QC filter (pre‑MDS)", layout="wide")
st.title("Step 3 — QC filter (pre‑MDS)")

def _dedup(names: List[str]) -> List[str]:
    seen = {}; out = []
    for x in names:
        n = str(x)
        if n in seen:
            seen[n] += 1; out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0; out.append(n)
    return out

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "DateTime" in d.columns:
        d["DateTime"] = pd.to_datetime(d["DateTime"], errors="coerce"); return d
    for c in ["timestamp","TIMESTAMP","TIMESTAMP_START","TIMESTAMP_END"]:
        if c in d.columns:
            d["DateTime"] = pd.to_datetime(d[c], errors="coerce"); return d
    if "date" in d.columns and "time" in d.columns:
        d["DateTime"] = pd.to_datetime(d["date"].astype(str)+" "+d["time"].astype(str), errors="coerce"); return d
    d["DateTime"] = pd.to_datetime(d.iloc[:,0], errors="coerce"); return d

def regrid_5min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["DateTime"]).sort_values("DateTime").copy()
    if d.empty: return d
    t0, t1 = d["DateTime"].min(), d["DateTime"].max()
    grid = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq="5min")})
    return grid.merge(d.drop_duplicates(subset=["DateTime"], keep="last"), on="DateTime", how="left")

def coerce_num(s): return pd.to_numeric(s, errors="coerce")

def guess_qc_for_target(target: str, cols: List[str]) -> Optional[str]:
    t = str(target).lower()
    cands = []
    if ("nee" in t) or ("co2_flux" in t):
        cands = ["qc_co2_flux","FO_qc_co2_flux","qc_co2","co2_qc","qc_NEE","qc_nee"]
    elif (t.startswith("le")) or ("h2o_flux" in t):
        cands = ["qc_LE","FO_qc_LE","qc_h2o_flux","qc_le"]
    elif (t == "h") or (t.startswith("h_")):
        cands = ["qc_H","FO_qc_H","qc_h"]
    else:
        base = target.split("__")[0].split("_uStarFilt")[0]
        cands = [f"qc_{base}", f"FO_qc_{base}"]
    for c in cands:
        if c in cols: return c
    any_qc = [c for c in cols if str(c).lower().startswith("qc_")]
    return any_qc[0] if any_qc else None

st.header("1) Load 5‑min file (ideally COMPLETE_5min.csv from Step 3c)")
c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("Upload CSV", type=["csv"])
with c2:
    path = st.text_input("…or enter a DIRECT PATH to CSV", value="")

df = None
if up is not None:
    df = pd.read_csv(up, low_memory=False)
elif path.strip():
    try:
        df = pd.read_csv(path.strip(), low_memory=False); st.info(f"Loaded from path: {path.strip()}")
    except Exception as e:
        st.error(f"Failed to read path: {e}"); st.stop()
else:
    st.stop()

df.columns = _dedup(list(df.columns))
df = ensure_datetime(df); df = regrid_5min(df)
st.success(f"Loaded {len(df):,} rows on a continuous 5‑min grid.")
st.dataframe(df.head(12), use_container_width=True)

st.header("2) Choose target columns and map QC columns")
def _ends(cols, suffix): return [c for c in cols if isinstance(c, str) and c.endswith(suffix)]
ustar_targets = _ends(df.columns, "_uStarFilt")
if ustar_targets:
    default_targets = [c for c in ustar_targets if ("NEE" in c or "co2_flux" in c.lower())] or ustar_targets
else:
    base = []
    for cand in ["NEE","LE","H","co2_flux","h2o_flux"]:
        if cand in df.columns: base.append(cand)
    base += [c for c in df.columns if isinstance(c, str) and ("_alloc_STRICT_class_" in c or "_RAW_class_" in c)]
    default_targets = base
targets = st.multiselect("Target columns to QC‑filter (set invalid rows to NaN)",
                         options=[c for c in df.columns if c != "DateTime"], default=default_targets)

qc_map = {}
if targets:
    st.subheader("QC mapping per target")
    qc_cols_present = [c for c in df.columns if ("qc" in str(c).lower())]
    max_qc_default = 1
    c1, c2 = st.columns([2,1])
    with c1: st.caption("Pick QC column per target (if applicable).")
    with c2: max_qc = st.number_input("Max allowed QC code", min_value=0, max_value=9, value=max_qc_default, step=1)
    for t in targets:
        guess = guess_qc_for_target(t, df.columns.tolist())
        qc_map[t] = st.selectbox(f"QC column for **{t}**", options=["<none>"] + qc_cols_present,
                                 index=(0 if (guess is None) else (qc_cols_present.index(guess)+1)))
else:
    max_qc = 1

ustar_flag_candidates = [c for c in df.columns if str(c).lower() in ["ustar_fail","ustar_flag","u*fail","ustar_fail_flag","ustar_fail"]]
ustar_flag = st.selectbox("u* fail flag column (optional)", options=["<none>"] + ustar_flag_candidates, index=(0 if not ustar_flag_candidates else 1))
apply_ustar = st.checkbox("Require u* pass (flag == 0)", value=True if ustar_flag != "<none>" else False)

st.header("3) Missing definition & optional plausible ranges")
c1, c2 = st.columns(2)
with c1:
    zero_as_nan = st.checkbox("Treat values with |x| ≤ ε as missing", value=True)
    zero_eps = st.number_input("ε for zero-as-missing (absolute)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
with c2:
    use_ranges = st.checkbox("Apply plausible ranges (per target)", value=False)

ranges = {}
if use_ranges and targets:
    st.subheader("Plausible ranges per target (optional)")
    for t in targets:
        vmin = st.number_input(f"{t}: min", value=-50.0 if ("nee" in t.lower() or "co2_flux" in t.lower()) else -1e6)
        vmax = st.number_input(f"{t}: max", value=50.0 if ("nee" in t.lower() or "co2_flux" in t.lower()) else 1e6)
        ranges[t] = (vmin, vmax)

out_mode = st.radio("Output mode", ["Overwrite selected targets", "Create new columns with suffix `_QC`"], index=0)
run = st.button("▶️ Apply QC and export")

def build_mask_for_target(D: pd.DataFrame, target: str) -> pd.Series:
    m = pd.Series(True, index=D.index)
    qcc = qc_map.get(target, "<none>")
    if qcc and qcc != "<none>" and qcc in D.columns:
        qv = coerce_num(D[qcc]); m &= (qv <= float(max_qc))
    if apply_ustar and ustar_flag and ustar_flag != "<none>" and (ustar_flag in D.columns):
        m &= (coerce_num(D[ustar_flag]) == 0)
    if use_ranges and (target in ranges):
        lo, hi = ranges[target]; tv = coerce_num(D[target]); m &= (tv >= float(lo)) & (tv <= float(hi))
    return m

if run and targets:
    D = df.copy(); prog = st.progress(0); status = st.empty(); report = []
    for i, t in enumerate(targets, start=1):
        status.write(f"Processing {t} ({i}/{len(targets)})…")
        m = build_mask_for_target(D, t)
        base = coerce_num(D[t])
        if zero_as_nan and zero_eps > 0.0:
            base = base.mask(np.isclose(base, 0.0, atol=zero_eps))
        outcol = t if out_mode.startswith("Overwrite") else f"{t}_QC"
        D[outcol] = base.where(m, np.nan)
        newly_na = int((~m & base.notna()).sum())
        report.append({"target": t, "rows_total": len(base), "rows_set_NA_by_QC": newly_na, "rows_valid_after_QC": int(m.sum())})
        prog.progress(int(100*i/max(1,len(targets))))
    prog.progress(100); status.write("Done.")
    st.subheader("Summary"); rep = pd.DataFrame(report); st.dataframe(rep, use_container_width=True)
    out_csv = D.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ STEP3_QC_Filtered_5min.csv", out_csv, "STEP3_QC_Filtered_5min.csv", "text/csv")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("STEP3_QC_Filtered_5min.csv", out_csv)
        z.writestr("STEP3_QC_Report.csv", rep.to_csv(index=False).encode("utf-8"))
    st.download_button("⬇️ STEP3_QC_Filtered_5min.zip", buf.getvalue(), "STEP3_QC_Filtered_5min.zip", "application/zip")
