
# step5_prep_daily_gc_FILTER_v2.py
# -----------------------------------------------------------------------------
# Step 5 — Prepare daily inputs (gC) with plausible-range filtering
# NEW in v2:
#   • Separates 'driver_ok' mask from per-column 'flux_ok' masks
#   • Creates per‑column filtered outputs: <col>_QC (invalid => NaN)
#   • Option to drop invalid rows in 5‑min export
#   • Keeps direct-path loading and 5‑min regridding
#   • Daily sums use the per‑column valid mask (driver_ok & flux_ok_col)
# -----------------------------------------------------------------------------

import io, zipfile, os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 5 — Prep daily gC (filter 5‑min)", layout="wide")
st.title("Step 5 — Prepare daily gC with real 5‑min filtering")

# ---------------- helpers ----------------
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
        d["DateTime"] = pd.to_datetime(d["date"].astype(str) + " " + d["time"].astype(str), errors="coerce"); return d
    d["DateTime"] = pd.to_datetime(d.iloc[:,0], errors="coerce"); return d

def regrid_5min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["DateTime"]).sort_values("DateTime").copy()
    if d.empty: return d
    t0, t1 = d["DateTime"].min(), d["DateTime"].max()
    grid = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq="5min")})
    return grid.merge(d.drop_duplicates(subset=["DateTime"], keep="last"), on="DateTime", how="left")

def coerce_num(s): return pd.to_numeric(s, errors="coerce")

def in_range(x, lo, hi):
    v = coerce_num(x); return (v >= lo) & (v <= hi)

# ---------------- UI: load ----------------
st.header("1) Load 5‑min table")
c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("Upload CSV (e.g., COMPLETE_5min.csv or STEP4_MDS_Gapfilled_COMPLETE.csv)", type=["csv"])
with c2:
    path = st.text_input("…or direct path to CSV", value="")

df = None
if up is not None:
    df = pd.read_csv(up, low_memory=False)
elif path.strip():
    try:
        df = pd.read_csv(path.strip(), low_memory=False)
        st.info(f"Loaded from path: {path.strip()}")
    except Exception as e:
        st.error(f"Failed to read path: {e}")
        st.stop()
else:
    st.stop()

df.columns = _dedup(list(df.columns))
df = ensure_datetime(df)
df = regrid_5min(df)
st.success(f"Loaded {len(df):,} rows on a continuous 5‑min grid.")
st.dataframe(df.head(12), use_container_width=True)

# ---------------- UI: map drivers and fluxes ----------------
st.header("2) Map drivers and choose flux columns")

def pick(df, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    return None

rg_col = st.selectbox("Rg (W m⁻²) column", options=[c for c in ["Rg_Wm2","Rg","SW_IN","FO_SW_IN","SW_IN_1_1_1"] if c in df.columns] + [c for c in df.columns if c not in ["DateTime"]])
ta_col = st.selectbox("Tair (°C) column", options=[c for c in ["Ta_C","TA","Tair","air_temperature","FO_air_temperature","TA_1_1_1"] if c in df.columns] + [c for c in df.columns if c not in ["DateTime"]])
vpd_col_opt = st.selectbox("VPD (kPa) column (optional)", options=["<none>"] + [c for c in df.columns if c != "DateTime"], index=0)
ustar_col_opt = st.selectbox("u* column (optional)", options=["<none>"] + [c for c in df.columns if c != "DateTime"], index=0)
ws_col_opt = st.selectbox("Wind speed column (optional)", options=["<none>"] + [c for c in df.columns if c != "DateTime"], index=0)

# default flux columns
default_flux = [c for c in df.columns if isinstance(c, str) and c.endswith("_GF")]
if not default_flux:
    default_flux = [c for c in df.columns if isinstance(c, str) and c.endswith("_uStarFilt")]
nee_like = [c for c in default_flux if ("NEE" in c or "co2_flux" in c.lower())]
if nee_like: default_flux = nee_like
flux_cols = st.multiselect("Flux columns to filter & aggregate (µmol m⁻² s⁻¹)", options=[c for c in df.columns if c != "DateTime"], default=default_flux)

# ---------------- UI: plausible ranges + behaviour ----------------
st.header("3) Plausible ranges & behaviour")

c1,c2,c3 = st.columns(3)
with c1:
    rg_min = st.number_input("Rg min (W m⁻²)", value=0.0, step=10.0); rg_max = st.number_input("Rg max (W m⁻²)", value=1500.0, step=10.0)
    ta_min = st.number_input("Ta min (°C)", value=-40.0, step=1.0); ta_max = st.number_input("Ta max (°C)", value=50.0, step=1.0)
with c2:
    vpd_min = st.number_input("VPD min (kPa)", value=0.0, step=0.1); vpd_max = st.number_input("VPD max (kPa)", value=6.0, step=0.1)
    ustar_min = st.number_input("u* min (m s⁻¹)", value=0.0, step=0.05); ustar_max = st.number_input("u* max (m s⁻¹)", value=2.0, step=0.05)
with c3:
    ws_min = st.number_input("WS min (m s⁻¹)", value=0.0, step=0.1); ws_max = st.number_input("WS max (m s⁻¹)", value=30.0, step=0.5)
    nee_min = st.number_input("NEE-like flux min (µmol m⁻² s⁻¹)", value=-50.0, step=1.0)
    nee_max = st.number_input("NEE-like flux max (µmol m⁻² s⁻¹)", value=50.0, step=1.0)

c4,c5 = st.columns(2)
with c4:
    treat_zero = st.checkbox("Treat exact zero in flux columns as missing", value=False)
with c5:
    filt_mode = st.radio("5‑min export behaviour for invalid rows",
                         ["Keep all rows, set invalid to NaN (recommended)",
                          "Drop rows that fail driver ranges"], index=0)

apply_btn = st.button("▶️ Apply filtering and export (5‑min + daily gC)")

# ---------------- Core ----------------
if apply_btn:
    D = df.copy()

    # 1) Driver mask
    driver_ok = pd.Series(True, index=D.index)
    if rg_col:  driver_ok &= in_range(D[rg_col], rg_min, rg_max)
    if ta_col:  driver_ok &= in_range(D[ta_col], ta_min, ta_max)
    if vpd_col_opt != "<none>": driver_ok &= in_range(D[vpd_col_opt], vpd_min, vpd_max)
    if ustar_col_opt != "<none>": driver_ok &= in_range(D[ustar_col_opt], ustar_min, ustar_max)
    if ws_col_opt != "<none>": driver_ok &= in_range(D[ws_col_opt], ws_min, ws_max)
    D["driver_ok"] = driver_ok.astype(int)

    # 2) Per‑column flux masks and filtered columns
    percol_valid = {}
    for c in flux_cols:
        col_ok = driver_ok.copy()
        # apply plausible range for NEE/co2_flux-like columns
        if ("NEE" in c) or ("co2_flux" in c.lower()):
            col_ok &= in_range(D[c], nee_min, nee_max)
        # treat zeros as missing if requested
        if treat_zero:
            col_ok &= ~np.isclose(coerce_num(D[c]).fillna(np.inf), 0.0)
        percol_valid[c] = col_ok
        D[f"{c}_QC"] = coerce_num(D[c]).where(col_ok, np.nan)

    # 3) Optionally drop invalid rows in 5‑min export (based on drivers only)
    if filt_mode.startswith("Drop"):
        D5 = D.loc[driver_ok].copy()
    else:
        D5 = D.copy()

    # 4) Daily sums (gC) using per‑column masks
    factor = 300.0 * 12e-6  # 5‑min * molar mass conversion
    D["Date"] = pd.to_datetime(D["DateTime"]).dt.date

    daily = None
    daily_rows = []
    for c in flux_cols:
        step_gc = coerce_num(D[c]) * factor
        step_gc = step_gc.where(percol_valid[c], np.nan)
        grp = step_gc.groupby(D["Date"])
        s = grp.sum(min_count=1)
        n = grp.count()
        daily_rows.append(pd.DataFrame({"Date": s.index, f"{c}_daily_gC": s.values, f"{c}_n_steps": n.values}))
    for t in daily_rows:
        daily = t if daily is None else daily.merge(t, on="Date", how="outer")

    # 5) Previews
    st.subheader("5‑min (filtered columns have *_QC) — preview")
    st.dataframe(D5.head(20), use_container_width=True)
    st.subheader("Daily gC — preview")
    st.dataframe(daily.head(20), use_container_width=True)

    # 6) Downloads
    st.subheader("Downloads")
    st.download_button("⬇️ STEP5_5min_filtered.csv",
                       D5.to_csv(index=False).encode("utf-8"),
                       "STEP5_5min_filtered.csv", "text/csv")
    st.download_button("⬇️ STEP5_daily_gC.csv",
                       daily.to_csv(index=False).encode("utf-8"),
                       "STEP5_daily_gC.csv", "text/csv")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("STEP5_5min_filtered.csv", D5.to_csv(index=False))
        z.writestr("STEP5_daily_gC.csv", daily.to_csv(index=False))
    st.download_button("⬇️ Download BOTH (ZIP)", data=zbuf.getvalue(),
                       file_name="STEP5_daily_bundle.zip", mime="application/zip")
