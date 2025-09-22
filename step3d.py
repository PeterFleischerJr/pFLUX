
# step3c_complete_export.py
# -----------------------------------------------------------------------------
# u* threshold detection (Python-only, REddyProc-like) + COMPLETE export:
# - loads merged 5-min file (biomet + allocations)
# - detects u* thresholds (3-mo windows × Tair bins, bootstrap)
# - applies selected threshold to overall flux + per-class columns
# - exports a SINGLE "COMPLETE_5min.csv" containing:
#     * all original columns
#     * uStar_fail flag + uStar_threshold_applied + threshold_choice
#     * *_uStarFilt columns for the overall flux + each per-class allocation
#     * standard drivers: Rg_Wm2, Ta_C, VPD_kPa (computed from RH if needed)
#     * reindexed to continuous 5-min grid (no gaps in time index; NA where missing)
# - also offers the separate thresholds CSVs and a ZIP bundle.
# -----------------------------------------------------------------------------

import io, os, zipfile
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 3c — u* + COMPLETE 5‑min export", layout="wide")
st.title("Step 3c — u* threshold + COMPLETE 5‑min export")

# ---------------- helpers ----------------
def _dedup(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for x in names:
        n = str(x)
        if n not in seen:
            seen[n] = 0; out.append(n)
        else:
            seen[n] += 1; out.append(f"{n}__{seen[n]}")
    return out

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "DateTime" in d.columns:
        d["DateTime"] = pd.to_datetime(d["DateTime"], errors="coerce")
        return d
    for c in ["timestamp","TIMESTAMP","TIMESTAMP_START","TIMESTAMP_END"]:
        if c in d.columns:
            d["DateTime"] = pd.to_datetime(d[c], errors="coerce")
            return d
    if "date" in d.columns and "time" in d.columns:
        d["DateTime"] = pd.to_datetime(d["date"].astype(str) + " " + d["time"].astype(str), errors="coerce")
        return d
    d["DateTime"] = pd.to_datetime(d.iloc[:,0], errors="coerce")
    return d

def regrid_5min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.dropna(subset=["DateTime"]).sort_values("DateTime")
    if d.empty:
        return d
    t0, t1 = d["DateTime"].min(), d["DateTime"].max()
    grid = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq="5min")})
    # merge_asof requires sorted on; we use left join by nearest within tolerance 0min to only align exact matches
    # Instead, do outer merge on DateTime exact, then fill NAs (so we keep truly blank 5-min rows)
    out = grid.merge(d.drop_duplicates(subset=["DateTime"], keep="last"), on="DateTime", how="left")
    return out

def coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def month_window_mask(dt: pd.Series, center_month: int) -> pd.Series:
    m = dt.dt.month
    left = (center_month - 2) % 12 + 1
    right = center_month % 12 + 1
    return (m == left) | (m == center_month) | (m == right)

def detect_ustar_threshold_onebin(df_bin: pd.DataFrame, flux_col: str, ustar_col: str,
                                  eps_rel: float = 0.05, min_count: int = 30) -> Optional[float]:
    d = df_bin[[flux_col, ustar_col]].dropna()
    if len(d) < min_count:
        return None
    d = d.sort_values(ustar_col)
    qs = np.linspace(0.10, 0.90, 9)
    cand = np.unique(np.quantile(d[ustar_col].values, qs))
    if len(cand) < 3:
        return None
    med_above = [np.nanmedian(d.loc[d[ustar_col] >= c, flux_col].values) for c in cand]
    top = med_above[-1]
    ref = max(abs(top), 1e-9)
    for c, med in zip(cand, med_above):
        if np.isfinite(med) and abs(top - med) <= eps_rel * ref:
            return float(c)
    return float(np.quantile(d[ustar_col].values, 0.95))

def bootstrap_bin_threshold(df_bin: pd.DataFrame, flux_col: str, ustar_col: str,
                            eps_rel: float, min_count: int, n_boot: int, rng: np.random.Generator) -> List[float]:
    idx = np.arange(len(df_bin))
    ths = []
    if len(idx) < min_count:
        return ths
    for _ in range(n_boot):
        ridx = rng.choice(idx, size=len(idx), replace=True)
        sub = df_bin.iloc[ridx]
        th = detect_ustar_threshold_onebin(sub, flux_col, ustar_col, eps_rel, min_count)
        if th is not None and np.isfinite(th):
            ths.append(float(th))
    return ths

def apply_ustar_filter(df: pd.DataFrame, flux_cols: List[str], ustar_col: str,
                       threshold: float, night_mask: pd.Series, threshold_choice: str) -> pd.DataFrame:
    out = df.copy()
    bad = night_mask & (coerce_num(out[ustar_col]) < float(threshold))
    out["uStar_threshold_applied"] = float(threshold)
    out["uStar_choice"] = threshold_choice
    out["uStar_fail"] = bad.astype(int)
    for c in flux_cols:
        if c in out.columns:
            out[f"{c}_uStarFilt"] = coerce_num(out[c]).where(~bad, np.nan)
    return out

def compute_vpd_kpa(ta_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    ta = coerce_num(ta_c)
    rh = coerce_num(rh_pct)
    es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))  # kPa
    return es * (1.0 - rh/100.0)

# ---------------- UI ----------------
st.header("1) Load CSV")
left, right = st.columns(2)
with left:
    up = st.file_uploader("CSV (merged biomet + allocations, 5‑min)", type=["csv"])
with right:
    path = st.text_input("…or path to CSV", value="")

df = None
if up is not None:
    df = pd.read_csv(up, low_memory=False)
elif path.strip():
    try:
        df = pd.read_csv(path.strip(), low_memory=False)
    except Exception as e:
        st.error(f"Failed to read path: {e}")
        df = None
else:
    df = None
if df is None:
    st.stop()

df.columns = _dedup(list(df.columns))
df = ensure_datetime(df).sort_values("DateTime")
st.write(f"Loaded rows: {len(df):,}")
st.dataframe(df.head(12), use_container_width=True)

# auto-map
def choose(df, cands: List[str]):
    for c in cands:
        if c in df.columns: return c
    return None

flux_col  = choose(df, ["NEE","FO_co2_flux","co2_flux"]) or st.selectbox("Overall flux column", options=df.columns.tolist())
ustar_col = choose(df, ["FO_u*","FO_ustar","u*","ustar"]) or st.selectbox("u* column", options=df.columns.tolist())
rg_col    = choose(df, ["Rg","FO_Rg","SW_IN","FO_SW_IN","RG","SW_IN_1_1_1"]) or st.selectbox("Rg/SW_IN column", options=df.columns.tolist())
tair_col  = choose(df, ["Tair","TA","air_temperature","FO_air_temperature","TA_1_1_1"])

# class columns
class_raw_cols    = [c for c in df.columns if isinstance(c, str) and "_RAW_class_" in c]
class_strict_cols = [c for c in df.columns if isinstance(c, str) and "_STRICT_class_" in c]

st.header("2) Nighttime & Tair bins")
c1, c2, c3 = st.columns(3)
with c1:
    rg_night_thr = st.number_input("Night Rg threshold (W m⁻²)", value=10.0, step=1.0, format="%.1f")
with c2:
    n_tair_bins = st.number_input("Tair bins (per season)", min_value=1, max_value=5, value=3, step=1)
with c3:
    eps_rel = st.number_input("Plateau tolerance ε", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

df["_ustar"] = coerce_num(df[ustar_col])
df["_flux"]  = coerce_num(df[flux_col])
if tair_col and tair_col in df.columns:
    ta = coerce_num(df[tair_col])
    if ta.quantile(0.5) > 200:  # Kelvin
        ta = ta - 273.15
    df["_Ta"] = ta
else:
    df["_Ta"] = np.nan
rg = coerce_num(df[rg_col]) if rg_col in df.columns else pd.Series(np.nan, index=df.index)
night = rg < float(rg_night_thr)
df["_is_night"] = night.astype(int)

st.header("3) Bootstrap settings")
b1, b2, b3 = st.columns(3)
with b1:
    n_boot = st.number_input("Bootstraps per bin", min_value=10, max_value=500, value=80, step=10)
with b2:
    min_count = st.number_input("Min points per bin", min_value=20, max_value=500, value=60, step=5)
with b3:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=1234, step=1)

run = st.button("▶️ Run u* detection & export COMPLETE")

if run:
    rng = np.random.default_rng(int(seed))
    dfN = df.loc[night & df["_ustar"].notna() & df["_flux"].notna()].copy()
    if dfN.empty:
        st.error("No nighttime points with both u* and flux.")
        st.stop()

    results = []
    distr_rows = []
    steps = 12 * max(1, int(n_tair_bins))
    pbar = st.progress(0); step_i = 0

    for m in range(1, 13):
        mask_season = month_window_mask(dfN["DateTime"], m)
        Dm = dfN.loc[mask_season].copy()
        if Dm.empty:
            step_i += int(n_tair_bins); pbar.progress(int(100*step_i/steps)); continue

        if int(n_tair_bins) == 1 or Dm["_Ta"].notna().sum() == 0:
            Dm["_tbin"] = 0
            edges = []
        else:
            q = [i/int(n_tair_bins) for i in range(1, int(n_tair_bins))]
            edges = np.unique(np.nanpercentile(Dm["_Ta"].dropna().values, [qq*100 for qq in q]))
            Dm["_tbin"] = np.digitize(Dm["_Ta"], bins=edges, right=False)

        for tb in range(int(n_tair_bins)):
            Db = Dm.loc[Dm["_tbin"] == tb].copy()
            if len(Db) < int(min_count):
                results.append({"season_center_month": m, "tair_bin": tb, "n": len(Db),
                                "uStar_p05": np.nan, "uStar_p50": np.nan, "uStar_p95": np.nan})
                step_i += 1; pbar.progress(int(100*step_i/steps)); continue
            ths = bootstrap_bin_threshold(Db, "_flux", "_ustar",
                                          eps_rel=float(eps_rel), min_count=int(min_count),
                                          n_boot=int(n_boot), rng=rng)
            if len(ths) == 0:
                u05=u50=u95=np.nan
            else:
                u05, u50, u95 = np.nanpercentile(ths, [5,50,95]).tolist()
            results.append({"season_center_month": m, "tair_bin": tb, "n": len(Db),
                            "uStar_p05": u05, "uStar_p50": u50, "uStar_p95": u95})
            for t in ths:
                distr_rows.append({"season_center_month": m, "tair_bin": tb, "uStar_thr": t})
            step_i += 1; pbar.progress(int(100*step_i/steps))

    res = pd.DataFrame(results)
    distr = pd.DataFrame(distr_rows)
    st.subheader("u* thresholds per season × Tair bin")
    st.dataframe(res, use_container_width=True)
    glob05=glob50=glob95=np.nan
    if not distr.empty:
        glob05, glob50, glob95 = np.nanpercentile(distr["uStar_thr"], [5,50,95]).tolist()
        fig, ax = plt.subplots(); ax.hist(distr["uStar_thr"].dropna().values, bins=30)
        ax.set_xlabel("u* threshold"); ax.set_ylabel("count"); ax.set_title("Bootstrap distribution")
        st.pyplot(fig)
    st.write(f"Global: p05={glob05:.3f}, p50={glob50:.3f}, p95={glob95:.3f}" if np.isfinite(glob05) else "Global thresholds NA")

    c1, c2 = st.columns(2)
    with c1:
        choice = st.selectbox("Apply which threshold?", ["p05 (conservative)","p50 (median)","p95 (liberal)","custom"], index=0)
    with c2:
        custom_thr = st.number_input("Custom u*", value=float(glob05 if np.isfinite(glob05) else 0.3))

    if choice.startswith("p05"):
        thr = float(glob05 if np.isfinite(glob05) else custom_thr)
    elif choice.startswith("p50"):
        thr = float(glob50 if np.isfinite(glob50) else custom_thr)
    elif choice.startswith("p95"):
        thr = float(glob95 if np.isfinite(glob95) else custom_thr)
    else:
        thr = float(custom_thr)

    # Apply to overall flux and per-class
    to_filter = [flux_col] + class_raw_cols + class_strict_cols
    filt = apply_ustar_filter(df, to_filter, ustar_col, thr, night, threshold_choice=choice.split()[0])

    # Standard drivers for post‑processing
    filt["Rg_Wm2"] = coerce_num(filt[rg_col]) if rg_col in filt.columns else np.nan
    filt["Ta_C"]   = filt["_Ta"]
    # VPD: use any existing vpd column; else compute from RH if present
    vpd_col = next((c for c in filt.columns if str(c).lower().startswith("vpd")), None)
    if vpd_col is not None:
        filt["VPD_kPa"] = coerce_num(filt[vpd_col])
    else:
        rh_col = next((c for c in filt.columns if str(c).lower() in ["rh","fo_rh","rh_1_1_1","rh_pct","relative_humidity"]), None)
        if rh_col is not None and filt["Ta_C"].notna().any():
            filt["VPD_kPa"] = compute_vpd_kpa(filt["Ta_C"], filt[rh_col])
        else:
            filt["VPD_kPa"] = np.nan

    # Reindex to continuous 5-min grid and keep ALL columns
    complete = regrid_5min(filt)

    st.subheader("COMPLETE 5‑min output — preview")
    st.dataframe(complete.head(20), use_container_width=True)

    # Downloads
    st.download_button("⬇️ COMPLETE_5min.csv", complete.to_csv(index=False).encode("utf-8"),
                       "COMPLETE_5min.csv", "text/csv")
    st.download_button("⬇️ thresholds_per_bin.csv", res.to_csv(index=False).encode("utf-8"),
                       "thresholds_per_bin.csv", "text/csv")
    st.download_button("⬇️ thresholds_distribution.csv", distr.to_csv(index=False).encode("utf-8"),
                       "thresholds_distribution.csv", "text/csv")

    # ZIP bundle
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("COMPLETE_5min.csv", complete.to_csv(index=False))
        z.writestr("thresholds_per_bin.csv", res.to_csv(index=False))
        z.writestr("thresholds_distribution.csv", distr.to_csv(index=False))
    st.download_button("⬇️ Download ALL (ZIP)", data=zbuf.getvalue(), file_name="step3c_complete_bundle.zip", mime="application/zip")

