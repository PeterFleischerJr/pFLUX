
# step4_mds_gapfill_PATH.py
# -----------------------------------------------------------------------------
# Step 4 — Python-only MDS gap filling with:
#  * File uploader OR direct disk path
#  * Fills NA and (optionally) zeros
#  * Predictors: Rg, Ta, VPD (compute VPD from RH+Ta if needed)
#  * Progress bars per column
#  * Exports: COMPLETE + ONLY + Report + ZIP
#  * FIXED: uses .dt.month for climatology fallback
# -----------------------------------------------------------------------------

from __future__ import annotations
import io, os, zipfile
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 4 — MDS Gap Filling (PATH enabled)", layout="wide")
st.title("Step 4 — MDS Gap Filling (Python-only, PATH enabled)")


# ------------------------ helpers ------------------------
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

def coerce_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def compute_vpd_kpa(ta_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    ta = coerce_num(ta_c); rh = coerce_num(rh_pct)
    es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))  # kPa
    return es * (1.0 - rh/100.0)

def regrid_5min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["DateTime"]).sort_values("DateTime").copy()
    if d.empty: return d
    t0, t1 = d["DateTime"].min(), d["DateTime"].max()
    grid = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq="5min")})
    return grid.merge(d.drop_duplicates(subset=["DateTime"], keep="last"), on="DateTime", how="left")

def make_bins(x: pd.Series, n: int, night_mask: Optional[pd.Series] = None) -> pd.Series:
    xv = coerce_num(x)
    if night_mask is not None:
        # Rg day/night bins: 0 for night; daylight binned 1..n
        bins = pd.Series(np.zeros(len(xv), dtype=int), index=x.index)
        day = ~night_mask
        try:
            q = pd.qcut(xv[day], q=n, labels=False, duplicates="drop")
            bins.loc[day] = np.asarray(q, dtype=float) + 1
            bins = bins.fillna(0).astype(int)
        except Exception:
            bins.loc[day] = 1
        return bins
    else:
        try:
            q = pd.qcut(xv, q=n, labels=False, duplicates="drop")
            return pd.Series(np.asarray(q, dtype=float), index=x.index).fillna(0).astype(int)
        except Exception:
            return pd.Series(np.zeros(len(xv), dtype=int), index=x.index)

def summarize_gaps(s: pd.Series, zero_is_nan: bool, zero_eps: float = 0.0) -> Dict[str,float]:
    v = coerce_num(s).copy()
    if zero_is_nan:
        v = v.mask(np.isclose(v, 0.0, atol=zero_eps))
    n = len(v); miss = v.isna().sum()
    return {"n": n, "n_missing": int(miss), "missing_pct": 100.0*miss/max(n,1)}


# ------------------------ UI: inputs ------------------------
st.header("1) Load file")
c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("Upload CSV (recommended: COMPLETE_5min.csv)", type=["csv"])
with c2:
    path = st.text_input("…or enter a DIRECT PATH to a CSV on disk", value="")

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
st.success(f"Loaded: {len(df):,} rows (continuous 5‑min grid).")
st.dataframe(df.head(12), use_container_width=True)


# ------------------------ UI: map predictors & targets ------------------------
st.header("2) Map predictors (for similarity search)")
def _choices(pref: List[str]) -> List[str]:
    # returns preferred columns if present first, then all columns
    found = [c for c in pref if c in df.columns]
    allc = [c for c in df.columns if c not in found and c != "DateTime"]
    return found + allc

rg_col  = st.selectbox("Rg (W m⁻²)", options=_choices(["Rg_Wm2","Rg","SW_IN","FO_SW_IN","SW_IN_1_1_1","SW_IN_1_1"]))
ta_col  = st.selectbox("Tair (°C)", options=_choices(["Ta_C","TA","Tair","air_temperature","FO_air_temperature","TA_1_1_1"]))
vpd_opt = st.selectbox("VPD (kPa) — choose a column or '<auto>' to compute from RH+Ta", options=["<auto>"] + _choices(["VPD_kPa","VPD","vpd"]))

# Build VPD series
if vpd_opt == "<auto>":
    if "VPD_kPa" in df.columns:
        df["_VPD"] = coerce_num(df["VPD_kPa"])
    else:
        rh_col = next((c for c in df.columns if str(c).lower() in ["rh","fo_rh","rh_1_1_1","rh_pct","relative_humidity"]), None)
        if rh_col is not None:
            df["_VPD"] = compute_vpd_kpa(coerce_num(df[ta_col]), coerce_num(df[rh_col]))
        else:
            st.warning("No VPD or RH found — VPD will be NA. Filling will rely on Rg/Ta bins.")
            df["_VPD"] = np.nan
else:
    df["_VPD"] = coerce_num(df[vpd_opt])

st.header("3) Choose target columns to fill")
default_targets = [c for c in df.columns if isinstance(c, str) and c.endswith("_uStarFilt")]
if "NEE_uStarFilt" in df.columns and "NEE_uStarFilt" not in default_targets:
    default_targets = ["NEE_uStarFilt"] + default_targets
targets = st.multiselect("Target columns", options=[c for c in df.columns if c != "DateTime"],
                         default=default_targets)

st.subheader("Gap definition")
zero_as_nan = st.checkbox("Treat zero as missing", value=True)
zero_eps = st.number_input("Zero tolerance (abs) — values with |x| ≤ eps are treated as zero", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
rg_night_thr = st.number_input("Night threshold for Rg (W m⁻²)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

st.subheader("MDS search parameters")
n_bins = st.number_input("Quantile bins for Rg(day)/Ta/VPD", min_value=2, max_value=10, value=6, step=1)
min_matches = st.number_input("Min matches to accept a fill", min_value=3, max_value=50, value=6, step=1)
windows = st.text_input("Time windows (days, comma-separated; widens stepwise)", value="7,14,30")
relax_bins = st.checkbox("Relax similarity if needed (±1 bin, drop VPD then Ta)", value=True)

run = st.button("▶️ Run MDS gap filling")


# ------------------------ Core MDS fill ------------------------
def mds_fill_column(ts: pd.Series,
                    hour: pd.Series,
                    rg_bin: pd.Series,
                    ta_bin: pd.Series,
                    vpd_bin: pd.Series,
                    dt: pd.Series,
                    windows_days: List[int],
                    min_matches: int = 6,
                    relax_bins: bool = True,
                    zero_as_nan: bool = True,
                    zero_eps: float = 0.0,
                    progress=None) -> Tuple[pd.Series, pd.Series]:
    """
    Return (filled_series, filled_flag) where filled_flag==1 for imputed values.
    """
    y = coerce_num(ts).copy()
    if zero_as_nan:
        y = y.mask(np.isclose(y, 0.0, atol=zero_eps))

    filled = y.copy()
    was_filled = pd.Series(np.zeros(len(y), dtype=int), index=y.index)

    arr_y = filled.to_numpy()
    arr_h = hour.to_numpy()
    arr_rb = rg_bin.to_numpy()
    arr_tb = ta_bin.to_numpy()
    arr_vb = vpd_bin.to_numpy()
    arr_t = pd.to_datetime(dt).to_numpy()

    idx_missing = np.where(~np.isfinite(arr_y))[0]
    total = len(idx_missing)
    if progress is not None:
        progress.progress(0)

    def mask_window(i, days):
        dt_i = arr_t[i]
        lo = dt_i - np.timedelta64(days, 'D')
        hi = dt_i + np.timedelta64(days, 'D')
        m = (arr_t >= lo) & (arr_t <= hi)
        m[i] = False
        return m

    for k, i in enumerate(idx_missing, start=1):
        if progress is not None:
            progress.progress(int(100*k/max(total,1)))

        for W in windows_days:
            # strict: same hour and all bins equal
            mask = mask_window(i, W) & (arr_h == arr_h[i]) & (arr_rb == arr_rb[i]) & (arr_tb == arr_tb[i]) & (arr_vb == arr_vb[i]) & np.isfinite(arr_y)
            vals = arr_y[mask]
            if len(vals) >= min_matches:
                arr_y[i] = np.nanmedian(vals); was_filled.iloc[i] = 1; break

            if relax_bins:
                # allow ±1 on VPD
                mask = mask_window(i, W) & (arr_h == arr_h[i]) & (arr_rb == arr_rb[i]) & (arr_tb == arr_tb[i]) & (np.abs(arr_vb - arr_vb[i]) <= 1) & np.isfinite(arr_y)
                vals = arr_y[mask]
                if len(vals) >= min_matches:
                    arr_y[i] = np.nanmedian(vals); was_filled.iloc[i] = 1; break

                # drop VPD, allow ±1 on Ta
                mask = mask_window(i, W) & (arr_h == arr_h[i]) & (arr_rb == arr_rb[i]) & (np.abs(arr_tb - arr_tb[i]) <= 1) & np.isfinite(arr_y)
                vals = arr_y[mask]
                if len(vals) >= min_matches:
                    arr_y[i] = np.nanmedian(vals); was_filled.iloc[i] = 1; break

                # keep only Rg_bin and hour
                mask = mask_window(i, W) & (arr_h == arr_h[i]) & (arr_rb == arr_rb[i]) & np.isfinite(arr_y)
                vals = arr_y[mask]
                if len(vals) >= min_matches:
                    arr_y[i] = np.nanmedian(vals); was_filled.iloc[i] = 1; break
            # else continue to next (wider) window

        # fallback: monthly × hour climatology
        # handled after loop for remaining NAs

    filled = pd.Series(arr_y, index=y.index)

    # Final vectorized fallback for any remaining NA
    if filled.isna().any():
        tmp = pd.DataFrame({
            "y": filled,
            "month": pd.to_datetime(dt).dt.month,   # <-- FIXED: .dt.month
            "hour": hour
        })
        clim = tmp.groupby(["month","hour"])["y"].transform("median")
        filled = filled.fillna(clim)

    # If still NA, use overall median
    filled = filled.fillna(np.nanmedian(filled.to_numpy()))
    return filled, was_filled


if run:
    D = df.copy()
    # time fields & bins
    D["hour"] = pd.to_datetime(D["DateTime"]).dt.hour
    rg = coerce_num(D[rg_col]); ta = coerce_num(D[ta_col]); vpd = coerce_num(D["_VPD"])
    night = rg < float(rg_night_thr)
    D["rg_bin"]  = make_bins(rg,  int(n_bins), night_mask=night)
    D["ta_bin"]  = make_bins(ta,  int(n_bins))
    D["vpd_bin"] = make_bins(vpd, int(n_bins))

    # parse windows
    try:
        windows_days = [int(x.strip()) for x in str(windows).split(",") if str(x).strip()!=""]
        windows_days = [w for w in windows_days if w>0]
    except Exception:
        windows_days = [7,14,30]

    report_rows = []
    filled_cols = []
    prog_overall = st.progress(0)

    for j, col in enumerate(targets, start=1):
        st.write(f"Filling: **{col}**")
        prog_col = st.progress(0)

        s = D[col] if col in D.columns else pd.Series(np.nan, index=D.index)
        before = summarize_gaps(s, zero_is_nan=bool(zero_as_nan), zero_eps=float(zero_eps))

        gf, flag = mds_fill_column(s, D["hour"], D["rg_bin"], D["ta_bin"], D["vpd_bin"], D["DateTime"],
                                   windows_days=windows_days, min_matches=int(min_matches),
                                   relax_bins=bool(relax_bins), zero_as_nan=bool(zero_as_nan),
                                   zero_eps=float(zero_eps), progress=prog_col)

        out_name = f"{col}_GF"
        D[out_name] = gf
        D[f"{col}_GF_filledFlag"] = flag

        after_missing = int(pd.isna(gf).sum())
        report_rows.append({
            "column": col,
            "n": before["n"],
            "missing_before": before["n_missing"],
            "missing_before_pct": before["missing_pct"],
            "missing_after": after_missing,
            "missing_after_pct": (100.0*after_missing/max(before["n"],1))
        })
        filled_cols.append(out_name)
        prog_overall.progress(int(100*j/max(len(targets),1)))

    report = pd.DataFrame(report_rows)
    st.subheader("Coverage report")
    st.dataframe(report, use_container_width=True)

    # Exports
    complete = D.copy()
    only = D[["DateTime"] + filled_cols].copy()

    st.subheader("Downloads")
    st.download_button("⬇️ STEP4_MDS_Gapfilled_COMPLETE.csv",
                       complete.to_csv(index=False).encode("utf-8"),
                       "STEP4_MDS_Gapfilled_COMPLETE.csv", "text/csv")
    st.download_button("⬇️ STEP4_MDS_Gapfilled_ONLY.csv",
                       only.to_csv(index=False).encode("utf-8"),
                       "STEP4_MDS_Gapfilled_ONLY.csv", "text/csv")
    st.download_button("⬇️ STEP4_MDS_Report.csv",
                       report.to_csv(index=False).encode("utf-8"),
                       "STEP4_MDS_Report.csv", "text/csv")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("STEP4_MDS_Gapfilled_COMPLETE.csv", complete.to_csv(index=False))
        z.writestr("STEP4_MDS_Gapfilled_ONLY.csv", only.to_csv(index=False))
        z.writestr("STEP4_MDS_Report.csv", report.to_csv(index=False))
    st.download_button("⬇️ Download ALL (ZIP)",
                       data=zbuf.getvalue(), file_name="STEP4_MDS_gapfill_bundle.zip", mime="application/zip")
