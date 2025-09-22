
# ec_suite_app_v8_multi_v3merge_allvars_qcflags_plausible__STRICT_ADDON.py
# -----------------------------------------------------------------------------
# Window 1: Multi‑file V3 merge (standard) + ALL‑VARIABLES 5‑min grid
# Window 2: Footprint allocation (FAST bin+cache) with QC incl. EddyPro flags,
#           plausible ranges for NEE/GPP/Reco, and NEW: STRICT assignment
#           (fraction ≥ threshold → full flux) with 5‑min + daily (gC) exports.
# -----------------------------------------------------------------------------

import os, re, tempfile, datetime as _dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import io

# --- Local column-name deduplicator (avoids pandas reindex errors on concat) ---
def _dedup(names):
    seen = {}
    out = []
    for n in [str(x) for x in names]:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
    return out


# Use your footprint+GIS core (falls back gracefully if libs absent)
from ffa_core import (
    compute_footprint_2d, rotate_to_EN, sample_raster_under_footprint,
    aggregate_by_class, HAVE_GIS
)

# ============================== Small helpers ==============================

def fetch_url_to_bytes(url: str) -> bytes:
    """
    Download a file from a URL and return its raw bytes.
    Supports common share links (Google Drive, Dropbox, GitHub).
    """
    try:
        import requests, re
    except Exception as e:
        raise RuntimeError("The 'requests' library is required to fetch URLs but is not available.") from e
    u = (url or "").strip()
    if not u:
        raise ValueError("Empty URL.")
    # Google Drive → direct download
    if "drive.google.com" in u:
        m = re.search(r"/file/d/([^/]+)", u) or re.search(r"[?&]id=([^&]+)", u)
        if m:
            u = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    # Dropbox → force download
    if "dropbox.com" in u:
        u = u.replace("?dl=0", "?dl=1")
    # GitHub blob → raw
    if "github.com" in u and "/blob/" in u:
        u = u.replace("/blob/", "/raw/")
    # Fetch
    resp = requests.get(u, timeout=45)
    resp.raise_for_status()
    return resp.content

def _parse_paths_list(text: str) -> List[str]:
    if not text:
        return []
    toks = re.split(r"[,;\n]+", text)
    paths = [t.strip() for t in toks if t.strip()]
    return paths

def _ensure_datetime(df: pd.DataFrame, col: str = "DateTime") -> pd.DataFrame:
    d = df.copy()
    d[col] = pd.to_datetime(d[col], errors="coerce")
    return d

def concat_and_coalesce(dfs: List[pd.DataFrame], dt_col: str = "DateTime", how: str = "last") -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    d = pd.concat(dfs, ignore_index=True, sort=False)
    from ffa_core import deduplicate   # you already have this helper in ffa_core.py
    d.columns = deduplicate(list(d.columns))
    d = _ensure_datetime(d, dt_col).dropna(subset=[dt_col]).sort_values(dt_col)
    if how == "none":
        return d
    if how in ("first", "last"):
        return d.drop_duplicates(subset=[dt_col], keep=how).sort_values(dt_col).reset_index(drop=True)
    if how == "mean":
        num = d.select_dtypes(include=[np.number]).columns.tolist()
        agg = {c: "mean" for c in num}
        others = [c for c in d.columns if c not in num + [dt_col]]
        for c in others:
            agg[c] = "first"
        out = d.groupby(dt_col, as_index=False).agg(agg).sort_values(dt_col).reset_index(drop=True)
        return out
    return d

def save_csv_with_dt_format(df: pd.DataFrame, dt_col: str, fmt: Optional[str]) -> bytes:
    if fmt is None:
        return df.to_csv(index=False).encode("utf-8")
    d = df.copy()
    if dt_col in d.columns:
        d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce").dt.strftime(fmt)
    return d.to_csv(index=False).encode("utf-8")

def dt_input(label_prefix: str, default_dt: pd.Timestamp, key_prefix: str) -> _dt.datetime:
    default_dt = pd.to_datetime(default_dt, errors="coerce")
    if pd.isna(default_dt):
        default_dt = pd.Timestamp.utcnow().floor("min")
    if hasattr(st, "datetime_input"):
        return st.datetime_input(label_prefix, value=default_dt.to_pydatetime(), key=f"{key_prefix}_dt")
    d_val = st.date_input(f"{label_prefix} — date", value=default_dt.date(), key=f"{key_prefix}_d")
    t_val = st.time_input(f"{label_prefix} — time", value=_dt.time(default_dt.hour, default_dt.minute), key=f"{key_prefix}_t")
    return _dt.datetime.combine(d_val, t_val)

def infer_step_minutes(ts) -> int:
    """Infer dominant step minutes (robust to gaps)."""
    t = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
    if len(t) < 3:
        return 5
    diffs = t.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 5
    vals = (diffs / 60.0).round().astype(int)
    try:
        return int(vals.mode().iloc[0])
    except Exception:
        return int(np.nanmedian(vals)) if len(vals) else 5

# ========================== V3 MERGE (with QC columns) ==========================

def read_eddypro_full_output(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        df0 = pd.read_csv(source, header=None, low_memory=False)
    else:
        df0 = pd.read_csv(str(source), header=None, low_memory=False)
    names = df0.iloc[1].fillna("").astype(str).tolist()
    df = df0.iloc[3:].copy()
    df.columns = names
    if "date" in df.columns and "time" in df.columns:
        df["DateTime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    else:
        df["DateTime"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    return df

def read_eddypro_biomet(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        df0 = pd.read_csv(source, header=None, low_memory=False)
    else:
        df0 = pd.read_csv(str(source), header=None, low_memory=False)
    names = df0.iloc[0].astype(str).tolist()
    df = df0.iloc[2:].copy()
    df.columns = names
    if "date" in df.columns and "time" in df.columns:
        df["DateTime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    else:
        df["DateTime"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    return df

def nearest_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, tol_min: int = 2) -> pd.DataFrame:
    return pd.merge_asof(
        left.sort_values(on), right.sort_values(on),
        on=on, direction="nearest", tolerance=pd.Timedelta(f"{tol_min}min")
    )

def compute_vpd_from_t_rh(Tair_C: pd.Series, RH_pct: pd.Series) -> pd.Series:
    T = pd.to_numeric(Tair_C, errors="coerce")
    RH = pd.to_numeric(RH_pct, errors="coerce")
    es = 6.112 * np.exp((17.67*T) / (T + 243.5))  # hPa
    ea = es * (RH / 100.0)
    VPD = es - ea
    return VPD

def build_standard_from_eddypro(flux: pd.DataFrame, bio: pd.DataFrame, step_min: int=5) -> pd.DataFrame:
    keep_flux = [
        "DateTime","co2_flux","H","LE","air_temperature","VPD","u*","wind_speed","wind_dir","L",
        # EddyPro QC flags if present:
        "qc_co2_flux","qc_H","qc_LE","qc_h2o_flux","qc_ch4_flux"
    ]
    present = [k for k in keep_flux if k in flux.columns]
    fsub = flux[present].copy()

    rg_candidates = ["SW_IN","Rg","RG","RG_1_1_1","SW_IN_1_1_1","SW_IN_1_1_1_1"]
    rg_col = next((c for c in rg_candidates if c in bio.columns), None)

    tair_candidates = ["TA","Tair","AIR_T_1_1_1","TA_1_1_1"]
    ta_bio_col = next((c for c in tair_candidates if c in bio.columns), None)

    rh_candidates = ["RH","RH_1_1_1","RH_2_1_1"]
    rh_col = next((c for c in rh_candidates if c in bio.columns), None)

    bcols = ["DateTime"] + [c for c in [rg_col, ta_bio_col, rh_col] if c is not None]
    bsub = bio[bcols].copy()

    df = nearest_merge(fsub, bsub, on="DateTime", tol_min=2)

    out = pd.DataFrame()
    out["DateTime"] = df["DateTime"]
    out["NEE"] = pd.to_numeric(df["co2_flux"], errors="coerce") if "co2_flux" in df.columns else np.nan

    # carry QC flags if present
    for q in ["qc_co2_flux","qc_H","qc_LE","qc_h2o_flux","qc_ch4_flux"]:
        if q in df.columns:
            out[q] = pd.to_numeric(df[q], errors="coerce")

    if rg_col:
        out["Rg"] = pd.to_numeric(df[rg_col], errors="coerce")
    else:
        out["Rg"] = np.nan
    if "air_temperature" in df.columns:
        out["Tair"] = pd.to_numeric(df["air_temperature"], errors="coerce") - 273.15
    elif ta_bio_col:
        s = pd.to_numeric(df[ta_bio_col], errors="coerce")
        out["Tair"] = np.where(s > 200, s - 273.15, s)
    else:
        out["Tair"] = np.nan
    if "VPD" in df.columns:
        out["VPD"] = pd.to_numeric(df["VPD"], errors="coerce") / 100.0
    else:
        if rh_col and "Tair" in out.columns:
            out["VPD"] = compute_vpd_from_t_rh(out["Tair"], pd.to_numeric(df[rh_col], errors="coerce"))
        else:
            out["VPD"] = np.nan
    out["Ustar"] = pd.to_numeric(df["u*"], errors="coerce") if "u*" in df.columns else np.nan
    out["WS"]    = pd.to_numeric(df["wind_speed"], errors="coerce") if "wind_speed" in df.columns else np.nan
    out["WD"]    = pd.to_numeric(df["wind_dir"], errors="coerce") if "wind_dir" in df.columns else np.nan
    out["L"]     = pd.to_numeric(df["L"], errors="coerce") if "L" in df.columns else np.nan

    t0, t1 = out["DateTime"].min(), out["DateTime"].max()
    full = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq=f"{step_min}min")})
    out = full.merge(out, on="DateTime", how="left")

    return out

# =================== ALL‑VARIABLES 5‑min grid (keeps every column) ===================

def build_allvars_grid(flux_df: pd.DataFrame, bio_df: pd.DataFrame, step_min:int=5, tol_min:int=2, prefix:bool=True) -> pd.DataFrame:
    f = flux_df.copy(); b = bio_df.copy()
    f["DateTime"] = pd.to_datetime(f["DateTime"], errors="coerce")
    b["DateTime"] = pd.to_datetime(b["DateTime"], errors="coerce")
    t0 = min(f["DateTime"].min(), b["DateTime"].min())
    t1 = max(f["DateTime"].max(), b["DateTime"].max())
    grid = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq=f"{int(step_min)}min")})
    for c in ("date","time"):
        if c in f.columns: f = f.drop(columns=[c])
        if c in b.columns: b = b.drop(columns=[c])
    if prefix:
        f = f.rename(columns={c: f"FO_{c}" for c in f.columns if c != "DateTime"})
        b = b.rename(columns={c: f"BIO_{c}" for c in b.columns if c != "DateTime"})
    grid = pd.merge_asof(grid.sort_values("DateTime"),
                         f.sort_values("DateTime"),
                         on="DateTime", direction="nearest",
                         tolerance=pd.Timedelta(f"{int(tol_min)}min"))
    grid = pd.merge_asof(grid.sort_values("DateTime"),
                         b.sort_values("DateTime"),
                         on="DateTime", direction="nearest",
                         tolerance=pd.Timedelta(f"{int(tol_min)}min"))
    return grid
# =============== NEW: standardize from external ALL-VARIABLES (FO_/BIO_) CSV ===============
def _coalesce_first(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    s = None
    for c in candidates:
        if c in df.columns:
            sc = pd.to_numeric(df[c], errors="coerce")
            s = sc if s is None else s.combine_first(sc)
    if s is None:
        return pd.Series([np.nan]*len(df), index=df.index)
    return s

def standardize_from_allvars(allvars: pd.DataFrame) -> pd.DataFrame:
    """Take an external merged_allvars_5min.csv (FO_/BIO_ prefixes) and
    construct the standard table used by Window-2 (NEE, Rg, Tair, VPD, Ustar, WS, WD, L, qc_* if present).
    """
    df = allvars.copy()
    if "DateTime" not in df.columns:
        raise ValueError("External ALL-VARIABLES file must contain 'DateTime' column.")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    out = pd.DataFrame()
    out["DateTime"] = df["DateTime"]

    # NEE from FULL OUTPUT co2_flux
    if "FO_co2_flux" in df.columns:
        out["NEE"] = pd.to_numeric(df["FO_co2_flux"], errors="coerce")
    elif "co2_flux" in df.columns:
        out["NEE"] = pd.to_numeric(df["co2_flux"], errors="coerce")
    else:
        out["NEE"] = np.nan

    # EddyPro QC flags if present (prefixed)
    for q in ["qc_co2_flux","qc_H","qc_LE","qc_h2o_flux","qc_ch4_flux"]:
        col = f"FO_{q}"
        if col in df.columns:
            out[q] = pd.to_numeric(df[col], errors="coerce")

    # Shortwave incoming radiation (prefer BIOMET)
    rg_candidates = [c for c in [
        "BIO_SW_IN","BIO_Rg","BIO_RG","BIO_RG_1_1_1","BIO_SW_IN_1_1_1","BIO_SW_IN_1_1_1_1",
        # sometimes people rename with lower-case
        "BIO_sw_in","BIO_rg","BIO_RG_1","BIO_RG_2"
    ] if c in df.columns]
    if rg_candidates:
        out["Rg"] = _coalesce_first(df, rg_candidates)
    else:
        # fall back to any FO Rg if present
        out["Rg"] = pd.to_numeric(df.get("FO_Rg", np.nan), errors="coerce")

    # Air temperature (K in FULL OUTPUT; °C in most BIOMETs)
    if "FO_air_temperature" in df.columns:
        out["Tair"] = pd.to_numeric(df["FO_air_temperature"], errors="coerce") - 273.15
    else:
        tair_candidates = [c for c in ["BIO_TA","BIO_Tair","BIO_AIR_T_1_1_1","BIO_TA_1_1_1","BIO_T_1_1_1"] if c in df.columns]
        if tair_candidates:
            s = _coalesce_first(df, tair_candidates)
            out["Tair"] = np.where(s > 200, s - 273.15, s)
        else:
            out["Tair"] = np.nan

    # VPD (Pa in FULL OUTPUT -> divide by 100 to hPa)
    if "FO_VPD" in df.columns:
        out["VPD"] = pd.to_numeric(df["FO_VPD"], errors="coerce") / 100.0
    else:
        rh_candidates = [c for c in ["BIO_RH","BIO_RH_1_1_1","BIO_RH_2_1_1"] if c in df.columns]
        if rh_candidates:
            rh = _coalesce_first(df, rh_candidates)
            out["VPD"] = compute_vpd_from_t_rh(out["Tair"], rh)
        else:
            out["VPD"] = np.nan

    # Friction velocity, wind speed/dir, Monin-Obukhov L from FULL OUTPUT
    out["Ustar"] = pd.to_numeric(df.get("FO_u*", np.nan), errors="coerce") if "FO_u*" in df.columns else pd.to_numeric(df.get("FO_ustar", np.nan), errors="coerce")
    out["WS"]    = pd.to_numeric(df.get("FO_wind_speed", np.nan), errors="coerce")
    out["WD"]    = pd.to_numeric(df.get("FO_wind_dir", np.nan), errors="coerce")
    out["L"]     = pd.to_numeric(df.get("FO_L", np.nan), errors="coerce")

    # Ensure 5-min grid continuity using the min..max of DateTime
    if not out.empty:
        t0, t1 = out["DateTime"].min(), out["DateTime"].max()
        full = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq="5min")})
        out = pd.merge(full, out, on="DateTime", how="left")
    return out


# ========================= FAST Footprint (bin+cache) =========================
@dataclass
class BinSpec:
    L_step: float = 20.0
    Ustar_step: float = 0.05
    WS_step: float = 0.5
    WD_step: float = 10.0

def _bin_value(x: float, step: float) -> float:
    if np.isnan(x) or step <= 0:
        return np.nan
    return float(np.round(x/step) * step)

def allocate_fast(base_df: pd.DataFrame,
                  rpath: str, tower_lat: float, tower_lon: float,
                  grid_size: float, dx: float, nodata_val: Optional[float],
                  classes_label: Dict[int,str], bs: BinSpec,
                  progress: bool = True) -> pd.DataFrame:
    d = base_df.copy()
    d["timestamp"] = d["DateTime"]
    arrL  = pd.to_numeric(d.get("L"), errors="coerce").values.astype(float)
    arrU  = pd.to_numeric(d.get("Ustar"), errors="coerce").values.astype(float)
    arrWS = pd.to_numeric(d.get("WS"), errors="coerce").values.astype(float)
    arrWD = pd.to_numeric(d.get("WD"), errors="coerce").values.astype(float)

    sigs = np.empty((len(d), 4), dtype=float)
    for i in range(len(d)):
        sig = (_bin_value(arrL[i], bs.L_step), _bin_value(arrU[i], bs.Ustar_step),
               _bin_value(arrWS[i], bs.WS_step), _bin_value(arrWD[i], bs.WD_step))
        sigs[i,0], sigs[i,1], sigs[i,2], sigs[i,3] = sig

    sig_to_idx: Dict[Tuple[float,float,float,float], List[int]] = {}
    for i in range(len(d)):
        sig = (sigs[i,0], sigs[i,1], sigs[i,2], sigs[i,3])
        sig_to_idx.setdefault(sig, []).append(i)

    sig_to_fracs: Dict[Tuple[float,float,float,float], Dict[int,float]] = {}
    unique_sigs = list(sig_to_idx.keys())
    n_unique = len(unique_sigs)

    prog = st.progress(0) if progress else None
    for k, sig in enumerate(unique_sigs, start=1):
        try:
            Lb, Ub, Umeanb, WDb = sig
            L_i = -100.0 if (np.isnan(Lb)) else float(Lb)
            ustar_i = 0.3 if (np.isnan(Ub)) else float(Ub)
            U_i = 3.0 if (np.isnan(Umeanb)) else float(Umeanb)
            wd_i = 225.0 if (np.isnan(WDb)) else float(WDb)
            X, Y, W = compute_footprint_2d(zm=30.0, d=0.0, z0=0.15, L=L_i, ustar=ustar_i, Umean=U_i, sigmav=0.5,
                                           wind_dir_deg=wd_i, grid_size=float(grid_size), dx=float(dx), use_pyffp=False)
            Xe, Yn = rotate_to_EN(X, Y, wd_i)
            if (rpath is None) or (not HAVE_GIS):
                fr = {}
            else:
                classes, meta = sample_raster_under_footprint(rpath, float(tower_lon), float(tower_lat), Xe, Yn, nodata_val)
                fr = aggregate_by_class(classes, W, nodata=meta.get("nodata", None))
        except Exception:
            fr = {}
        sig_to_fracs[sig] = fr
        if prog:
            prog.progress(min(1.0, k / max(1, n_unique)))
    if prog: prog.empty()

    all_cids = sorted(set([cid for fr in sig_to_fracs.values() for cid in fr.keys() if cid is not None]))
    wide = pd.DataFrame({"timestamp": d["timestamp"].astype(str)})
    for cid in all_cids:
        wide[cid] = np.nan
    for sig, idxs in sig_to_idx.items():
        fr = sig_to_fracs.get(sig, {})
        for cid in all_cids:
            val = float(fr.get(cid, 0.0))
            wide.iloc[idxs, wide.columns.get_loc(cid)] = val
    wide["DateTime"] = pd.to_datetime(d["DateTime"])
    return wide

# ================================ STRICT assignment helpers ================================

def build_strict_tables(subset_qc: pd.DataFrame, wide_frac: pd.DataFrame, flux_col: str,
                        thr: float = 0.4, mode: str = "multi", fill_unassigned_zero: bool = False,
                        labels: Optional[Dict[int,str]] = None):
    """
    Parameters
    ----------
    subset_qc : DataFrame with DateTime and <flux_col> (µmol m⁻² s⁻¹)
    wide_frac : DataFrame from allocate_fast with DateTime + class‑ID fraction columns
    flux_col  : Which flux to assign (e.g., 'NEE')
    thr       : Fraction threshold (e.g., 0.4 -> 40%)
    mode      : 'multi' → assign to all classes with frac ≥ thr;
                'top1'  → assign only to the max‑fraction class if its frac ≥ thr
    fill_unassigned_zero : if True → unassigned cells set to 0; else NaN
    labels    : optional {class_id: label}
    Returns
    -------
    strict_5min : 5‑min table (µmol m⁻² s⁻¹)
    daily_gC    : daily sums (gC m⁻² day⁻¹)
    cid_cols    : list of class IDs
    """
    tmp = wide_frac.drop(columns=[c for c in ["timestamp"] if c in wide_frac.columns]).copy()
    cid_cols = [c for c in tmp.columns if isinstance(c, (int, np.integer))]

    W = pd.merge(subset_qc[["DateTime", flux_col]].copy(), tmp, on="DateTime", how="left")
    for cid in cid_cols:
        W[f"{flux_col}_alloc_STRICT_class_{cid}"] = (0.0 if fill_unassigned_zero else np.nan)

    frac_mat = W[cid_cols].apply(pd.to_numeric, errors="coerce")
    flux_vec = pd.to_numeric(W[flux_col], errors="coerce")

    if mode == "multi":
        for cid in cid_cols:
            m = frac_mat[cid] >= float(thr)
            W.loc[m, f"{flux_col}_alloc_STRICT_class_{cid}"] = flux_vec[m]
        W["assigned_class_id"] = np.nan
        W["assigned_label"] = ""
        W["assigned_fraction"] = np.nan
    else:  # top1
        best_cid = frac_mat.idxmax(axis=1)
        best_val = frac_mat.max(axis=1)
        keep = best_val >= float(thr)
        for cid in cid_cols:
            sel = keep & (best_cid == cid)
            W.loc[sel, f"{flux_col}_alloc_STRICT_class_{cid}"] = flux_vec[sel]
        W["assigned_class_id"] = best_cid.where(keep, np.nan)
        lab_map = (labels or {})
        W["assigned_label"] = W["assigned_class_id"].map(lambda x: lab_map.get(int(x), "") if pd.notna(x) else "")
        W["assigned_fraction"] = best_val.where(keep, np.nan)

    alloc_cols = [f"{flux_col}_alloc_STRICT_class_{cid}" for cid in cid_cols]
    cols = (["DateTime", flux_col, "assigned_class_id", "assigned_label", "assigned_fraction"] + alloc_cols) \
           if "assigned_class_id" in W.columns else (["DateTime", flux_col] + alloc_cols)
    strict_5min = W[cols].copy()

    step_min = infer_step_minutes(strict_5min["DateTime"])
    fac = step_min * 60.0 * 12e-6
    DD = strict_5min.copy()
    DD["Date"] = pd.to_datetime(DD["DateTime"]).dt.date
    daily_gC = DD.groupby("Date")[alloc_cols].sum() * fac
    daily_gC = daily_gC.reset_index()
    return strict_5min, daily_gC, cid_cols

# ================================ UI ================================

st.set_page_config(page_title="EC Suite — MULTI‑FILE V3 merge + ALL‑VARS + FAST footprint (QC+flags+STRICT)", layout="wide")
st.sidebar.title("EC Flux Suite — Multi‑file")

page = st.sidebar.radio("Windows (pages)",
                        ["1) Multi‑file V3 merge + ALL‑VARIABLES",
                         "2) Footprint allocation (FAST) + QC + Flags + STRICT"],
                        index=0)

# Session
for k in ["merged_std","merged_allvars","fractions_wide","fractions_long","class_labels","tmpdir"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Sidebar downloads with UNIQUE KEYS
st.sidebar.markdown("---")
st.sidebar.subheader("Window‑1 Outputs")
if st.session_state.get("merged_std") is not None and not st.session_state["merged_std"].empty:
    st.sidebar.download_button("⬇️ merged_for_next_windows.csv",
                               st.session_state["merged_std"].to_csv(index=False).encode("utf-8"),
                               "merged_for_next_windows.csv", "text/csv",
                               key="dl_std_sidebar_v8s")
if st.session_state.get("merged_allvars") is not None and not st.session_state["merged_allvars"].empty:
    st.sidebar.download_button("⬇️ merged_allvars_5min.csv",
                               st.session_state["merged_allvars"].to_csv(index=False).encode("utf-8"),
                               "merged_allvars_5min.csv", "text/csv",
                               key="dl_allvars_sidebar_v8s")

# ============================== WINDOW 1 ==============================
if page.startswith("1)"):
    st.title("Window 1 — Multi‑file V3 merge (standard) **and** ALL‑VARIABLES 5‑min grid")
    st.caption("Standard table follows V3 logic; also carries EddyPro QC flags if present.")

    st.subheader("1) Load FULL OUTPUT files")
    c1, c2 = st.columns(2)
    with c1:
        f_flux_list = st.file_uploader("FULL OUTPUT CSV(s)", type=["csv"], accept_multiple_files=True, key="flux_upl_v8s")
    with c2:
        flux_paths_text = st.text_area("…or local FULL OUTPUT path(s) (comma/semicolon/newline separated)", value="", key="flux_paths_v8s")

    st.subheader("2) Load BIOMET files")
    c3, c4 = st.columns(2)
    with c3:
        f_bio_list  = st.file_uploader("BIOMET CSV(s)", type=["csv"], accept_multiple_files=True, key="bio_upl_v8s")
    with c4:
        bio_paths_text = st.text_area("…or local BIOMET path(s) (comma/semicolon/newline separated)", value="", key="bio_paths_v8s")

    st.subheader("3) Merge options")
    c5, c6, c7 = st.columns(3)
    with c5:
        coalesce_flux = st.selectbox("Coalesce duplicates (FULL OUTPUT)", ["last","first","mean","none"], index=0, key="coal_flux_v8s")
    with c6:
        coalesce_bio  = st.selectbox("Coalesce duplicates (BIOMET)", ["last","first","mean","none"], index=0, key="coal_bio_v8s")
    with c7:
        tol_min = st.number_input("Nearest‑merge tolerance (min)", min_value=0, max_value=10, value=2, step=1, key="tol_v8s")

    st.subheader("4) Timestamp format on **CSV export**")
    fmt_choice = st.radio("How to write DateTime into CSV?", ["YYYY‑MM‑DD HH:MM","YYYY‑MM‑DD HH:MM:SS","Pandas default (no forced format)"], index=0, key="fmt_v8s")
    fmt_map = {"YYYY‑MM‑DD HH:MM": "%Y-%m-%d %H:%M","YYYY‑MM‑DD HH:MM:SS": "%Y-%m-%d %H:%M:%S","Pandas default (no forced format)": None}
    dt_fmt = fmt_map.get(fmt_choice, "%Y-%m-%d %H:%M")

    run = st.button("▶️ Run multi‑file merge", key="run_merge_v8s")

    if run:
        # Gather FULL OUTPUT sources
        flux_sources: List[pd.DataFrame] = []
        if f_flux_list:
            for f in f_flux_list:
                try:
                    flux_sources.append(read_eddypro_full_output(f))
                except Exception as e:
                    st.error(f"FULL OUTPUT upload '{getattr(f,'name','')}' failed: {e}")
        for p in _parse_paths_list(flux_paths_text):
            if os.path.exists(p):
                try:
                    flux_sources.append(read_eddypro_full_output(p))
                except Exception as e:
                    st.error(f"FULL OUTPUT path '{p}' failed: {e}")
            else:
                st.warning(f"FULL OUTPUT path not found: {p}")
        # Gather BIOMET sources
        bio_sources: List[pd.DataFrame] = []
        if f_bio_list:
            for f in f_bio_list:
                try:
                    bio_sources.append(read_eddypro_biomet(f))
                except Exception as e:
                    st.error(f"BIOMET upload '{getattr(f,'name','')}' failed: {e}")
        for p in _parse_paths_list(bio_paths_text):
            if os.path.exists(p):
                try:
                    bio_sources.append(read_eddypro_biomet(p))
                except Exception as e:
                    st.error(f"BIOMET path '{p}' failed: {e}")
            else:
                st.warning(f"BIOMET path not found: {p}")

        if (not flux_sources) or (not bio_sources):
            st.error("Provide at least one FULL OUTPUT and one BIOMET file (upload or path).")
        else:
            flux_union = concat_and_coalesce(flux_sources, dt_col="DateTime", how=coalesce_flux)
            bio_union  = concat_and_coalesce(bio_sources,  dt_col="DateTime", how=coalesce_bio)

            merged_std = build_standard_from_eddypro(flux_union, bio_union, step_min=5)
            st.session_state["merged_std"] = merged_std.copy()

            merged_all = build_allvars_grid(flux_union, bio_union, step_min=5, tol_min=int(tol_min), prefix=True)
            st.session_state["merged_allvars"] = merged_all.copy()

            st.success(f"Done. STANDARD rows: {len(merged_std):,} | ALL‑VARIABLES rows: {len(merged_all):,}, cols: {len(merged_all.columns):,}")
            with st.expander("Preview — standard (V3 + QC flags)", expanded=False):
                st.dataframe(merged_std.head(30), use_container_width=True)
            with st.expander("Preview — ALL‑VARIABLES (prefixed)", expanded=False):
                st.dataframe(merged_all.head(30), use_container_width=True)

            st.download_button("⬇️ merged_for_next_windows.csv",
                               save_csv_with_dt_format(merged_std, "DateTime", dt_fmt),
                               "merged_for_next_windows.csv", "text/csv",
                               key="dl_std_main_v8s")
            st.download_button("⬇️ merged_allvars_5min.csv",
                               save_csv_with_dt_format(merged_all, "DateTime", dt_fmt),
                               "merged_allvars_5min.csv", "text/csv",
                               key="dl_allvars_main_v8s")

# ============================== WINDOW 2 ==============================
else:
    st.title("Window 2 — Footprint allocation (FAST bin+cache) + QC + Flags + STRICT")

    merged = st.session_state.get("merged_std")
    
    if (merged is None) or merged.empty:
        st.warning("No standard table in session. Load one of the Window‑1 exports below.")
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            up_std = st.file_uploader("Upload **merged_for_next_windows.csv** (STANDARD)", type=["csv"], key="merged_std_upl_v8s")
        with c_up2:
            up_all = st.file_uploader("Upload **merged_allvars_5min.csv** (ALL‑VARIABLES, FO_/BIO_ prefixes)", type=["csv"], key="merged_allvars_upl_v8s")

        if up_std is not None:
            try:
                dfm = pd.read_csv(up_std)
                if "DateTime" not in dfm.columns:
                    st.error("Uploaded STANDARD file must contain 'DateTime' column.")
                else:
                    dfm["DateTime"] = pd.to_datetime(dfm["DateTime"], errors="coerce")
                    st.session_state["merged_std"] = dfm.sort_values("DateTime").reset_index(drop=True)
                    merged = st.session_state["merged_std"]
                    st.success(f"Loaded STANDARD. Rows: {len(merged):,}")
            except Exception as e:
                st.error(f"Failed to read STANDARD file: {e}")

        if ((merged is None) or merged.empty) and (up_all is not None):
            try:
                dfa = pd.read_csv(up_all)
                std = standardize_from_allvars(dfa)
                st.session_state["merged_std"] = std.sort_values("DateTime").reset_index(drop=True)
                merged = st.session_state["merged_std"]
                st.success(f"Loaded ALL‑VARIABLES and converted to STANDARD columns. Rows: {len(merged):,}")
                with st.expander("Preview (converted from ALL‑VARIABLES)", expanded=False):
                    st.dataframe(merged.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to convert ALL‑VARIABLES file: {e}")

    # NEW: Option to fetch ALL‑VARIABLES directly from a URL
    if ((merged is None) or merged.empty):
        st.markdown("**…or fetch ALL‑VARIABLES by URL**")
        allvars_url = st.text_input("URL to merged_allvars_5min.csv", value="", key="merged_allvars_url_v8s")
        fetch_allvars = st.button("⬇️ Fetch ALL‑VARIABLES from URL", key="fetch_allvars_url_btn_v8s")
        if fetch_allvars and allvars_url.strip():
            try:
                raw = fetch_url_to_bytes(allvars_url.strip())
                # auto-detect delimiter
                dfa = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", low_memory=False)
                std = standardize_from_allvars(dfa)
                st.session_state["merged_std"] = std.sort_values("DateTime").reset_index(drop=True)
                merged = st.session_state["merged_std"]
                st.success(f"Downloaded and converted ALL‑VARIABLES from URL. Rows: {len(merged):,}")
                with st.expander("Preview (converted from ALL‑VARIABLES via URL)", expanded=False):
                    st.dataframe(merged.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to download/convert ALL‑VARIABLES from URL: {e}")


    # Raster & tower
    st.subheader("1) Raster & tower")
    c1, c2, c3 = st.columns(3)
    with c1:
        grid_size = st.number_input("Half grid size (m)", value=150, step=10)
        dx = st.number_input("Grid resolution (m)", value=3, step=1)
    with c2:
        tower_lat = st.number_input("Tower latitude (WGS84)", value=49.166000, format="%.6f")
        tower_lon = st.number_input("Tower longitude (WGS84)", value=20.283000, format="%.6f")
    with c3:
        raster = st.file_uploader("GeoTIFF (classes)", type=["tif","tiff"], key="raster_upl_v8s")
        raster_path_text = st.text_input("…or local GeoTIFF path", value="", key="raster_path_v8s")
        nodata_txt = st.text_input("Raster NoData (blank if 0 is valid)", value="", key="nodata_v8s")

    # Legend
    st.subheader("2) Legend (optional)")
    legend_csv = st.file_uploader("Legend CSV: class_id,label OR class_id,class_name", type=["csv"], key="legend_upl_v8s")
    legend_path_text = st.text_input("…or local legend CSV path", value="", key="legend_path_v8s")

    # QC panel (ranges + Hampel) + Flags + Plausible ranges
    st.subheader("3) Quality control (before allocation)")
    cqc1, cqc2, cqc3, cqc4 = st.columns(4)
    with cqc1:
        rg_zero = st.checkbox("Negative Rg → 0", value=True, key="qc_rg0_v8s")
        hampel_on = st.checkbox("Hampel spikes → NA (NEE)", value=True, key="qc_hampel_v8s")
    with cqc2:
        tair_min = st.number_input("Tair min (°C)", value=-50.0, step=0.5, key="qc_tmin_v8s")
        tair_max = st.number_input("Tair max (°C)", value=60.0, step=0.5, key="qc_tmax_v8s")
    with cqc3:
        vpd_min  = st.number_input("VPD min (hPa)", value=0.0, step=0.5, key="qc_vpdmin_v8s")
        vpd_max  = st.number_input("VPD max (hPa)", value=80.0, step=0.5, key="qc_vpdmax_v8s")
    with cqc4:
        ustar_min = st.number_input("u* min (m s⁻¹)", value=0.0, step=0.05, key="qc_umin_v8s")
        ustar_max = st.number_input("u* max (m s⁻¹)", value=2.0, step=0.05, key="qc_umax_v8s")

    cqc6, cqc7 = st.columns(2)
    with cqc6:
        hampel_w  = st.number_input("Hampel window (points)", min_value=3, max_value=61, value=7, step=2, key="qc_hw_v8s")
    with cqc7:
        hampel_k  = st.number_input("Hampel k (MAD multipliers)", min_value=1.0, max_value=10.0, value=3.5, step=0.5, key="qc_hk_v8s")

    # NEW: EddyPro quality flags removal for NEE (if available)
    st.markdown("**Quality-flag removal (EddyPro)**")
    cfl1, cfl2, cfl3 = st.columns(3)
    with cfl1:
        use_flag = st.checkbox("Apply qc_co2_flux filter to NEE", value=True, key="qc_flag_on_v8s")
    with cfl2:
        max_qc = st.selectbox("Max allowed qc_co2_flux", options=[0,1,2], index=1, key="qc_flag_thr_v8s")
    with cfl3:
        flag_mode = st.radio("Action when qc fails", ["set NEE→NA", "drop row"], horizontal=True, key="qc_flag_act_v8s")

    # NEW: Plausible ranges for NEE/GPP/Reco (set to NA when out-of-range; applied if columns exist)
    st.markdown("**Plausible ranges (set to NA when outside)**")
    rng1, rng2, rng3 = st.columns(3)
    with rng1:
        nee_min = st.number_input("NEE min (µmol m⁻² s⁻¹)", value=-50.0, step=1.0, key="rng_nee_min_v8s")
        nee_max = st.number_input("NEE max (µmol m⁻² s⁻¹)", value=50.0, step=1.0, key="rng_nee_max_v8s")
    with rng2:
        gpp_min = st.number_input("GPP min (µmol m⁻² s⁻¹)", value=0.0, step=1.0, key="rng_gpp_min_v8s")
        gpp_max = st.number_input("GPP max (µmol m⁻² s⁻¹)", value=60.0, step=1.0, key="rng_gpp_max_v8s")
    with rng3:
        re_min  = st.number_input("Reco min (µmol m⁻² s⁻¹)", value=0.0, step=1.0, key="rng_re_min_v8s")
        re_max  = st.number_input("Reco max (µmol m⁻² s⁻¹)", value=40.0, step=1.0, key="rng_re_max_v8s")

    skip_missing = st.checkbox("Skip rows with missing drivers (L, u*, WS, WD) after QC", value=False, key="qc_skip_missing_v8s")

    # FAST binning
    st.subheader("4) FAST binning")
    c4,c5,c6,c7 = st.columns(4)
    with c4:
        L_step = st.number_input("L step", value=20.0, step=5.0, key="bin_L_v8s")
    with c5:
        ustar_step = st.number_input("u* step", value=0.05, step=0.01, format="%.2f", key="bin_Ustar_v8s")
    with c6:
        WS_step = st.number_input("WS step", value=0.5, step=0.1, format="%.1f", key="bin_WS_v8s")
    with c7:
        WD_step = st.number_input("WD step", value=10.0, step=5.0, format="%.0f", key="bin_WD_v8s")

    # Time window (robust inputs)
    st.subheader("5) Optional time window")
    tmin, tmax = merged["DateTime"].min(), merged["DateTime"].max()
    cA, cB = st.columns(2)
    with cA:
        t0 = dt_input("Start", tmin, "t0_win2_v8s")
    with cB:
        t1 = dt_input("End",   tmax, "t1_win2_v8s")

    subset_df = merged[(merged["DateTime"]>=pd.to_datetime(t0)) & (merged["DateTime"]<=pd.to_datetime(t1))].copy()
    st.caption(f"Rows in window (before QC): {len(subset_df):,}")

    # Helpers for raster/legend
    def resolve_raster_path(raster_upload, raster_path_text: str):
        if raster_upload is not None:
            tmpdir = st.session_state.get("tmpdir") or tempfile.mkdtemp()
            st.session_state["tmpdir"] = tmpdir
            rpath = os.path.join(tmpdir, raster_upload.name)
            with open(rpath, "wb") as f:
                f.write(raster_upload.getbuffer())
            return rpath
        if raster_path_text.strip():
            if os.path.exists(raster_path_text):
                return raster_path_text
            else:
                st.warning(f"GeoTIFF path does not exist: {raster_path_text}")
        return None

    def parse_legend(legend_csv, legend_path_text: str) -> Dict[int,str]:
        labels = {}
        ldf = None
        if legend_csv is not None:
            ldf = pd.read_csv(legend_csv)
        elif legend_path_text.strip():
            if os.path.exists(legend_path_text):
                ldf = pd.read_csv(legend_path_text)
            else:
                st.warning(f"Legend path does not exist: {legend_path_text}")
        if ldf is not None:
            if "class_id" not in ldf.columns:
                st.warning("Legend must contain 'class_id'."); return labels
            label_col = "label" if "label" in ldf.columns else ("class_name" if "class_name" in ldf.columns else None)
            if label_col is None:
                st.warning("Legend must contain 'label' or 'class_name'."); return labels
            for _, rr in ldf.iterrows():
                try: labels[int(rr["class_id"])] = str(rr[label_col])
                except Exception: pass
        return labels

    try:
        nodata_val = None if (nodata_txt.strip()=="") else float(nodata_txt)
    except Exception:
        nodata_val = None

    rpath = resolve_raster_path(raster, raster_path_text)
    class_labels: Dict[int,str] = parse_legend(legend_csv, legend_path_text)
    if class_labels:
        st.session_state["class_labels"] = class_labels.copy()

    # QC helpers
    def hampel_filter_to_na(series: pd.Series, window: int = 7, k: float = 3.5) -> tuple[pd.Series, int]:
        x = pd.to_numeric(series, errors="coerce").astype(float)
        med = x.rolling(window=window, center=True, min_periods=1).median()
        mad = (x - med).abs().rolling(window=window, center=True, min_periods=1).median() * 1.4826
        diff = (x - med).abs()
        mask = diff > (k * mad.replace(0, np.nan))
        x[mask] = np.nan
        return x, int(mask.sum())

    def apply_qc_ranges(df_in: pd.DataFrame) -> pd.DataFrame:
        d = df_in.copy()
        if rg_zero and "Rg" in d.columns:
            x = pd.to_numeric(d["Rg"], errors="coerce"); d.loc[x < 0, "Rg"] = 0.0
        def clamp(col, lo, hi):
            if col in d.columns:
                x = pd.to_numeric(d[col], errors="coerce")
                d.loc[(x < lo) | (x > hi), col] = np.nan
        clamp("Tair", float(tair_min), float(tair_max))
        clamp("VPD", float(vpd_min), float(vpd_max))
        clamp("Ustar", float(ustar_min), float(ustar_max))
        # Plausible ranges on available flux variables
        clamp("NEE", float(nee_min), float(nee_max))
        clamp("GPP", float(gpp_min), float(gpp_max))
        clamp("Reco", float(re_min), float(re_max))
        return d

    # Apply QC
    subset_qc = apply_qc_ranges(subset_df)
    spike_count = 0
    if hampel_on and "NEE" in subset_qc.columns:
        subset_qc["NEE"], spike_count = hampel_filter_to_na(subset_qc["NEE"], window=int(hampel_w), k=float(hampel_k))
    st.caption(f"QC applied. Hampel spikes removed (NEE): {spike_count}")

    # EddyPro qc flag removal
    if use_flag and ("qc_co2_flux" in subset_qc.columns):
        m_bad = pd.to_numeric(subset_qc["qc_co2_flux"], errors="coerce") > int(max_qc)
        if flag_mode.startswith("set"):
            subset_qc.loc[m_bad, "NEE"] = np.nan
            st.caption(f"EddyPro QC applied to NEE: set {int(m_bad.sum())} points to NA (qc_co2_flux > {max_qc}).")
        else:
            before = len(subset_qc)
            subset_qc = subset_qc.loc[~m_bad].copy()
            st.caption(f"EddyPro QC applied to NEE: dropped {before - len(subset_qc)} rows (qc_co2_flux > {max_qc}).")
    elif use_flag:
        st.info("qc_co2_flux column not available; skip flag filter.")

    if skip_missing:
        before = len(subset_qc)
        subset_qc = subset_qc.dropna(subset=["L","Ustar","WS","WD"])
        st.caption(f"Dropped rows with missing L/u*/WS/WD after QC: {before - len(subset_qc)}")

    st.download_button("⬇️ subset_used_for_allocation.csv",
                       subset_qc.to_csv(index=False).encode("utf-8"),
                       "subset_used_for_allocation.csv", "text/csv",
                       key="dl_subset_qc_v8s")

    disabled_reason = None
    if rpath is None:
        disabled_reason = "Provide a GeoTIFF raster (upload or path)."
    elif subset_qc.empty:
        disabled_reason = "No rows in the selected window after QC."
    elif not HAVE_GIS:
        disabled_reason = "GIS libraries are not available (rasterio/pyproj). Allocation needs them."

    run = st.button("▶️ Run FAST footprint allocation", disabled=(disabled_reason is not None), key="run_fp_v8s")
    if disabled_reason:
        st.info(disabled_reason)

    if run:
        bs = BinSpec(L_step=float(L_step), Ustar_step=float(ustar_step), WS_step=float(WS_step), WD_step=float(WD_step))
        wide = allocate_fast(subset_qc, rpath, float(tower_lat), float(tower_lon),
                             float(grid_size), float(dx), nodata_val, class_labels, bs, progress=True)
        st.session_state["fractions_wide"] = wide.copy()

        # Long format
        tmp = wide.drop(columns=["timestamp"]).copy()
        cid_cols = [c for c in tmp.columns if isinstance(c, (int, np.integer))]
        long_rows = []
        for _, row in tmp.iterrows():
            dt = row["DateTime"]
            for cid in cid_cols:
                val = row[cid]
                if pd.notna(val):
                    long_rows.append({"DateTime": dt, "class_id": int(cid), "fraction": float(val),
                                      "label": class_labels.get(int(cid), "")})
        long_df = pd.DataFrame(long_rows) if long_rows else pd.DataFrame(columns=["DateTime","class_id","fraction","label"])
        st.session_state["fractions_long"] = long_df.copy()

        st.success(f"Allocation complete. Rows: {len(wide):,}; Classes: {len(cid_cols)}")
        st.dataframe(wide.head(20), use_container_width=True)

        st.download_button("⬇️ footprint_fractions_wide.csv",
                           wide.to_csv(index=False).encode("utf-8"),
                           "footprint_fractions_wide.csv", "text/csv",
                           key="dl_fp_wide_v8s")
        st.download_button("⬇️ footprint_fractions_long.csv",
                           long_df.to_csv(index=False).encode("utf-8"),
                           "footprint_fractions_long.csv", "text/csv",
                           key="dl_fp_long_v8s")

        # ============== NEW: STRICT assignment (fraction ≥ threshold → full flux) ==============
        st.markdown("---")
        st.subheader("6) STRICT priradenie: ak frakcia ≥ prah, priraď **plnú hodnotu** fluxu")

        possible_flux = [c for c in ["NEE","co2_flux","H","LE","h2o_flux","ch4_flux"] if c in subset_qc.columns]
        if not possible_flux:
            st.warning("V dataset-e chýba NEE/co2_flux/H/LE/h2o_flux/ch4_flux — nie je čo priraďovať.")
        else:
            flux_col = st.selectbox("Flux na STRICT priradenie", options=possible_flux, index=0, key="strict_flux_v8s")
            thr = st.number_input("Prah frakcie (0–1, napr. 0.40 = 40%)", 0.0, 1.0, 0.40, 0.05, key="strict_thr_v8s")
            mode = st.radio("Režim priradenia", ["Viacnásobné (všetky triedy ≥ prah)","Top‑1 trieda (ak ≥ prah)"], horizontal=True, key="strict_mode_v8s")
            fill0 = st.checkbox("Nepriradené → 0 (inak NA)", value=False, key="strict_fill0_v8s")

            labels_map = st.session_state.get("class_labels", {}) or {}
            strict_5min, strict_daily, _ = build_strict_tables(subset_qc, wide, flux_col=flux_col,
                                                               thr=float(thr), mode=("multi" if mode.startswith("Viac") else "top1"),
                                                               fill_unassigned_zero=bool(fill0), labels=labels_map)

            st.markdown("**5‑min STRICT (µmol m⁻² s⁻¹)**")
            st.dataframe(strict_5min.head(20), use_container_width=True)
            st.download_button("⬇️ allocated_flux_STRICT_5min_umol_s.csv",
                               strict_5min.to_csv(index=False).encode("utf-8"),
                               "allocated_flux_STRICT_5min_umol_s.csv", "text/csv",
                               key="dl_strict5_v8s")

            st.markdown("**Denné sumy STRICT (gC m⁻² deň⁻¹)**")
            st.dataframe(strict_daily.head(20), use_container_width=True)
            st.download_button("⬇️ allocated_flux_STRICT_daily_gC.csv",
                               strict_daily.to_csv(index=False).encode("utf-8"),
                               "allocated_flux_STRICT_daily_gC.csv", "text/csv",
                               key="dl_strictday_v8s")