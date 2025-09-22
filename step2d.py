
# footprint_flux_allocation_app_PROGRESS_BOTH_MERGED.py
# -----------------------------------------------------------------------------
# Standalone app: Flux allocation (RAW + STRICT) side-by-side with progress bar,
# optional Top‑1 fallback, footprint caching, and merged outputs with input.
# NOTE: Daily sums removed per request.
# -----------------------------------------------------------------------------

import os, io, tempfile, math, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps for GIS and Kljun
HAVE_RAST = HAVE_PYPROJ = HAVE_PYFFP = False
try:
    import rasterio
    from rasterio.transform import rowcol, xy
    HAVE_RAST = True
except Exception:
    HAVE_RAST = False

try:
    from pyproj import Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

try:
    import pyffp as _pyffp  # Kljun 2015
    HAVE_PYFFP = True
except Exception:
    HAVE_PYFFP = False

st.set_page_config(page_title="Footprint Flux Allocation (RAW + STRICT, Merged)", layout="wide")
st.title("Footprint Flux Allocation — RAW + STRICT (Progress • Cache • Merged outputs)")

# ----------------------------- Utilities ------------------------------------
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
    cand = [c for c in ["timestamp","TIMESTAMP","TIMESTAMP_START","TIMESTAMP_END"] if c in d.columns]
    if cand:
        d["DateTime"] = pd.to_datetime(d[cand[0]], errors="coerce")
        return d
    if "date" in d.columns and "time" in d.columns:
        d["DateTime"] = pd.to_datetime(d["date"].astype(str) + " " + d["time"].astype(str), errors="coerce")
        return d
    d["DateTime"] = pd.to_datetime(d.iloc[:,0], errors="coerce")
    return d

def regrid_5min_continuous(df: pd.DataFrame) -> pd.DataFrame:
    d = ensure_datetime(df).dropna(subset=["DateTime"]).sort_values("DateTime")
    if d.empty:
        return d
    t0, t1 = d["DateTime"].min(), d["DateTime"].max()
    grid = pd.DataFrame({"DateTime": pd.date_range(t0, t1, freq="5min")})
    out = pd.merge_asof(grid, d.sort_values("DateTime"), on="DateTime", direction="nearest",
                        tolerance=pd.Timedelta("2min"))
    return out

def coerce_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def choose_col(d: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in d.columns:
            return c
    return None

# --------------------------- Footprint models --------------------------------
def _gaussian_kernel(grid_size: float, dx: float, x_peak: float = 80.0,
                     sx0: float = 25.0, kx: float = 0.12, sy: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = float(grid_size)
    dx = float(dx)
    xs = np.arange(-L, L + dx, dx)
    ys = np.arange(-L, L + dx, dx)
    X, Y = np.meshgrid(xs, ys)
    sx = sx0 + kx * np.maximum(0.0, X - x_peak)
    sx = np.maximum(sx, sx0)
    Gx = np.exp(-0.5 * ((X - x_peak) / sx)**2)
    Gy = np.exp(-0.5 * (Y / sy)**2)
    W = Gx * Gy
    W[W < 1e-16] = 0.0
    W = W / (W.sum() + 1e-16)
    return X, Y, W

def rotate_to_EN(X: np.ndarray, Y: np.ndarray, wind_dir_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(270.0 - float(wind_dir_deg))
    Xe = X * np.cos(theta) - Y * np.sin(theta)
    Yn = X * np.sin(theta) + Y * np.cos(theta)
    return Xe, Yn

# In-session cache for repeated footprints (binned WD + rounded drivers)
_foot_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

def compute_footprint_grid(zm: float, z0: float, L: float, ustar: float, Umean: float,
                           wind_dir_deg: float, grid_size: float, dx: float,
                           backend: str = "Gaussian", wd_bin: int = 0, round_L: int = 0,
                           round_ustar: int = 2, round_ws: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wd_key = int(round(wind_dir_deg / max(wd_bin,1))) * max(wd_bin,1) if wd_bin > 0 else float(wind_dir_deg)
    key = (backend, round(zm,1), round(z0,3), round(L, round_L), round(ustar, round_ustar),
           round(Umean, round_ws), wd_key, round(grid_size,1), round(dx,1))
    if key in _foot_cache:
        return _foot_cache[key]

    if backend == "Kljun" and HAVE_PYFFP:
        try:
            xs = np.arange(-grid_size, grid_size + dx, dx)
            ys = np.arange(-grid_size, grid_size + dx, dx)
            X, Y = np.meshgrid(xs, ys)
            f = _pyffp.FFP(zm=zm, z0=z0, ustar=ustar, wind_dir=float(wind_dir_deg),
                           sigmav=None, ol=L if np.isfinite(L) else None,
                           umean=Umean if np.isfinite(Umean) else None,
                           x=xs, y=ys)
            W = None
            for k in ("f","fclim","z"):
                if hasattr(f, k):
                    W = getattr(f, k); break
                if isinstance(f, dict) and k in f:
                    W = f[k]; break
            if W is None and hasattr(f, "calc_footprint"):
                W = f.calc_footprint()
            if W is None:
                raise RuntimeError("pyffp returned no grid")
            W = np.array(W, dtype=float); W[~np.isfinite(W)] = 0.0
            if W.sum() <= 0: raise RuntimeError("pyffp sum=0")
            W = W / W.sum()
            Xe, Yn = rotate_to_EN(X, Y, wind_dir_deg)
            _foot_cache[key] = (Xe, Yn, W)
            return Xe, Yn, W
        except Exception as e:
            st.warning(f"Kljun (pyffp) failed: {e}. Falling back to Gaussian.")
    X, Y, W = _gaussian_kernel(grid_size=grid_size, dx=dx)
    Xe, Yn = rotate_to_EN(X, Y, wind_dir_deg)
    _foot_cache[key] = (Xe, Yn, W)
    return Xe, Yn, W

# ------------------------- Raster sampling -----------------------------------
@dataclass
class RasterMeta:
    crs: str
    transform: any
    nodata: Optional[float]

def load_raster(path: str):
    if not HAVE_RAST or not HAVE_PYPROJ:
        raise RuntimeError("rasterio/pyproj missing. Install: pip install rasterio pyproj")
    ds = rasterio.open(path)
    meta = RasterMeta(crs=str(ds.crs), transform=ds.transform, nodata=ds.nodata)
    arr = ds.read(1)
    return ds, arr, meta

def EN_to_rowcol(ds, x_abs: np.ndarray, y_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = rasterio.transform.rowcol(ds.transform, x_abs, y_abs, op=round)
    return np.array(rows), np.array(cols)

def sample_classes_under_footprint(ds, arr: np.ndarray, meta: RasterMeta,
                                   tower_lon: float, tower_lat: float,
                                   Xe: np.ndarray, Yn: np.ndarray, nodata_override: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", meta.crs, always_xy=True)
    tx, ty = transformer.transform(float(tower_lon), float(tower_lat))

    x_abs = tx + Xe
    y_abs = ty + Yn
    x_flat = x_abs.ravel(); y_flat = y_abs.ravel()
    rows, cols = EN_to_rowcol(ds, x_flat, y_flat)
    H, W = arr.shape
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    vals = arr[rows[ok], cols[ok]].astype(float)
    return vals, ok

def aggregate_class_fractions(class_vals: np.ndarray, weights: np.ndarray,
                              ok_mask: np.ndarray, nodata: Optional[float]) -> Dict[int, float]:
    w_flat = weights.ravel()[ok_mask]
    cls = class_vals
    good = np.isfinite(cls) & np.isfinite(w_flat)
    if nodata is not None and np.isfinite(nodata):
        good &= (cls != float(nodata))
    cls = cls[good]; w_flat = w_flat[good]
    if w_flat.size == 0 or w_flat.sum() <= 0:
        return {}
    out: Dict[int, float] = {}
    for c, w in zip(cls.astype(int), w_flat):
        out[c] = out.get(c, 0.0) + float(w)
    tot = sum(out.values())
    if tot > 0:
        for k in list(out.keys()):
            out[k] = out[k] / tot
    return out

# ---------------------- UI: Inputs & Parameters ------------------------------
st.header("1) Inputs")
c1, c2 = st.columns(2)
with c1:
    csv_file = st.file_uploader("merged_allvars_5min CSV", type=["csv"])
    csv_path = st.text_input("…or local path to merged_allvars_5min.csv", value="")
with c2:
    geotiff = st.file_uploader("GeoTIFF (land-cover classes)", type=["tif","tiff"])
    geotiff_path = st.text_input("…or local path to GeoTIFF", value="")

legend_file = st.file_uploader("Legend CSV (class_id,label or class_id,class_name)", type=["csv"])
legend_path = st.text_input("…or local path to legend CSV", value="")

st.subheader("Footprint & Speed Settings")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    backend = st.selectbox("Model", ["Gaussian (fallback)", "Kljun (pyffp)"], index=(1 if HAVE_PYFFP else 0))
    grid_size = st.number_input("Half grid size (m)", min_value=50, max_value=2000, value=250, step=25)
    dx = st.number_input("Grid resolution (m)", min_value=1, max_value=50, value=5, step=1)
with cc2:
    zm = st.number_input("Measurement height zm (m)", min_value=1.0, max_value=200.0, value=30.0, step=0.5)
    z0 = st.number_input("Roughness length z0 (m)", min_value=0.001, max_value=2.0, value=0.15, step=0.01)
    nodata_text = st.text_input("Raster NoData (blank=use internal)", value="")
with cc3:
    tower_lat = st.number_input("Tower latitude (WGS84)", value=49.166000, format="%.6f")
    tower_lon = st.number_input("Tower longitude (WGS84)", value=20.283000, format="%.6f")
    wd_bin = st.number_input("WD cache bin (deg, 0=off)", min_value=0, max_value=60, value=10, step=5,
                             help="Cache footprints by rounding wind direction to this bin. Speeds up runs.")
    wthr = st.number_input("Min required drivers fraction", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

st.subheader("Flux & STRICT settings")
cfx1, cfx2, cfx3 = st.columns(3)
with cfx1:
    strict_thr = st.slider("STRICT threshold (fraction)", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    top1_fallback = st.checkbox("If no class ≥ threshold: assign Top‑1 anyway", value=False)
with cfx2:
    strict_mode = st.selectbox("STRICT mode", ["Top‑1 class per timestamp", "Multiple classes ≥ threshold"], index=0)
with cfx3:
    unassigned_as_zero = st.checkbox("Unassigned → 0 (else NA)", value=True)

# Resolve paths
def _resolve_csv():
    if csv_file is not None:
        return csv_file
    if csv_path.strip() and os.path.exists(csv_path.strip()):
        return csv_path.strip()
    return None

def _resolve_tif():
    if geotiff is not None:
        tmpdir = st.session_state.get("tmpdir") or tempfile.mkdtemp()
        st.session_state["tmpdir"] = tmpdir
        pth = os.path.join(tmpdir, geotiff.name)
        with open(pth, "wb") as f: f.write(geotiff.getbuffer())
        return pth
    if geotiff_path.strip() and os.path.exists(geotiff_path.strip()):
        return geotiff_path.strip()
    return None

def _resolve_legend():
    if legend_file is not None:
        return legend_file
    if legend_path.strip() and os.path.exists(legend_path.strip()):
        return legend_path.strip()
    return None

csv_src = _resolve_csv()
tif_src = _resolve_tif()
leg_src = _resolve_legend()

# ---------------------- Load merged CSV & detect columns ---------------------
st.header("2) Load merged ALL‑VARS and detect columns")
df_in = None
if csv_src is not None:
    try:
        df_in = pd.read_csv(csv_src, low_memory=False)
        df_in.columns = _dedup(list(df_in.columns))
        df_in = ensure_datetime(df_in)
        df_in = df_in.sort_values("DateTime")
        df_in = regrid_5min_continuous(df_in)  # handles time gaps
        st.success(f"Loaded input: {len(df_in):,} rows. Reindexed to continuous 5‑min grid.")
        st.dataframe(df_in.head(12), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read merged CSV: {e}")

flux_candidates = ["FO_co2_flux","NEE","FO_ch4_flux","FO_h2o_flux","FO_LE","FO_H","co2_flux","ch4_flux","h2o_flux","LE","H"]
ustar_candidates = ["FO_u*","FO_ustar","u*","ustar"]
ws_candidates    = ["FO_wind_speed","wind_speed"]
wd_candidates    = ["FO_wind_dir","wind_dir","WD"]
L_candidates     = ["FO_L","L"]

flux_col = ustar_col = ws_col = wd_col = L_col = None
if df_in is not None:
    flux_col  = choose_col(df_in, flux_candidates)
    ustar_col = choose_col(df_in, ustar_candidates)
    ws_col    = choose_col(df_in, ws_candidates)
    wd_col    = choose_col(df_in, wd_candidates)
    L_col     = choose_col(df_in, L_candidates)

csel1, csel2, csel3 = st.columns(3)
with csel1:
    flux_col = st.selectbox("Flux column to allocate", options=[c for c in [flux_col] + (df_in.columns.to_list() if df_in is not None else [])], index=0 if df_in is not None else 0)
with csel2:
    ustar_col = st.selectbox("u* column", options=[c for c in [ustar_col] + (df_in.columns.to_list() if df_in is not None else [])], index=0 if df_in is not None else 0)
with csel3:
    ws_col = st.selectbox("Wind speed (WS) column", options=[c for c in [ws_col] + (df_in.columns.to_list() if df_in is not None else [])], index=0 if df_in is not None else 0)

csel4, csel5 = st.columns(2)
with csel4:
    wd_col = st.selectbox("Wind direction (WD) column", options=[c for c in [wd_col] + (df_in.columns.to_list() if df_in is not None else [])], index=0 if df_in is not None else 0)
with csel5:
    L_col = st.selectbox("Obukhov length (L) column", options=[c for c in [L_col] + (df_in.columns.to_list() if df_in is not None else [])], index=0 if df_in is not None else 0)

# ---------------------- Load raster + legend --------------------------------
class_labels: Dict[int,str] = {}
if leg_src is not None:
    try:
        ldf = pd.read_csv(leg_src)
        if "class_id" not in ldf.columns:
            st.warning("Legend CSV needs 'class_id' column."); 
        else:
            label_col = "label" if "label" in ldf.columns else ("class_name" if "class_name" in ldf.columns else None)
            if label_col is None:
                st.warning("Legend CSV needs 'label' or 'class_name'.")
            else:
                for _, rr in ldf.iterrows():
                    class_labels[int(rr["class_id"])] = str(rr[label_col])
                st.success(f"Loaded legend with {len(class_labels)} classes.")
    except Exception as e:
        st.error(f"Failed to read legend CSV: {e}")

r_ds = r_arr = r_meta = None
if tif_src is not None:
    try:
        r_ds, r_arr, r_meta = load_raster(tif_src)
        st.success(f"Loaded GeoTIFF {tif_src}. Shape: {r_arr.shape}, CRS: {r_meta.crs}, NoData: {r_meta.nodata}")
    except Exception as e:
        st.error(f"Failed to open GeoTIFF: {e}")

# -------------------------- Allocation engine --------------------------------
st.header("3) Run allocation")
subset_opts = []
if df_in is not None and not df_in.empty:
    subset_opts = df_in["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

sel_ts = st.selectbox("Pick one timestamp to preview", options=subset_opts, index=0 if subset_opts else 0)
sel_all = st.checkbox("Process ALL timestamps", value=False)
max_rows = st.number_input("Max timestamps to process (if not ALL)", min_value=1, max_value=100000, value=500, step=50)

run = st.button("▶️ Compute fractions and allocate flux")

def fractions_for_rows(rows: pd.DataFrame, progress=None, status=None) -> pd.DataFrame:
    results = []
    if r_ds is None:
        st.error("GeoTIFF not loaded."); return pd.DataFrame()
    try:
        nodata_val = float(nodata_text) if nodata_text.strip() != "" else (r_meta.nodata if r_meta else None)
    except Exception:
        nodata_val = r_meta.nodata if r_meta else None

    total = len(rows)
    if progress is None:
        progress = st.progress(0)
    if status is None:
        status = st.empty()

    for idx, (_, rr) in enumerate(rows.iterrows(), start=1):
        pct = int(idx * 100 / total) if total else 100
        progress.progress(min(pct, 100))
        status.write(f"Computing footprint {idx}/{total} …")

        # Required drivers
        L_val     = coerce_numeric(rr.get(L_col, np.nan))
        ustar_val = coerce_numeric(rr.get(ustar_col, np.nan))
        ws_val    = coerce_numeric(rr.get(ws_col, np.nan))
        wd_val    = coerce_numeric(rr.get(wd_col, np.nan))
        if not (np.isfinite(L_val) and np.isfinite(ustar_val) and np.isfinite(ws_val) and np.isfinite(wd_val)):
            continue

        Xe, Yn, W = compute_footprint_grid(zm=float(zm), z0=float(z0), L=float(L_val),
                                           ustar=float(ustar_val), Umean=float(ws_val),
                                           wind_dir_deg=float(wd_val),
                                           grid_size=float(grid_size), dx=float(dx),
                                           backend=("Kljun" if backend.startswith("Kljun") else "Gaussian"),
                                           wd_bin=int(wd_bin))
        class_vals, ok_mask = sample_classes_under_footprint(r_ds, r_arr, r_meta,
                                                             float(tower_lon), float(tower_lat),
                                                             Xe, Yn, nodata_val)
        fracs = aggregate_class_fractions(class_vals, W, ok_mask, nodata=nodata_val)
        if not fracs:
            continue
        for cid, frac in fracs.items():
            results.append({
                "DateTime": pd.to_datetime(rr["DateTime"]),
                "class_id": int(cid),
                "fraction": float(frac),
                "label": class_labels.get(int(cid), "")
            })
    if status is not None:
        status.write("Done.")
    if progress is not None:
        progress.progress(100)
    return pd.DataFrame(results)

def fractions_to_wide(df_frac: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    if df_frac is None or df_frac.empty:
        return pd.DataFrame(), []
    d = df_frac.copy()
    d["class_id"] = d["class_id"].astype(int)
    wide = d.pivot(index="DateTime", columns="class_id", values="fraction").sort_index(axis=1)
    wide = wide.reset_index()
    cls_ids = [c for c in wide.columns if isinstance(c, (int, np.integer))]
    if cls_ids:
        vals = wide[cls_ids].values
        s = np.nansum(vals, axis=1, keepdims=True)
        s[s==0] = np.nan
        wide[cls_ids] = vals / s
    return wide, cls_ids

def allocate_raw(wide_frac: pd.DataFrame, cls_ids: List[int], base: pd.DataFrame, flux_col: str) -> pd.DataFrame:
    base2 = base[["DateTime", flux_col]].copy()
    out = pd.merge(base2, wide_frac, on="DateTime", how="left")
    for cid in cls_ids:
        cname = f"{flux_col}_RAW_class_{cid}"
        out[cname] = out[flux_col] * out.get(cid, np.nan)
    return out

def allocate_strict(wide_frac: pd.DataFrame, cls_ids: List[int], base: pd.DataFrame, flux_col: str,
                    thr: float, mode: str, unassigned_zero: bool, top1_fallback: bool) -> pd.DataFrame:
    base2 = base[["DateTime", flux_col]].copy()
    W = pd.merge(base2, wide_frac, on="DateTime", how="left")
    for cid in cls_ids:
        W[f"{flux_col}_STRICT_class_{cid}"] = np.nan
    best_id, best_frac = [], []
    for i, row in W.iterrows():
        frs = {cid: row.get(cid, np.nan) for cid in cls_ids}
        valid = [(cid, float(v)) for cid, v in frs.items() if pd.notna(v)]
        if not valid:
            best_id.append(np.nan); best_frac.append(np.nan); continue
        cid_top, frac_top = max(valid, key=lambda t: t[1])
        best_id.append(cid_top); best_frac.append(frac_top)
        assigned = False
        if mode.startswith("Top"):
            if pd.notna(frac_top) and frac_top >= thr:
                W.at[i, f"{flux_col}_STRICT_class_{cid_top}"] = W.at[i, flux_col]
                assigned = True
            elif top1_fallback and pd.notna(frac_top):
                W.at[i, f"{flux_col}_STRICT_class_{cid_top}"] = W.at[i, flux_col]
                assigned = True
        else:
            for cid, v in valid:
                if pd.notna(v) and v >= thr:
                    W.at[i, f"{flux_col}_STRICT_class_{cid}"] = W.at[i, flux_col]
                    assigned = True
            if (not assigned) and top1_fallback and pd.notna(frac_top):
                W.at[i, f"{flux_col}_STRICT_class_{cid_top}"] = W.at[i, flux_col]
                assigned = True
    W["best_class_id"] = best_id
    W["best_fraction"] = best_frac
    if unassigned_zero:
        for cid in cls_ids:
            cname = f"{flux_col}_STRICT_class_{cid}"
            W[cname] = W[cname].fillna(0.0)
    return W

df_frac = None
if run and df_in is not None and r_ds is not None:
    full = df_in.copy()
    # Determine subset to process
    if sel_all:
        rows = full
    else:
        if sel_ts:
            t0 = pd.to_datetime(sel_ts)
            rows = full[full["DateTime"] >= t0].head(int(max_rows))
        else:
            rows = full.head(int(max_rows))
    if all([c is not None for c in [ustar_col, ws_col, wd_col, L_col]]):
        req = full[[ustar_col, ws_col, wd_col, L_col]].notna().sum(axis=1) / 4.0
        rows = rows[req.loc[rows.index] >= float(wthr)]
    else:
        st.error("Please select driver columns (u*, WS, WD, L)."); rows = rows.iloc[0:0]
    st.write(f"Allocating for {len(rows)} timestamps…")

    # Prepare progress bar + status line
    pbar = st.progress(0); stat = st.empty()

    df_frac = fractions_for_rows(rows, progress=pbar, status=stat)
    if df_frac is None or df_frac.empty:
        st.warning("No fractions computed (missing drivers or footprint outside raster?).")
    else:
        st.success(f"Computed fractions for {df_frac['DateTime'].nunique()} timestamps.")
        st.subheader("Fractions (long) — preview")
        st.dataframe(df_frac.head(20), use_container_width=True)
        st.download_button("⬇️ footprint_fractions_LONG.csv",
                           df_frac.to_csv(index=False).encode("utf-8"),
                           "footprint_fractions_LONG.csv", "text/csv")
        wide, cls_ids = fractions_to_wide(df_frac)
        if wide.empty:
            st.warning("Empty wide fractions after pivot.")
        else:
            # Prefix fraction columns for clarity before merging
            frac_cols_map = {cid: f"frac_class_{cid}" for cid in cls_ids}
            wide_pref = wide.rename(columns=frac_cols_map)

            st.subheader("Fractions (wide) — preview")
            st.dataframe(wide_pref.head(20), use_container_width=True)
            st.download_button("⬇️ footprint_fractions_WIDE.csv",
                               wide_pref.to_csv(index=False).encode("utf-8"),
                               "footprint_fractions_WIDE.csv", "text/csv")

            if flux_col is None:
                st.error("Select a flux column first.")
            else:
                base = df_in.copy()
                base[flux_col] = coerce_numeric(base[flux_col])

                # RAW allocation
                raw = allocate_raw(wide, cls_ids, base, flux_col)
                # STRICT allocation
                strict = allocate_strict(wide, cls_ids, base, flux_col,
                                         thr=float(strict_thr), mode=strict_mode,
                                         unassigned_zero=bool(unassigned_as_zero),
                                         top1_fallback=bool(top1_fallback))

                # Build COMBINED table (RAW + STRICT side-by-side)
                raw_cols = [c for c in raw.columns if isinstance(c, str) and c.startswith(f"{flux_col}_RAW_class_")]
                strict_cols = [c for c in strict.columns if isinstance(c, str) and c.startswith(f"{flux_col}_STRICT_class_")]
                combined = (raw[["DateTime"] + raw_cols]
                            .merge(strict[["DateTime"] + strict_cols + ["best_class_id","best_fraction"]],
                                   on="DateTime", how="outer")
                            .sort_values("DateTime"))

                st.subheader("COMBINED allocation (RAW + STRICT) — preview")
                st.dataframe(combined.head(20), use_container_width=True)

                # ------------------ New: MERGE outputs back into input ------------------
                st.header("4) Merged outputs with original input")

                merged_frac_in   = df_in.merge(wide_pref, on="DateTime", how="left")
                merged_raw_in    = df_in.merge(raw[["DateTime"] + raw_cols], on="DateTime", how="left")
                merged_strict_in = df_in.merge(strict[["DateTime"] + strict_cols + ["best_class_id","best_fraction"]], on="DateTime", how="left")
                merged_comb_in   = df_in.merge(combined, on="DateTime", how="left")

                st.markdown("**Merged with fractions (wide):**")
                st.dataframe(merged_frac_in.head(12), use_container_width=True)
                st.download_button("⬇️ merged_input_with_fractions.csv",
                                   merged_frac_in.to_csv(index=False).encode("utf-8"),
                                   "merged_input_with_fractions.csv", "text/csv")

                st.markdown("**Merged with RAW allocation:**")
                st.dataframe(merged_raw_in.head(12), use_container_width=True)
                st.download_button("⬇️ merged_input_with_RAW.csv",
                                   merged_raw_in.to_csv(index=False).encode("utf-8"),
                                   "merged_input_with_RAW.csv", "text/csv")

                st.markdown("**Merged with STRICT allocation:**")
                st.dataframe(merged_strict_in.head(12), use_container_width=True)
                st.download_button("⬇️ merged_input_with_STRICT.csv",
                                   merged_strict_in.to_csv(index=False).encode("utf-8"),
                                   "merged_input_with_STRICT.csv", "text/csv")

                st.markdown("**Merged with COMBINED (RAW + STRICT):**")
                st.dataframe(merged_comb_in.head(12), use_container_width=True)
                st.download_button("⬇️ merged_input_with_COMBINED.csv",
                                   merged_comb_in.to_csv(index=False).encode("utf-8"),
                                   "merged_input_with_COMBINED.csv", "text/csv")

                # Also still offer the separate allocation tables (un-merged) for flexibility
                st.header("Optional: separate allocation tables (un‑merged)")
                st.download_button("⬇️ allocated_flux_RAW_5min.csv",
                                   raw.to_csv(index=False).encode("utf-8"),
                                   "allocated_flux_RAW_5min.csv", "text/csv")
                st.download_button("⬇️ allocated_flux_STRICT_5min.csv",
                                   strict.to_csv(index=False).encode("utf-8"),
                                   "allocated_flux_STRICT_5min.csv", "text/csv")
                st.download_button("⬇️ allocated_flux_COMBINED_5min.csv",
                                   combined.to_csv(index=False).encode("utf-8"),
                                   "allocated_flux_COMBINED_5min.csv", "text/csv")
