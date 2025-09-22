
# polygon_to_geotiff_CLASSFIX_v3.py
# Streamlit app: draw/import polygons and rasterize to GeoTIFF.
# v3: Persist polygons + per‑polygon class editor + **placemark on map center**

import os, io, json, zipfile
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Optional libraries
try:
    import geopandas as gpd
    from shapely.geometry import shape, Polygon, MultiPolygon
except Exception:
    gpd = None

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.io import MemoryFile
except Exception:
    rasterio = None

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except Exception:
    folium = None

try:
    from fastkml import kml as _fastkml
except Exception:
    _fastkml = None

st.set_page_config(page_title="Polygon → GeoTIFF (persist + class editor + center marker)", layout="wide")
st.title("Polygon → GeoTIFF — persistent polygons + per‑polygon class editor")
st.caption("v3: Adds a placemark at the **map center** (red marker).")

# -------------------- helpers --------------------
def _ensure_gpd():
    if gpd is None:
        st.error("geopandas/shapely not installed. Run: pip install geopandas shapely")
        st.stop()

def gdf_from_geojson_dict(gj: dict):
    _ensure_gpd()
    feats = gj.get("features", [])
    geoms, names = [], []
    for f in feats:
        try:
            geom = shape(f.get("geometry"))
            if geom.geom_type not in ("Polygon","MultiPolygon"):
                continue
            nm = f.get("properties", {}).get("name", "")
            geoms.append(geom)
            names.append(nm)
        except Exception:
            continue
    if not geoms:
        return gpd.GeoDataFrame(columns=["name","geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame({"name": names, "geometry": geoms}, crs="EPSG:4326")

def gdf_from_kml_bytes(data: bytes):
    _ensure_gpd()
    # Try geopandas driver first
    try:
        bio = io.BytesIO(data)
        gdf = gpd.read_file(bio, driver="KML")
        gdf = gdf[gdf.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
        if not gdf.empty:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            return gdf
    except Exception:
        pass
    # Fallback: fastkml
    if _fastkml is None:
        raise RuntimeError("KML read failed and fastkml is not installed")
    txt = data.decode("utf-8", errors="ignore")
    k = _fastkml.KML(); k.from_string(txt)
    polys, names = [], []
    stack = list(k.features())
    while stack:
        f = stack.pop()
        geom = getattr(f, "geometry", None)
        if geom is not None:
            try:
                shp = shape(geom.__geo_interface__)
                if shp.geom_type in ("Polygon","MultiPolygon"):
                    polys.append(shp)
                    names.append(getattr(f, "name", ""))
            except Exception:
                pass
        if hasattr(f, "features"):
            stack.extend(list(f.features()))
    if not polys:
        raise RuntimeError("KML contains no polygons")
    return gpd.GeoDataFrame({"name": names, "geometry": polys}, crs="EPSG:4326")

def ensure_singleparts(gdf):
    rows = []
    for _, r in gdf.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Polygon":
            rows.append(r)
        elif g.geom_type == "MultiPolygon":
            for p in g.geoms:
                rr = r.copy(); rr.geometry = p; rows.append(rr)
    if not rows:
        return gpd.GeoDataFrame(columns=list(gdf.columns), crs=gdf.crs)
    return gpd.GeoDataFrame(rows, crs=gdf.crs)

def rasterize_polygons(
    gdf_in,
    epsg_out: str,
    pixel_size: float,
    nodata: int,
    class_vals: List[int],
    template_tif: Optional[bytes] = None,
    all_touched: bool = True,
) -> Tuple[bytes, dict, "gpd.GeoDataFrame"]:
    _ensure_gpd()
    if rasterio is None:
        raise RuntimeError("rasterio is not installed. Run: pip install rasterio")
    if gdf_in.empty:
        raise ValueError("No polygons to rasterize")
    gdf = ensure_singleparts(gdf_in)

    vals = [int(v) for v in class_vals]
    if template_tif is not None:
        with rasterio.io.MemoryFile(template_tif) as mf:
            with mf.open() as src:
                crs = src.crs; transform = src.transform
                width, height = src.width, src.height
                g = gdf.to_crs(crs)
                shapes = [(geom, int(v)) for geom, v in zip(g.geometry, vals)]
                out = rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                                fill=nodata, dtype="uint16", all_touched=all_touched)
                profile = src.profile.copy()
                profile.update({"count":1, "dtype":"uint16", "nodata":nodata, "compress":"lzw"})
                with MemoryFile() as mem:
                    with mem.open(**profile) as dst:
                        dst.write(out, 1)
                    return mem.read(), profile, g

    # No template: compute grid from bounds
    g = gdf.to_crs(epsg_out)
    minx, miny, maxx, maxy = g.total_bounds
    if not all(map(np.isfinite, [minx,miny,maxx,maxy])) or (maxx <= minx) or (maxy <= miny):
        raise ValueError("Invalid bounds")
    width  = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    shapes = [(geom, int(v)) for geom, v in zip(g.geometry, vals)]
    out = rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                    fill=nodata, dtype="uint16", all_touched=all_touched)
    profile = {"driver":"GTiff","height":height,"width":width,"count":1,"dtype":"uint16",
               "crs":g.crs,"transform":transform,"nodata":nodata,"compress":"lzw"}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(out, 1)
        return mem.read(), profile, g


# -------------------- session state --------------------
if "poly_store" not in st.session_state:
    st.session_state["poly_store"] = None  # GeoDataFrame

def _append_to_store(gdf_new: "gpd.GeoDataFrame", reset: bool = False):
    _ensure_gpd()
    if gdf_new is None or gdf_new.empty:
        return
    g = gdf_new.copy()
    if "class_id" not in g.columns: g["class_id"] = 1
    if "label" not in g.columns:    g["label"] = g["name"] if "name" in g.columns else ""
    if reset or (st.session_state["poly_store"] is None) or st.session_state["poly_store"].empty:
        st.session_state["poly_store"] = g.set_crs("EPSG:4326", allow_override=True)
    else:
        S = st.session_state["poly_store"]
        S = pd.concat([S, g], ignore_index=True)
        try:
            S["_wkt"] = S.geometry.apply(lambda x: x.wkt)
            S = S.drop_duplicates(subset=["_wkt"]).drop(columns=["_wkt"])
        except Exception:
            pass
        st.session_state["poly_store"] = S.set_crs("EPSG:4326", allow_override=True)

def _clear_store():
    """Reset the polygon store to an empty GeoDataFrame when possible."""
    if gpd is None:
        st.session_state["poly_store"] = None
        return

    st.session_state["poly_store"] = gpd.GeoDataFrame(
        columns=["name", "label", "class_id", "geometry"],
        crs="EPSG:4326",
    )


# -------------------- UI: draw / import --------------------
tab_draw, tab_kml = st.tabs(["Draw on map", "Import KML/GeoJSON"])

with tab_draw:
    st.caption("Draw polygon(s), then click **Add drawn features** to persist. Saved polygons remain across reruns.")
    if folium is None:
        st.info("Install folium + streamlit-folium to enable drawing.")
    else:
        # Map center + zoom
        c_left, c_right = st.columns(2)
        with c_left:
            center_lat = st.number_input("Map center lat", value=49.166000, format="%.6f")
        with c_right:
            center_lon = st.number_input("Map center lon", value=20.283000, format="%.6f")
        zoom = st.slider("Zoom", 4, 19, 14)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=int(zoom), control_scale=True)

        # --- NEW: Placemark at map center ---
        folium.Marker(
            location=[center_lat, center_lon],
            tooltip="Map center",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        # Show already-saved polygons
        if gpd is not None and isinstance(st.session_state.get("poly_store"), gpd.GeoDataFrame):
            S = st.session_state["poly_store"]
            if S is not None and not S.empty:
                try:
                    gj = json.loads(S.to_json())
                    folium.GeoJson(gj, name="Saved polygons",
                                   tooltip=folium.GeoJsonTooltip(fields=["name","label","class_id"],
                                                                 aliases=["name","label","class_id"])).add_to(m)
                except Exception:
                    pass

        Draw(draw_options={"polyline":False,"circle":False,"circlemarker":False,"marker":False,
                           "rectangle":True,"polygon":True},
             edit_options={"edit":True,"remove":True}).add_to(m)
        out = st_folium(m, width=None, height=550, returned_objects=["all_drawings"])

        c1, c2 = st.columns(2)
        with c1:
            add_btn = st.button("➕ Add drawn features")
        with c2:
            clear_btn = st.button("🧹 Clear saved polygons")

        if clear_btn:
            _clear_store(); st.success("Cleared saved polygons.")

        if add_btn:
            try:
                if out and isinstance(out, dict) and out.get("all_drawings"):
                    gj = {"type":"FeatureCollection","features":out["all_drawings"]}
                    gdf_new = gdf_from_geojson_dict(gj)
                    if not gdf_new.empty:
                        _append_to_store(gdf_new, reset=False)
                        st.success(f"Added {len(gdf_new)} polygon(s) to store.")
                    else:
                        st.warning("No polygon geometry found to add.")
                else:
                    st.warning("Nothing to add — draw a polygon first.")
            except Exception as e:
                st.error(f"Failed to add drawn features: {e}")

with tab_kml:
    st.caption("Upload KML (Google Earth) or GeoJSON. Choose Append/Replace.")
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Upload KML/GeoJSON", type=["kml","KML","geojson","json"])
    with c2:
        path = st.text_input("…or direct path", value="")
    mode = st.radio("How to load", ["Append to saved polygons","Replace saved polygons"], index=0, horizontal=True)
    load_btn = st.button("📥 Load file")
    if load_btn:
        try:
            data = None; is_geojson = False
            if up is not None:
                data = up.read(); is_geojson = up.name.lower().endswith((".geojson",".json"))
            elif path.strip() and os.path.exists(path.strip()):
                with open(path.strip(),"rb") as f: data = f.read()
                is_geojson = path.lower().endswith((".geojson",".json"))
            if data is None:
                st.warning("Provide a file or valid path."); 
            else:
                if is_geojson:
                    gj = json.loads(data.decode("utf-8", errors="ignore"))
                    gdf_new = gdf_from_geojson_dict(gj)
                else:
                    gdf_new = gdf_from_kml_bytes(data)
                _append_to_store(gdf_new, reset=(mode.startswith("Replace")))
                st.success(f"Loaded {len(gdf_new)} polygon(s).")
        except Exception as e:
            st.error(f"Failed to load: {e}")

# -------------------- Per‑polygon editor --------------------
st.subheader("Per‑polygon class assignment")

S = st.session_state.get("poly_store")
if (S is None) or (S.empty):
    st.info("No saved polygons yet. Draw/import polygons and add them to the list first.")
    st.stop()

if "class_id" not in S.columns: S["class_id"] = 1
if "label" not in S.columns:    S["label"] = S["name"] if "name" in S.columns else ""

tbl_cols = [c for c in ["name","label","class_id"] if c in S.columns]
edit_df = st.data_editor(
    S.drop(columns=["geometry"])[tbl_cols],
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "class_id": st.column_config.NumberColumn("class_id", min_value=0, max_value=65534, step=1),
        "label": st.column_config.TextColumn("label"),
        "name": st.column_config.TextColumn("name (from KML)")
    },
    key="class_editor_v3"
)
try:
    S.loc[:, "class_id"] = pd.to_numeric(edit_df["class_id"], errors="coerce").fillna(0).astype(int)
    if "label" in edit_df.columns:
        S.loc[:, "label"] = edit_df["label"].astype(str)
    st.session_state["poly_store"] = S
except Exception:
    pass

to_del = st.multiselect("Delete rows (by index)", options=list(S.index), default=[])
if st.button("Delete selected"):
    S2 = S.drop(index=to_del).reset_index(drop=True)
    st.session_state["poly_store"] = S2
    st.experimental_rerun()

# -------------------- Output grid / CRS --------------------
st.header("Output grid / CRS")
use_template = st.checkbox("Use template GeoTIFF (inherit CRS/transform/shape)")
epsg_out = st.text_input("Output CRS (EPSG:code)", value="EPSG:32634")
pixel_size = st.number_input("Pixel size (in units of CRS)", min_value=0.1, value=5.0, step=0.5)
nodata = st.number_input("NODATA value", min_value=0, max_value=65535, value=65535, step=1)
all_touched = st.checkbox("all_touched (paint any pixel touched by polygon)", value=True)

templ_bytes = None
if use_template:
    t1, t2 = st.columns(2)
    with t1:
        templ_up = st.file_uploader("Template GeoTIFF", type=["tif","tiff"])
    with t2:
        templ_path = st.text_input("…or path to template GeoTIFF", value="")
    if templ_up is not None:
        templ_bytes = templ_up.read()
    elif templ_path.strip() and os.path.exists(templ_path.strip()):
        with open(templ_path.strip(),"rb") as f:
            templ_bytes = f.read()
    else:
        st.info("Provide a template via upload or path (or uncheck the option).")

if st.button("Rasterize to GeoTIFF"):
    try:
        gdf_polys = st.session_state["poly_store"]
        class_vals = gdf_polys["class_id"].astype(int).tolist()
        tif_bytes, profile, g_proj = rasterize_polygons(
            gdf_polys, epsg_out, float(pixel_size), int(nodata), class_vals,
            template_tif=templ_bytes, all_touched=bool(all_touched)
        )
        st.success("Rasterization complete.")
        st.json({k: str(v) for k,v in profile.items() if k in ["crs","height","width","transform","dtype","nodata"]})
        st.download_button("Download GeoTIFF", data=tif_bytes, file_name="polygons_classes.tif", mime="image/tiff")

        try:
            g_out = gdf_polys.to_crs(profile["crs"])
        except Exception:
            g_out = gdf_polys.copy()
        gj = g_out.to_json()
        st.download_button("Download polygons (GeoJSON)", data=gj.encode("utf-8"),
                           file_name="polygons_output_crs.geojson", mime="application/geo+json")

        legend_rows = [{"class_id": int(cid), "label": str(lbl)} for cid, lbl in zip(gdf_polys["class_id"], gdf_polys["label"])]
        legend_csv = pd.DataFrame(legend_rows).drop_duplicates().to_csv(index=False)
        st.download_button("Download legend.csv", data=legend_csv.encode("utf-8"),
                           file_name="legend.csv", mime="text/csv")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("polygons_classes.tif", tif_bytes)
            z.writestr("polygons_output_crs.geojson", gj)
            z.writestr("legend.csv", legend_csv)
        st.download_button("Download ALL (ZIP)", data=zbuf.getvalue(),
                           file_name="polygons_to_geotiff_bundle.zip", mime="application/zip")

    except Exception as e:
        st.error(f"Rasterization failed: {e}")
