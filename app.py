import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import leafmap.foliumap as leafmap
from shapely.geometry import shape
import json
import pandas as pd

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Urban & Environmental Intelligence Dashboard",
    layout="wide"
)

st.title("🌍 Urban & Environmental Intelligence Dashboard")
st.caption(
    "Ward-level vegetation loss, urban expansion, rankings, and hotspot detection "
    "using Sentinel-2 satellite imagery"
)

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    st.header("⚙️ Controls")

    CITY_FILES = {
        "Bangalore": "geojson/BBMP.geojson",
        "Chennai": "geojson/chennai_boundary.geojson",
        "Hyderabad": "geojson/hyderabad.geojson",
    }

    city = st.selectbox("City", list(CITY_FILES.keys()))
    year = st.slider("Analysis year (vs previous year)", 2019, 2025, 2024)

    ndvi_thresh = st.slider("Vegetation change threshold (NDVI)", 0.1, 0.4, 0.2, 0.05)
    ndbi_thresh = st.slider("Urban growth threshold (NDBI)", 0.1, 0.4, 0.2, 0.05)

# ==================================================
# LOAD CITY GEOJSON & AUTO BBOX
# ==================================================
@st.cache_data
def load_city(path):
    with open(path) as f:
        data = json.load(f)

    coords = []
    for feat in data["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords += geom["coordinates"][0]
        else:
            coords += geom["coordinates"][0][0]

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    return data, bbox

# ==================================================
# SATELLITE PROCESSING (NDVI / NDBI)
# ==================================================
@st.cache_data(show_spinner=False)
def compute_change(bbox, y1, y2):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    def scene(year):
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
        )
        return pc.sign(list(search.items())[0])

    before = scene(y1)
    after = scene(y2)

    def read(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average,
            ).astype("float32")

    red_b, nir_b = read("B04", before, 4), read("B08", before, 4)
    red_a, nir_a = read("B04", after, 4), read("B08", after, 4)
    swir_b, swir_a = read("B11", before, 8), read("B11", after, 8)

    ndvi_change = (nir_a - red_a) / (nir_a + red_a + 1e-10) - (
        (nir_b - red_b) / (nir_b + red_b + 1e-10)
    )

    ndbi_change = (swir_a - nir_a[:swir_a.shape[0], :swir_a.shape[1]]) / (
        swir_a + nir_a[:swir_a.shape[0], :swir_a.shape[1]] + 1e-10
    ) - (
        (swir_b - nir_b[:swir_b.shape[0], :swir_b.shape[1]]) /
        (swir_b + nir_b[:swir_b.shape[0], :swir_b.shape[1]] + 1e-10)
    )

    return ndvi_change, ndbi_change

# ==================================================
# SAFE PERCENT
# ==================================================
def safe_percent(mask):
    return (np.count_nonzero(mask) / mask.size) * 100 if mask.size else 0.0

# ==================================================
# ROBUST WARD NAME EXTRACTOR
# ==================================================
def get_ward_name(props):
    for key in [
        "ward_name", "WARD_NAME", "wardname", "division",
        "ward_no", "WARD_NO", "wardnumber", "ward_id", "Ward_No"
    ]:
        if key in props and props[key] not in [None, ""]:
            return str(props[key])

    for k, v in props.items():
        if "ward" in k.lower():
            return str(v)

    return "Unknown"

# ==================================================
# WARD ANALYTICS + POPUPS
# ==================================================
def analyze_wards(boundary, ndvi, ndbi, bbox):
    hv, wv = ndvi.shape
    hu, wu = ndbi.shape

    rows = []
    features = []

    for feat in boundary["features"]:
        poly = shape(feat["geometry"])
        minx, miny, maxx, maxy = poly.bounds

        x0v = int((minx - bbox[0]) / (bbox[2] - bbox[0]) * wv)
        x1v = int((maxx - bbox[0]) / (bbox[2] - bbox[0]) * wv)
        y0v = int((miny - bbox[1]) / (bbox[3] - bbox[1]) * hv)
        y1v = int((maxy - bbox[1]) / (bbox[3] - bbox[1]) * hv)

        x0u = int((minx - bbox[0]) / (bbox[2] - bbox[0]) * wu)
        x1u = int((maxx - bbox[0]) / (bbox[2] - bbox[0]) * wu)
        y0u = int((miny - bbox[1]) / (bbox[3] - bbox[1]) * hu)
        y1u = int((maxy - bbox[1]) / (bbox[3] - bbox[1]) * hu)

        zone_ndvi = ndvi[max(0,y0v):min(hv,y1v), max(0,x0v):min(wv,x1v)]
        zone_ndbi = ndbi[max(0,y0u):min(hu,y1u), max(0,x0u):min(wu,x1u)]

        veg_loss = safe_percent(zone_ndvi < -ndvi_thresh)
        urban = safe_percent(zone_ndbi > ndbi_thresh)

        props = feat.get("properties", {})
        ward_name = get_ward_name(props)

        rows.append({
            "Ward": ward_name,
            "Vegetation Loss (%)": veg_loss,
            "Urban Expansion (%)": urban,
        })

        feat["properties"]["popup"] = (
            f"<b>Ward:</b> {ward_name}<br>"
            f"<b>Vegetation Loss:</b> {veg_loss:.2f}%<br>"
            f"<b>Urban Expansion:</b> {urban:.2f}%"
        )

        features.append(feat)

    return pd.DataFrame(rows), {"type": "FeatureCollection", "features": features}

# ==================================================
# RUN PIPELINE
# ==================================================
boundary, bbox = load_city(CITY_FILES[city])
ndvi_change, ndbi_change = compute_change(bbox, year - 1, year)
ward_df, ward_geo = analyze_wards(boundary, ndvi_change, ndbi_change, bbox)

# ==================================================
# MAP (OPENSTREETMAP DEFAULT)
# ==================================================
m = leafmap.Map(
    center=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
    zoom=11,
    tiles="OpenStreetMap.Mapnik",
)

m.add_geojson(ward_geo, layer_name="Wards (Click for details)")
m.add_layer_control()
m.to_streamlit(height=420)

# ==================================================
# KPIs
# ==================================================
st.subheader("📊 City-Level KPIs")

c1, c2 = st.columns(2)
c1.metric("Average Vegetation Loss (%)", f"{ward_df['Vegetation Loss (%)'].mean():.2f}")
c2.metric("Average Urban Expansion (%)", f"{ward_df['Urban Expansion (%)'].mean():.2f}")

# ==================================================
# WARD RANKINGS
# ==================================================
st.subheader("🏆 Ward Rankings")

st.markdown("### 🔴 Top 10 Vegetation Loss Wards")
st.dataframe(ward_df.sort_values("Vegetation Loss (%)", ascending=False).head(10))

st.markdown("### 🏗️ Top 10 Urban Expansion Wards")
st.dataframe(ward_df.sort_values("Urban Expansion (%)", ascending=False).head(10))

# ==================================================
# HOTSPOT DETECTION
# ==================================================
st.subheader("🔥 Hotspot Detection (Top 10%)")

veg_hotspots = ward_df[
    ward_df["Vegetation Loss (%)"] > ward_df["Vegetation Loss (%)"].quantile(0.9)
]
urban_hotspots = ward_df[
    ward_df["Urban Expansion (%)"] > ward_df["Urban Expansion (%)"].quantile(0.9)
]

st.markdown("### Vegetation Loss Hotspots")
st.dataframe(veg_hotspots)

st.markdown("### Urban Expansion Hotspots")
st.dataframe(urban_hotspots)

# ==================================================
# WARD-WISE CHARTS (NEW)
# ==================================================
st.subheader("📈 Ward-wise Charts")

top_veg = ward_df.sort_values("Vegetation Loss (%)", ascending=False).head(10)
top_urban = ward_df.sort_values("Urban Expansion (%)", ascending=False).head(10)

st.markdown("### Vegetation Loss by Ward (Top 10)")
st.bar_chart(top_veg.set_index("Ward")["Vegetation Loss (%)"])

st.markdown("### Urban Expansion by Ward (Top 10)")
st.bar_chart(top_urban.set_index("Ward")["Urban Expansion (%)"])

# ==================================================
# DOWNLOAD
# ==================================================
st.subheader("⬇️ Download")

st.download_button(
    "Download Ward Analytics (CSV)",
    ward_df.to_csv(index=False),
    "ward_analytics.csv",
)

st.download_button(
    "Download Ward GeoJSON",
    json.dumps(ward_geo),
    "ward_analysis.geojson",
    "application/geo+json",
)
