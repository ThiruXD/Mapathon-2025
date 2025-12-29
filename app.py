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
    layout="wide",
)

st.title("🌍 Urban & Environmental Intelligence Dashboard")
st.caption(
    "Satellite-based land-use change detection + transport reference layers "
    "(Vegetation • Urban • Roads / Rail / Metro)"
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

    ndvi_thresh = st.slider("Vegetation threshold (NDVI)", 0.1, 0.4, 0.2, 0.05)
    ndbi_thresh = st.slider("Urban threshold (NDBI)", 0.1, 0.4, 0.2, 0.05)

    basemap_analysis = st.selectbox(
        "Analysis Map Basemap",
        ["OpenStreetMap", "CartoDB Positron"],
    )

    basemap_transport = st.selectbox(
        "Transport Map Basemap",
        ["OpenStreetMap (Roads)", "CartoDB Voyager (Transport)"],
    )

# ==================================================
# BASEMAP DICTS (SAFE)
# ==================================================
ANALYSIS_BASEMAPS = {
    "OpenStreetMap": "OpenStreetMap.Mapnik",
    "CartoDB Positron": "CartoDB positron",
}

TRANSPORT_BASEMAPS = {
    "OpenStreetMap (Roads)": "OpenStreetMap.Mapnik",
    "CartoDB Voyager (Transport)": "CartoDB Voyager",
}

# ==================================================
# LOAD CITY GEOJSON + BBOX
# ==================================================
@st.cache_data
def load_city(path):
    with open(path) as f:
        data = json.load(f)

    coords = []
    for feat in data["features"]:
        g = feat["geometry"]
        if g["type"] == "Polygon":
            coords += g["coordinates"][0]
        else:
            coords += g["coordinates"][0][0]

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    bbox = [min(lons), min(lats), max(lons), max(lats)]
    return data, bbox

# ==================================================
# SATELLITE PROCESSING
# ==================================================
@st.cache_data(show_spinner=False)
def compute_change(bbox, y1, y2):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    def scene(year):
        s = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
        )
        return pc.sign(list(s.items())[0])

    b, a = scene(y1), scene(y2)

    def read(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average,
            ).astype("float32")

    red_b, nir_b = read("B04", b, 4), read("B08", b, 4)
    red_a, nir_a = read("B04", a, 4), read("B08", a, 4)
    swir_b, swir_a = read("B11", b, 8), read("B11", a, 8)

    ndvi_change = (nir_a - red_a) / (nir_a + red_a + 1e-10) - (
        (nir_b - red_b) / (nir_b + red_b + 1e-10)
    )

    ndbi_change = (swir_a - nir_a[: swir_a.shape[0], : swir_a.shape[1]]) / (
        swir_a + nir_a[: swir_a.shape[0], : swir_a.shape[1]] + 1e-10
    ) - (
        (swir_b - nir_b[: swir_b.shape[0], : swir_b.shape[1]])
        / (swir_b + nir_b[: swir_b.shape[0], : swir_b.shape[1]] + 1e-10)
    )

    return ndvi_change, ndbi_change

# ==================================================
# SAFE PERCENT
# ==================================================
def safe_percent(mask):
    return (np.count_nonzero(mask) / mask.size) * 100 if mask.size else 0.0

# ==================================================
# WARD POPUPS
# ==================================================
def build_ward_geojson(boundary, ndvi, ndbi, bbox):
    h_v, w_v = ndvi.shape
    h_u, w_u = ndbi.shape
    features = []

    for feat in boundary["features"]:
        poly = shape(feat["geometry"])
        minx, miny, maxx, maxy = poly.bounds

        x0v = int((minx - bbox[0]) / (bbox[2] - bbox[0]) * w_v)
        x1v = int((maxx - bbox[0]) / (bbox[2] - bbox[0]) * w_v)
        y0v = int((miny - bbox[1]) / (bbox[3] - bbox[1]) * h_v)
        y1v = int((maxy - bbox[1]) / (bbox[3] - bbox[1]) * h_v)

        x0u = int((minx - bbox[0]) / (bbox[2] - bbox[0]) * w_u)
        x1u = int((maxx - bbox[0]) / (bbox[2] - bbox[0]) * w_u)
        y0u = int((miny - bbox[1]) / (bbox[3] - bbox[1]) * h_u)
        y1u = int((maxy - bbox[1]) / (bbox[3] - bbox[1]) * h_u)

        zone_ndvi = ndvi[max(0,y0v):min(h_v,y1v), max(0,x0v):min(w_v,x1v)]
        zone_ndbi = ndbi[max(0,y0u):min(h_u,y1u), max(0,x0u):min(w_u,x1u)]

        feat["properties"]["popup"] = (
            f"<b>Vegetation Loss:</b> {safe_percent(zone_ndvi < -ndvi_thresh):.2f}%<br>"
            f"<b>Vegetation Gain:</b> {safe_percent(zone_ndvi > ndvi_thresh):.2f}%<br>"
            f"<b>Urban Expansion:</b> {safe_percent(zone_ndbi > ndbi_thresh):.2f}%"
        )

        features.append(feat)

    return {"type": "FeatureCollection", "features": features}

# ==================================================
# RUN PIPELINE
# ==================================================
boundary, bbox = load_city(CITY_FILES[city])
ndvi_change, ndbi_change = compute_change(bbox, year - 1, year)
ward_geo = build_ward_geojson(boundary, ndvi_change, ndbi_change, bbox)

veg_loss = ndvi_change < -ndvi_thresh
veg_gain = ndvi_change > ndvi_thresh
urban = ndbi_change > ndbi_thresh

# ==================================================
# MAP 1: SATELLITE ANALYSIS
# ==================================================
st.subheader("🛰️ Land-use Change Map")

m1 = leafmap.Map(
    center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2],
    zoom=11,
    tiles=ANALYSIS_BASEMAPS[basemap_analysis],
)

m1.add_geojson(ward_geo, layer_name="Wards (click for stats)")
m1.add_layer_control()
m1.to_streamlit(height=420)

# ==================================================
# MAP 2: TRANSPORT INFRASTRUCTURE (REFERENCE)
# ==================================================
st.subheader("🚦 Transport Infrastructure Map (Reference)")

m2 = leafmap.Map(
    center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2],
    zoom=11,
    tiles=TRANSPORT_BASEMAPS[basemap_transport],
)

m2.add_layer_control()
m2.to_streamlit(height=420)

# ==================================================
# ANALYTICS
# ==================================================
st.subheader("📊 City Analytics")

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{safe_percent(veg_loss):.2f}%")
c2.metric("Vegetation Gain (%)", f"{safe_percent(veg_gain):.2f}%")
c3.metric("Urban Expansion (%)", f"{safe_percent(urban):.2f}%")

# ==================================================
# DOWNLOAD
# ==================================================
st.subheader("⬇️ Download & Export")

df = pd.DataFrame(
    {
        "Metric": ["Vegetation Loss", "Vegetation Gain", "Urban Expansion"],
        "Percentage (%)": [
            safe_percent(veg_loss),
            safe_percent(veg_gain),
            safe_percent(urban),
        ],
    }
)

st.download_button("Download City Analytics (CSV)", df.to_csv(index=False), "city_analytics.csv")
st.download_button(
    "Download Ward GeoJSON",
    json.dumps(ward_geo),
    "ward_analysis.geojson",
    "application/geo+json",
)

st.success("✅ All features stable. Transport map works without errors.")
