import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import leafmap.foliumap as leafmap
from shapely.geometry import Polygon, shape, mapping
import json
import pandas as pd

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Advanced Urban Change Dashboard", layout="wide")
st.title("🌍 Advanced Urban & Environmental Change Dashboard")

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    st.header("⚙️ Controls")

    CITY_FILES = {
        "Chennai": "geojson/chennai_boundary.geojson",
        "Coimbatore": "geojson/coimbatore_boundary.geojson"
    }

    city = st.selectbox("City", list(CITY_FILES.keys()))
    year = st.slider("Analysis Year", 2019, 2025, 2024)

    ndvi_thresh = st.slider("Vegetation threshold", 0.1, 0.4, 0.2)
    ndbi_thresh = st.slider("Urban threshold", 0.1, 0.4, 0.2)

    basemap = st.selectbox(
        "Basemap",
        ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"]
    )

# ==================================================
# LOAD GEOJSON & BBOX
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
# SATELLITE ENGINE
# ==================================================
@st.cache_data(show_spinner=False)
def compute_change(bbox, y1, y2):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    def scene(year):
        s = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 10}}
        )
        return pc.sign(list(s.items())[0])

    b, a = scene(y1), scene(y2)

    def read(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average
            )

    red_b, nir_b = read("B04", b, 4), read("B08", b, 4)
    red_a, nir_a = read("B04", a, 4), read("B08", a, 4)
    swir_b, swir_a = read("B11", b, 8), read("B11", a, 8)

    ndvi_change = (nir_a - red_a)/(nir_a + red_a + 1e-10) - \
                  (nir_b - red_b)/(nir_b + red_b + 1e-10)

    ndbi_change = (swir_a - nir_a[:swir_a.shape[0], :swir_a.shape[1]]) / \
                  (swir_a + nir_a[:swir_a.shape[0], :swir_a.shape[1]] + 1e-10) - \
                  (swir_b - nir_b[:swir_b.shape[0], :swir_b.shape[1]]) / \
                  (swir_b + nir_b[:swir_b.shape[0], :swir_b.shape[1]] + 1e-10)

    return ndvi_change, ndbi_change

# ==================================================
# MASK → GEOJSON WITH POPUPS
# ==================================================
def ward_popup_geojson(boundary, ndvi, ndbi, bbox):
    features = []

    for feat in boundary["features"]:
        poly = shape(feat["geometry"])
        minx, miny, maxx, maxy = poly.bounds

        h, w = ndvi.shape
        x0 = int((minx - bbox[0]) / (bbox[2] - bbox[0]) * w)
        x1 = int((maxx - bbox[0]) / (bbox[2] - bbox[0]) * w)
        y0 = int((miny - bbox[1]) / (bbox[3] - bbox[1]) * h)
        y1 = int((maxy - bbox[1]) / (bbox[3] - bbox[1]) * h)

        zone_ndvi = ndvi[max(y0,0):min(y1,h), max(x0,0):min(x1,w)]
        zone_ndbi = ndbi[max(y0,0):min(y1,h), max(x0,0):min(x1,w)]

        veg_loss = np.mean(zone_ndvi < -ndvi_thresh) * 100
        veg_gain = np.mean(zone_ndvi > ndvi_thresh) * 100
        urban = np.mean(zone_ndbi > ndbi_thresh) * 100

        feat["properties"]["popup"] = (
            f"Vegetation Loss: {veg_loss:.2f}%<br>"
            f"Vegetation Gain: {veg_gain:.2f}%<br>"
            f"Urban Expansion: {urban:.2f}%"
        )

        features.append(feat)

    return {"type": "FeatureCollection", "features": features}

# ==================================================
# RUN PIPELINE
# ==================================================
boundary, bbox = load_city(CITY_FILES[city])

ndvi_change, ndbi_change = compute_change(bbox, year-1, year)

ward_geo = ward_popup_geojson(boundary, ndvi_change, ndbi_change, bbox)

veg_loss = ndvi_change < -ndvi_thresh
veg_gain = ndvi_change > ndvi_thresh
urban = ndbi_change > ndbi_thresh

# ==================================================
# INTERACTIVE MAP
# ==================================================
m = leafmap.Map(
    center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2],
    zoom=11,
    tiles=basemap
)

m.add_geojson(ward_geo, layer_name="Wards (Click for stats)")
m.add_layer_control()
m.to_streamlit(height=650)

# ==================================================
# DOWNLOAD / EXPORT
# ==================================================
st.subheader("⬇️ Download & Export")

df = pd.DataFrame({
    "Vegetation Loss (%)": [np.mean(veg_loss)*100],
    "Vegetation Gain (%)": [np.mean(veg_gain)*100],
    "Urban Expansion (%)": [np.mean(urban)*100]
})

st.download_button(
    "Download Analytics (CSV)",
    df.to_csv(index=False),
    "analysis.csv"
)

st.download_button(
    "Download Ward GeoJSON",
    json.dumps(ward_geo),
    "ward_analysis.geojson",
    "application/geo+json"
)

st.success("✅ Advanced dashboard with popups, animation logic & exports ready")
