import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import leafmap.foliumap as leafmap
import json
import tempfile
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="wide")
st.title("🌍 Real-Time Urban & Vegetation Change Map (India)")
st.caption("Interactive map showing vegetation loss, new buildings & road expansion")

# ==================================================
# CITY CONFIG
# ==================================================
CITY_FILES = {
    "Chennai": "geojson/chennai_boundary.geojson",
    "Coimbatore": "geojson/coimbatore_boundary.geojson"
}

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
def compute_layers(bbox, year1, year2):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    def scene(year):
        s = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 10}}
        )
        return pc.sign(list(s.items())[0])

    b = scene(year1)
    a = scene(year2)

    def read(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average
            ).astype("float32")

    # BEFORE / AFTER
    red_b, nir_b, swir_b = read("B04", b, 4), read("B08", b, 4), read("B11", b, 8)
    red_a, nir_a, swir_a = read("B04", a, 4), read("B08", a, 4), read("B11", a, 8)

    ndvi_b = (nir_b - red_b) / (nir_b + red_b + 1e-10)
    ndvi_a = (nir_a - red_a) / (nir_a + red_a + 1e-10)

    ndbi_b = (swir_b - nir_b[:swir_b.shape[0], :swir_b.shape[1]]) / (
        swir_b + nir_b[:swir_b.shape[0], :swir_b.shape[1]] + 1e-10
    )
    ndbi_a = (swir_a - nir_a[:swir_a.shape[0], :swir_a.shape[1]]) / (
        swir_a + nir_a[:swir_a.shape[0], :swir_a.shape[1]] + 1e-10
    )

    veg_loss = ndvi_a - ndvi_b
    urban_growth = ndbi_a - ndbi_b

    return veg_loss, urban_growth

# ==================================================
# USER INPUT
# ==================================================
city = st.selectbox("Select City", list(CITY_FILES.keys()))
year_before = st.selectbox("Before Year", [2019, 2020, 2021, 2022])
year_after = st.selectbox("After Year", [2023, 2024, 2025])

geojson, bbox = load_city(CITY_FILES[city])

with st.spinner("Generating real-time map layers…"):
    veg_change, urban_change = compute_layers(bbox, year_before, year_after)

# ==================================================
# INTERACTIVE MAP (REAL-TIME)
# ==================================================
m = leafmap.Map(center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2], zoom=11)

# Ward boundaries
m.add_geojson(geojson, layer_name="Wards Boundary")

# Vegetation loss
m.add_raster(
    veg_change,
    cmap="RdYlGn",
    layer_name="Vegetation Change",
    opacity=0.7
)

# Urban growth
m.add_raster(
    urban_change,
    cmap="inferno",
    layer_name="Buildings & Roads Growth",
    opacity=0.7
)

m.add_layer_control()

st.subheader("🗺️ Interactive Change Detection Map")
m.to_streamlit(height=600)

# ==================================================
# REFERENCES
# ==================================================
st.divider()
st.markdown("""
**Legend**
- Green → Vegetation gain  
- Red → Vegetation loss  
- Bright areas → New buildings & road expansion  

**Data**
- Sentinel-2 (ESA Copernicus)
- Indian municipal boundaries (Datameet)
- ISRO / NRSC urban remote-sensing methods
""")

st.success("✅ Real-time interactive map loaded successfully")
