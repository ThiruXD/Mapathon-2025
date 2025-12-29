import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import leafmap.foliumap as leafmap
from shapely.geometry import Polygon, mapping
import json

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="wide")
st.title("🌍 Real-Time Urban & Vegetation Change Map (India)")
st.caption("Interactive map with vegetation loss & urban expansion")

# ==================================================
# CITY CONFIG
# ==================================================
CITY_FILES = {
    "Chennai": "geojson/chennai_boundary.geojson",
    "Coimbatore": "geojson/coimbatore_boundary.geojson"
}

# ==================================================
# LOAD CITY & BBOX
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
# SATELLITE ENGINE (STABLE)
# ==================================================
@st.cache_data(show_spinner=False)
def compute_masks(bbox, y1, y2):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    def scene(year):
        s = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 10}}
        )
        return pc.sign(list(s.items())[0])

    b = scene(y1)
    a = scene(y2)

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

    veg_loss = ndvi_change < -0.2
    veg_gain = ndvi_change > 0.2
    urban_growth = ndbi_change > 0.2

    return veg_loss, veg_gain, urban_growth

# ==================================================
# MASK → GEOJSON (LIGHTWEIGHT)
# ==================================================
def mask_to_geojson(mask, bbox, color):
    features = []
    h, w = mask.shape
    dx = (bbox[2] - bbox[0]) / w
    dy = (bbox[3] - bbox[1]) / h

    for i in range(0, h, 10):          # skip for performance
        for j in range(0, w, 10):
            if mask[i, j]:
                x1 = bbox[0] + j * dx
                y1 = bbox[1] + i * dy
                poly = Polygon([
                    (x1, y1),
                    (x1+dx, y1),
                    (x1+dx, y1+dy),
                    (x1, y1+dy)
                ])
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {"color": color}
                })

    return {"type": "FeatureCollection", "features": features}

# ==================================================
# USER INPUT
# ==================================================
city = st.selectbox("Select City", list(CITY_FILES.keys()))
year_before = st.selectbox("Before Year", [2019, 2020, 2021, 2022])
year_after = st.selectbox("After Year", [2023, 2024, 2025])

boundary, bbox = load_city(CITY_FILES[city])

with st.spinner("Building interactive layers…"):
    veg_loss, veg_gain, urban = compute_masks(bbox, year_before, year_after)

veg_loss_geo = mask_to_geojson(veg_loss, bbox, "red")
veg_gain_geo = mask_to_geojson(veg_gain, bbox, "green")
urban_geo = mask_to_geojson(urban, bbox, "orange")

# ==================================================
# INTERACTIVE MAP
# ==================================================
m = leafmap.Map(center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2], zoom=11)

m.add_geojson(boundary, layer_name="Wards")
m.add_geojson(veg_loss_geo, layer_name="Vegetation Loss")
m.add_geojson(veg_gain_geo, layer_name="Vegetation Gain")
m.add_geojson(urban_geo, layer_name="New Buildings & Roads")

m.add_layer_control()
m.to_streamlit(height=600)

# ==================================================
# LEGEND
# ==================================================
st.markdown("""
### 🗺️ Map Legend
- 🟥 Vegetation loss  
- 🟩 Vegetation gain  
- 🟧 New buildings & road expansion  

**Data**: Sentinel-2 | Datameet | ISRO/NRSC methods
""")

st.success("✅ Real-time interactive map generated successfully")

