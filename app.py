import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import leafmap.foliumap as leafmap
from shapely.geometry import Polygon, mapping
import json
import math

# ==================================================
# PAGE CONFIG (MOBILE FIRST)
# ==================================================
st.set_page_config(
    page_title="Urban & Environmental Intelligence Dashboard",
    layout="wide"
)

st.title("🌍 Urban & Environmental Intelligence Dashboard")
st.caption(
    "Interactive smart-city analytics using open satellite data "
    "(Vegetation • Buildings • Roads)"
)

# ==================================================
# SIDEBAR – CONTROL PANEL
# ==================================================
with st.sidebar:
    st.header("⚙️ Dashboard Controls")

    CITY_FILES = {
        "Chennai": "geojson/chennai_boundary.geojson",
        "Coimbatore": "geojson/coimbatore_boundary.geojson"
    }

    city = st.selectbox("Select City", list(CITY_FILES.keys()))
    year_before = st.selectbox("Before Year", [2019, 2020, 2021, 2022])
    year_after = st.selectbox("After Year", [2023, 2024, 2025])

    st.subheader("🔬 Analysis Sensitivity")
    ndvi_thresh = st.slider("Vegetation Change Threshold", 0.1, 0.4, 0.2, 0.05)
    ndbi_thresh = st.slider("Urban Growth Threshold", 0.1, 0.4, 0.2, 0.05)

    st.subheader("🗺️ Basemap Style")
    basemap = st.selectbox(
        "Map Base",
        ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"]
    )

    st.subheader("ℹ️ About")
    st.caption(
        "• Data: Sentinel-2 (ESA)\n"
        "• Methods: NDVI, NDBI\n"
        "• Context: ISRO / NRSC practices"
    )

# ==================================================
# LOAD CITY GEOJSON & AUTO BBOX
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
# SATELLITE ANALYSIS ENGINE (ACCURATE)
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
            ).astype("float32")

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
# MASK → GEOJSON (SHADED & LIGHT)
# ==================================================
def mask_to_geojson(mask, bbox, color, step=10):
    features = []
    h, w = mask.shape
    dx = (bbox[2] - bbox[0]) / w
    dy = (bbox[3] - bbox[1]) / h

    for i in range(0, h, step):
        for j in range(0, w, step):
            if mask[i, j]:
                x = bbox[0] + j * dx
                y = bbox[1] + i * dy
                poly = Polygon([
                    (x, y),
                    (x+dx, y),
                    (x+dx, y+dy),
                    (x, y+dy)
                ])
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "style": {
                            "color": color,
                            "fillOpacity": 0.6,
                            "weight": 0
                        }
                    }
                })

    return {"type": "FeatureCollection", "features": features}

# ==================================================
# PIPELINE
# ==================================================
boundary, bbox = load_city(CITY_FILES[city])

with st.spinner("Running automated satellite analysis…"):
    ndvi_change, ndbi_change = compute_change(bbox, year_before, year_after)

veg_loss = ndvi_change < -ndvi_thresh
veg_gain = ndvi_change > ndvi_thresh
urban = ndbi_change > ndbi_thresh

veg_loss_geo = mask_to_geojson(veg_loss, bbox, "#b2182b")
veg_gain_geo = mask_to_geojson(veg_gain, bbox, "#1a9850")
urban_geo = mask_to_geojson(urban, bbox, "#fdae61")

# ==================================================
# INTERACTIVE MAP (CORE DASHBOARD)
# ==================================================
m = leafmap.Map(
    center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2],
    zoom=11,
    tiles=basemap
)

m.add_geojson(boundary, layer_name="Administrative Boundary")
m.add_geojson(veg_loss_geo, layer_name="Vegetation Loss")
m.add_geojson(veg_gain_geo, layer_name="Vegetation Gain")
m.add_geojson(urban_geo, layer_name="New Buildings & Roads")

m.add_layer_control()
m.to_streamlit(height=650)

# ==================================================
# ANALYTICS
# ==================================================
st.subheader("📊 Analytical Insights")

pixel_area_km2 = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) / ndvi_change.size * 12365

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{np.mean(veg_loss)*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{np.mean(veg_gain)*100:.2f}%")
c3.metric("Urban Expansion (%)", f"{np.mean(urban)*100:.2f}%")

st.markdown("""
### 🗺️ Legend
- 🔴 Red shades → Vegetation loss  
- 🟢 Green shades → Vegetation gain  
- 🟠 Orange shades → New buildings & road expansion  

### 📚 Data & Methods
- Sentinel-2 Level-2A (ESA Copernicus)
- NDVI & NDBI indices
- Indian municipal boundaries (Datameet)
- ISRO / NRSC urban remote-sensing methodologies
""")

st.success("✅ Advanced GIS dashboard ready for evaluation")
