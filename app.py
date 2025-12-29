import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import leafmap.foliumap as leafmap
from shapely.geometry import shape, Polygon, mapping
import json
import pandas as pd

# ==================================================
# PAGE CONFIG (MOBILE FRIENDLY)
# ==================================================
st.set_page_config(
    page_title="Urban & Environmental Intelligence Dashboard",
    layout="wide"
)

st.title("🌍 Urban & Environmental Intelligence Dashboard")
st.caption(
    "Interactive smart-city GIS dashboard using open satellite data "
    "(Vegetation • Buildings • Roads)"
)

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    st.header("⚙️ Controls")

    CITY_FILES = {
        "Chennai": "geojson/chennai_boundary.geojson",
        "Coimbatore": "geojson/coimbatore_boundary.geojson",
    }

    city = st.selectbox("City", list(CITY_FILES.keys()))

    year = st.slider(
        "Analysis year (compared with previous year)",
        2019, 2025, 2024
    )

    ndvi_thresh = st.slider(
        "Vegetation change threshold (NDVI)",
        0.1, 0.4, 0.2, 0.05
    )

    ndbi_thresh = st.slider(
        "Urban growth threshold (NDBI)",
        0.1, 0.4, 0.2, 0.05
    )

    basemap = st.selectbox(
        "Basemap",
        ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"]
    )

    st.markdown("---")
    st.caption(
        "Data: Sentinel-2 (ESA Copernicus)\n\n"
        "Methods: NDVI, NDBI\n\n"
        "Context: ISRO / NRSC urban remote-sensing practices"
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
# SATELLITE PROCESSING ENGINE (ACCURATE)
# ==================================================
@st.cache_data(show_spinner=False)
def compute_change(bbox, year_before, year_after):
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    def get_scene(year):
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 10}}
        )
        return pc.sign(list(search.items())[0])

    item_b = get_scene(year_before)
    item_a = get_scene(year_after)

    def read_band(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average
            ).astype("float32")

    red_b = read_band("B04", item_b, 4)
    nir_b = read_band("B08", item_b, 4)
    swir_b = read_band("B11", item_b, 8)

    red_a = read_band("B04", item_a, 4)
    nir_a = read_band("B08", item_a, 4)
    swir_a = read_band("B11", item_a, 8)

    ndvi_b = (nir_b - red_b) / (nir_b + red_b + 1e-10)
    ndvi_a = (nir_a - red_a) / (nir_a + red_a + 1e-10)

    ndbi_b = (swir_b - nir_b[:swir_b.shape[0], :swir_b.shape[1]]) / (
        swir_b + nir_b[:swir_b.shape[0], :swir_b.shape[1]] + 1e-10
    )
    ndbi_a = (swir_a - nir_a[:swir_a.shape[0], :swir_a.shape[1]]) / (
        swir_a + nir_a[:swir_a.shape[0], :swir_a.shape[1]] + 1e-10
    )

    return ndvi_a - ndvi_b, ndbi_a - ndbi_b

# ==================================================
# WARD-WISE POPUP GEOJSON
# ==================================================
def build_ward_popup_geojson(boundary, ndvi_change, ndbi_change, bbox):
    features = []
    h, w = ndvi_change.shape

    for feat in boundary["features"]:
        poly = shape(feat["geometry"])
        minx, miny, maxx, maxy = poly.bounds

        x0 = int((minx - bbox[0]) / (bbox[2] - bbox[0]) * w)
        x1 = int((maxx - bbox[0]) / (bbox[2] - bbox[0]) * w)
        y0 = int((miny - bbox[1]) / (bbox[3] - bbox[1]) * h)
        y1 = int((maxy - bbox[1]) / (bbox[3] - bbox[1]) * h)

        zone_ndvi = ndvi_change[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]
        zone_ndbi = ndbi_change[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]

        veg_loss = np.mean(zone_ndvi < -ndvi_thresh) * 100
        veg_gain = np.mean(zone_ndvi > ndvi_thresh) * 100
        urban = np.mean(zone_ndbi > ndbi_thresh) * 100

        feat["properties"]["popup"] = (
            f"<b>Ward Statistics</b><br>"
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

with st.spinner("Running automated satellite analysis…"):
    ndvi_change, ndbi_change = compute_change(bbox, year-1, year)

veg_loss_mask = ndvi_change < -ndvi_thresh
veg_gain_mask = ndvi_change > ndvi_thresh
urban_mask = ndbi_change > ndbi_thresh

ward_geo = build_ward_popup_geojson(
    boundary, ndvi_change, ndbi_change, bbox
)

# ==================================================
# INTERACTIVE MAP
# ==================================================
m = leafmap.Map(
    center=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2],
    zoom=11,
    tiles=basemap
)

m.add_geojson(
    ward_geo,
    layer_name="Wards (click for stats)"
)

m.add_layer_control()
m.to_streamlit(height=650)

# ==================================================
# ANALYTICS SUMMARY
# ==================================================
st.subheader("📊 City-Level Analytics")

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{np.mean(veg_loss_mask)*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{np.mean(veg_gain_mask)*100:.2f}%")
c3.metric("Urban Expansion (%)", f"{np.mean(urban_mask)*100:.2f}%")

# ==================================================
# DOWNLOAD / EXPORT
# ==================================================
st.subheader("⬇️ Download & Export")

summary_df = pd.DataFrame({
    "Metric": ["Vegetation Loss", "Vegetation Gain", "Urban Expansion"],
    "Percentage (%)": [
        np.mean(veg_loss_mask)*100,
        np.mean(veg_gain_mask)*100,
        np.mean(urban_mask)*100,
    ]
})

st.download_button(
    "Download City Analytics (CSV)",
    summary_df.to_csv(index=False),
    "city_analytics.csv",
    "text/csv"
)

st.download_button(
    "Download Ward GeoJSON (with popups)",
    json.dumps(ward_geo),
    "ward_analysis.geojson",
    "application/geo+json"
)

st.markdown("""
### 🗺️ Legend
- Vegetation loss → NDVI decrease  
- Vegetation gain → NDVI increase  
- Urban expansion → NDBI increase  

### 📚 Data & Methods
- Sentinel-2 Level-2A (ESA Copernicus)
- NDVI & NDBI spectral indices
- Indian municipal boundaries (Datameet)
- ISRO / NRSC urban remote-sensing practices
""")

st.success("✅ Fully automated advanced GIS dashboard ready")
