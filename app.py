import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import matplotlib.pyplot as plt
import json

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="centered")
st.title("🌍 Automated Land & Urban Change Detection (India)")
st.caption("Python-based satellite analytics using open Sentinel-2 data")

st.info(
    "Select any city boundary. Satellite data, analysis, and visuals "
    "are generated automatically."
)

# ==================================================
# CITY CONFIGURATION
# ==================================================
CITY_FILES = {
    "Chennai": "geojson/chennai_boundary.geojson",
    "Coimbatore": "geojson/coimbatore_boundary.geojson"
}

# ==================================================
# LOAD & PROCESS GEOJSON
# ==================================================
@st.cache_data
def load_city_geojson(path):
    with open(path) as f:
        data = json.load(f)

    rings = []
    lons, lats = [], []

    for feat in data["features"]:
        geom = feat["geometry"]

        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
        else:  # MultiPolygon
            coords = geom["coordinates"][0][0]

        rings.append(coords)

        for c in coords:
            lons.append(c[0])
            lats.append(c[1])

    bbox = [min(lons), min(lats), max(lons), max(lats)]
    return rings, bbox

# ==================================================
# SATELLITE PROCESSING (CORE ENGINE)
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

    def read(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average
            ).astype("float32")

    # BEFORE
    red_b = read("B04", item_b, 4)
    nir_b = read("B08", item_b, 4)
    swir_b = read("B11", item_b, 8)

    # AFTER
    red_a = read("B04", item_a, 4)
    nir_a = read("B08", item_a, 4)
    swir_a = read("B11", item_a, 8)

    ndvi_b = (nir_b - red_b) / (nir_b + red_b + 1e-10)
    ndvi_a = (nir_a - red_a) / (nir_a + red_a + 1e-10)

    ndbi_b = (swir_b - nir_b[:swir_b.shape[0], :swir_b.shape[1]]) / (
        swir_b + nir_b[:swir_b.shape[0], :swir_b.shape[1]] + 1e-10
    )
    ndbi_a = (swir_a - nir_a[:swir_a.shape[0], :swir_a.shape[1]]) / (
        swir_a + nir_a[:swir_a.shape[0], :swir_a.shape[1]] + 1e-10
    )

    return ndvi_b, ndvi_a, ndvi_a - ndvi_b, ndbi_a - ndbi_b

# ==================================================
# USER INPUT
# ==================================================
city = st.selectbox("Select City", list(CITY_FILES.keys()))
year_before = st.selectbox("Before Year", [2019, 2020, 2021, 2022])
year_after = st.selectbox("After Year", [2023, 2024, 2025])

wards, bbox = load_city_geojson(CITY_FILES[city])

with st.spinner("Processing satellite data automatically..."):
    ndvi_before, ndvi_after, veg_change, urban_change = compute_change(
        bbox, year_before, year_after
    )

# ==================================================
# NDVI BEFORE / AFTER TOGGLE
# ==================================================
st.subheader("🖼️ NDVI Before / After")

view = st.radio("View", ["Before", "After"], horizontal=True)

fig0, ax0 = plt.subplots()
if view == "Before":
    ax0.imshow(ndvi_before, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax0.set_title(f"{city} NDVI – {year_before}")
else:
    ax0.imshow(ndvi_after, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax0.set_title(f"{city} NDVI – {year_after}")

ax0.axis("off")
st.pyplot(fig0)
plt.close(fig0)

# ==================================================
# VEGETATION CHANGE + BOUNDARY
# ==================================================
st.subheader("🌱 Vegetation Change with Ward Overlay")

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.imshow(veg_change, cmap="RdYlGn", vmin=-0.4, vmax=0.4)

for ring in wards:
    ax1.plot(
        [c[0] for c in ring],
        [c[1] for c in ring],
        color="black",
        linewidth=0.25
    )

ax1.set_title(f"{city} Vegetation Change (NDVI)")
ax1.axis("off")
st.pyplot(fig1)
plt.close(fig1)

# ==================================================
# URBAN CHANGE + BOUNDARY
# ==================================================
st.subheader("🏗️ Urban Growth (Buildings & Roads)")

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.imshow(urban_change, cmap="inferno", vmin=-0.3, vmax=0.3)

for ring in wards:
    ax2.plot(
        [c[0] for c in ring],
        [c[1] for c in ring],
        color="white",
        linewidth=0.25
    )

ax2.set_title(f"{city} Urban Expansion (NDBI)")
ax2.axis("off")
st.pyplot(fig2)
plt.close(fig2)

# ==================================================
# CLASSIFIED MAP
# ==================================================
st.subheader("📌 Classified Vegetation Change")

classified = np.zeros(veg_change.shape)
classified[veg_change < -0.2] = -1
classified[veg_change > 0.2] = 1

fig3, ax3 = plt.subplots()
ax3.imshow(classified, cmap="bwr")
ax3.axis("off")
st.pyplot(fig3)
plt.close(fig3)

# ==================================================
# ANALYTICS
# ==================================================
st.subheader("📊 Automatic Analytics")

total = veg_change.size
veg_loss = np.sum(veg_change < -0.2)
veg_gain = np.sum(veg_change > 0.2)
urban_growth = np.sum(urban_change > 0.2)

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{veg_loss/total*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{veg_gain/total*100:.2f}%")
c3.metric("Urban Growth (%)", f"{urban_growth/total*100:.2f}%")

st.bar_chart({
    "Vegetation Loss": veg_loss,
    "Vegetation Gain": veg_gain,
    "Urban Growth": urban_growth
})

# ==================================================
# REFERENCES
# ==================================================
st.divider()
st.subheader("📚 Data Sources & References")

st.markdown("""
**Satellite Data**
- Sentinel-2 Level-2A (ESA Copernicus Programme)

**Indian Remote Sensing Context**
- ISRO
- NRSC, Hyderabad

**Methods**
- NDVI for vegetation monitoring  
- NDBI for urban (roads & buildings) expansion
""")

st.success("✅ Fully automated analysis completed successfully")
