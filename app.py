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
st.title("🌍 Chennai Land & Urban Change Detection")
st.caption("Python-based satellite analytics using open Sentinel-2 data")

st.info(
    "This application demonstrates vegetation and urban change in Chennai "
    "using open satellite data. Optimized for fast cloud execution."
)

# ==================================================
# CHENNAI BOUNDING BOX
# ==================================================
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# ==================================================
# LOAD CHENNAI BOUNDARY (LIGHTWEIGHT)
# ==================================================
@st.cache_data
def load_boundary():
    with open("chennai_boundary.geojson") as f:
        return json.load(f)

boundary = load_boundary()

# ==================================================
# SATELLITE PROCESSING (STABLE CORE)
# ==================================================
@st.cache_data(show_spinner=False)
def compute_change(year_before, year_after):
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    def get_scene(year):
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=CHENNAI_BBOX,
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

    # ---- BEFORE
    red_b = read("B04", item_b, 4)
    nir_b = read("B08", item_b, 4)
    swir_b = read("B11", item_b, 8)

    # ---- AFTER
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
year_before = st.selectbox("Before year", [2019, 2020, 2021, 2022])
year_after = st.selectbox("After year", [2023, 2024, 2025])

with st.spinner("Processing satellite data…"):
    ndvi_before, ndvi_after, veg_change, urban_change = compute_change(
        year_before, year_after
    )

# ==================================================
# CHENNAI BOUNDARY VISUAL (SAFE)
# ==================================================
st.subheader("🗺️ Chennai Administrative Boundary")

fig_b, ax_b = plt.subplots()
for feat in boundary["features"]:
    coords = feat["geometry"]["coordinates"][0]
    ax_b.plot([c[0] for c in coords], [c[1] for c in coords], color="blue")
ax_b.set_xlabel("Longitude")
ax_b.set_ylabel("Latitude")
ax_b.set_title("Chennai District Boundary")
st.pyplot(fig_b)
plt.close(fig_b)

# ==================================================
# BEFORE / AFTER TOGGLE (SAFE)
# ==================================================
st.subheader("🖼️ Before / After NDVI View")

view = st.radio("Select view", ["Before", "After"], horizontal=True)

fig_ndvi, ax_ndvi = plt.subplots()
if view == "Before":
    ax_ndvi.imshow(ndvi_before, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax_ndvi.set_title(f"NDVI – {year_before}")
else:
    ax_ndvi.imshow(ndvi_after, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax_ndvi.set_title(f"NDVI – {year_after}")

ax_ndvi.axis("off")
st.pyplot(fig_ndvi)
plt.close(fig_ndvi)

# ==================================================
# VEGETATION CHANGE MAP
# ==================================================
st.subheader("🌱 Vegetation Change (NDVI Difference)")

fig1, ax1 = plt.subplots()
ax1.imshow(veg_change, cmap="RdYlGn", vmin=-0.4, vmax=0.4)
ax1.axis("off")
st.pyplot(fig1)
plt.close(fig1)

# ==================================================
# URBAN CHANGE MAP (ROADS + BUILDINGS)
# ==================================================
st.subheader("🏗️ Urban Expansion (Buildings & Roads – NDBI)")

fig2, ax2 = plt.subplots()
ax2.imshow(urban_change, cmap="inferno", vmin=-0.3, vmax=0.3)
ax2.axis("off")
st.pyplot(fig2)
plt.close(fig2)

# ==================================================
# CLASSIFIED MAPS (ADVANCED IMAGE)
# ==================================================
st.subheader("📌 Classified Change Map")

classified = np.zeros(veg_change.shape)
classified[veg_change < -0.2] = -1
classified[veg_change > 0.2] = 1

fig3, ax3 = plt.subplots()
ax3.imshow(classified, cmap="bwr")
ax3.axis("off")
st.pyplot(fig3)
plt.close(fig3)

# ==================================================
# ADVANCED ANALYTICS
# ==================================================
st.subheader("📊 Analytics Summary")

total = veg_change.size
veg_loss = np.sum(veg_change < -0.2)
veg_gain = np.sum(veg_change > 0.2)
urban_growth = np.sum(urban_change > 0.2)

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{veg_loss/total*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{veg_gain/total*100:.2f}%")
c3.metric("Urban Expansion (%)", f"{urban_growth/total*100:.2f}%")

st.bar_chart({
    "Vegetation Loss": veg_loss,
    "Vegetation Gain": veg_gain,
    "Urban Expansion": urban_growth
})

# ==================================================
# ISRO / NRSC CITATION
# ==================================================
st.divider()
st.subheader("📚 Data Sources & References")

st.markdown("""
**Satellite Data**
- Sentinel-2 Level-2A (ESA Copernicus Programme)

**Indian Remote Sensing Context**
- Indian Space Research Organisation (ISRO)
- National Remote Sensing Centre (NRSC), Hyderabad

**Methods**
- NDVI for vegetation monitoring  
- NDBI for urban (roads & buildings) expansion
""")

st.success("✅ Analysis completed successfully and stably")
