import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import json

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Chennai Land & Urban Change Detection", layout="wide")

st.title("🌍 Land & Urban Change Detection – Chennai, Tamil Nadu")
st.caption("Python-based satellite analytics using open Sentinel-2 data")

# --------------------------------------------------
# CHENNAI BBOX
# --------------------------------------------------
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# --------------------------------------------------
# LOAD CHENNAI BOUNDARY
# --------------------------------------------------
@st.cache_data
def load_boundary():
    with open("chennai_boundary.geojson") as f:
        return json.load(f)

boundary = load_boundary()

# --------------------------------------------------
# NDVI + NDBI FUNCTION
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_indices(date_range):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=CHENNAI_BBOX,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )

    items = list(search.items())
    item = pc.sign(items[0])

    def read_band(band, res):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // res, src.width // res),
                resampling=Resampling.average
            ).astype("float32")

    red = read_band("B04", 2)
    nir = read_band("B08", 2)

    with rasterio.open(item.assets["B11"].href) as src:
        swir = src.read(
            1,
            out_shape=nir.shape,
            resampling=Resampling.bilinear
        ).astype("float32")

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndbi = (swir - nir) / (swir + nir + 1e-10)

    return ndvi, ndbi

# --------------------------------------------------
# YEAR SLIDERS
# --------------------------------------------------
year_before = st.slider("Select BEFORE year", 2018, 2022, 2019)
year_after = st.slider("Select AFTER year", 2023, 2025, 2024)

# --------------------------------------------------
# PROCESS DATA
# --------------------------------------------------
with st.spinner("Processing satellite data..."):
    ndvi_b, ndbi_b = get_indices(f"{year_before}-01-01/{year_before}-12-31")
    ndvi_a, ndbi_a = get_indices(f"{year_after}-01-01/{year_after}-12-31")

veg_change = ndvi_a - ndvi_b
urban_change = ndbi_a - ndbi_b

# --------------------------------------------------
# CHENNAI BOUNDARY PLOT
# --------------------------------------------------
st.subheader("🗺️ Chennai Administrative Boundary")
fig_b, ax_b = plt.subplots()
for feat in boundary["features"]:
    xs = [c[0] for c in feat["geometry"]["coordinates"][0]]
    ys = [c[1] for c in feat["geometry"]["coordinates"][0]]
    ax_b.plot(xs, ys, color="blue")
ax_b.set_xlabel("Longitude")
ax_b.set_ylabel("Latitude")
st.pyplot(fig_b)

# --------------------------------------------------
# VEGETATION CHANGE MAP (SAFE COLORBAR)
# --------------------------------------------------
st.subheader("🌱 Vegetation Change (NDVI)")
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(veg_change, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
fig1.colorbar(im1, ax=ax1)
ax1.axis("off")
st.pyplot(fig1)

# --------------------------------------------------
# URBAN CHANGE MAP (SAFE COLORBAR)
# --------------------------------------------------
st.subheader("🏗️ Urban Expansion (NDBI)")
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(urban_change, cmap="inferno", vmin=-0.3, vmax=0.3)
fig2.colorbar(im2, ax=ax2)
ax2.axis("off")
st.pyplot(fig2)

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------
st.subheader("📊 Analytics")

total = veg_change.size
loss = np.sum(veg_change < -0.2)
gain = np.sum(veg_change > 0.2)
urban = np.sum(urban_change > 0.2)

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{loss/total*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{gain/total*100:.2f}%")
c3.metric("Urban Growth Pixels", int(urban))

st.bar_chart({
    "Vegetation Loss": loss,
    "Vegetation Gain": gain,
    "Urban Growth": urban
})

# --------------------------------------------------
# REFERENCES
# --------------------------------------------------
st.divider()
st.markdown("""
**Data Sources & References**
- Sentinel-2 Level-2A (ESA Copernicus)
- ISRO & NRSC urban remote sensing practices
- NDVI & NDBI spectral indices
""")

st.success("✅ App running successfully without Streamlit crashes!")
