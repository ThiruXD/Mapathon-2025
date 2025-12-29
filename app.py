import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="centered")
st.title("🌍 Chennai Land & Urban Change Detection")
st.caption("Stable Streamlit Cloud version (Python only)")

# -----------------------------
# CHENNAI BBOX
# -----------------------------
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# -----------------------------
# NDVI / NDBI FUNCTION
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_change(year1, year2):
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

    item1 = get_scene(year1)
    item2 = get_scene(year2)

    def read(band, item, scale):
        with rasterio.open(item.assets[band].href) as src:
            return src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.average
            ).astype("float32")

    # ---- BEFORE
    red1 = read("B04", item1, 4)
    nir1 = read("B08", item1, 4)
    swir1 = read("B11", item1, 8)

    # ---- AFTER
    red2 = read("B04", item2, 4)
    nir2 = read("B08", item2, 4)
    swir2 = read("B11", item2, 8)

    ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-10)
    ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-10)

    ndbi1 = (swir1 - nir1[:swir1.shape[0], :swir1.shape[1]]) / (swir1 + nir1[:swir1.shape[0], :swir1.shape[1]] + 1e-10)
    ndbi2 = (swir2 - nir2[:swir2.shape[0], :swir2.shape[1]]) / (swir2 + nir2[:swir2.shape[0], :swir2.shape[1]] + 1e-10)

    return ndvi2 - ndvi1, ndbi2 - ndbi1

# -----------------------------
# USER INPUT
# -----------------------------
year_before = st.selectbox("Before year", [2019, 2020, 2021, 2022])
year_after = st.selectbox("After year", [2023, 2024, 2025])

# -----------------------------
# PROCESS
# -----------------------------
with st.spinner("Processing satellite data (stable mode)..."):
    veg_change, urban_change = compute_change(year_before, year_after)

# -----------------------------
# VEGETATION MAP
# -----------------------------
st.subheader("🌱 Vegetation Change (NDVI)")
fig1, ax1 = plt.subplots()
ax1.imshow(veg_change, cmap="RdYlGn", vmin=-0.4, vmax=0.4)
ax1.axis("off")
st.pyplot(fig1)
plt.close(fig1)

# -----------------------------
# URBAN MAP
# -----------------------------
st.subheader("🏗️ Urban Growth (Buildings & Roads)")
fig2, ax2 = plt.subplots()
ax2.imshow(urban_change, cmap="inferno", vmin=-0.3, vmax=0.3)
ax2.axis("off")
st.pyplot(fig2)
plt.close(fig2)

# -----------------------------
# ANALYTICS
# -----------------------------
st.subheader("📊 Analytics")
st.metric("Vegetation Loss (%)", f"{np.mean(veg_change < -0.2)*100:.2f}%")
st.metric("Urban Expansion (%)", f"{np.mean(urban_change > 0.2)*100:.2f}%")

# -----------------------------
# REFERENCES
# -----------------------------
st.divider()
st.markdown("""
**Data & References**
- Sentinel-2 (ESA Copernicus Programme)
- ISRO / NRSC Urban Remote Sensing Practices
- NDVI & NDBI indices
""")

st.success("✅ Stable execution completed")
