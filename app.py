import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pystac_client import Client
import planetary_computer as pc

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(
    page_title="Chennai Land Change Detection",
    layout="wide"
)

st.title("🌍 Land Change Detection – Chennai, Tamil Nadu")
st.caption(
    "Fast, area-specific satellite analysis using open Sentinel-2 data"
)

st.info(
    "⏳ First load may take ~15 seconds due to satellite download. "
    "Results are cached and load instantly afterwards."
)

# ----------------------------
# CHENNAI BOUNDING BOX (SMALL AREA)
# lon_min, lat_min, lon_max, lat_max
# ----------------------------
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# ----------------------------
# CACHED NDVI FUNCTION
# ----------------------------
@st.cache_data(show_spinner=False)
def get_ndvi(date_range):
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=CHENNAI_BBOX,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )

    items = list(search.get_items())
    if len(items) == 0:
        raise Exception("No satellite images found")

    item = pc.sign(items[0])

    with rasterio.open(item.assets["B04"].href) as red_src:
        red = red_src.read(1).astype("float32")

    with rasterio.open(item.assets["B08"].href) as nir_src:
        nir = nir_src.read(1).astype("float32")

    # 🔥 SPEED BOOST: downsample (every 4th pixel)
    red = red[::4, ::4]
    nir = nir[::4, ::4]

    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

# ----------------------------
# PROCESSING
# ----------------------------
with st.spinner("Processing Chennai satellite data..."):
    ndvi_2019 = get_ndvi("2019-01-01/2019-12-31")
    ndvi_2024 = get_ndvi("2024-01-01/2024-12-31")

change_map = ndvi_2024 - ndvi_2019

# ----------------------------
# VISUAL OUTPUT
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌱 NDVI Change Map (2019 → 2024)")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(change_map, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Change Statistics")

    vegetation_loss = int(np.sum(change_map < -0.2))
    vegetation_gain = int(np.sum(change_map > 0.2))
    no_change = int(np.sum((-0.2 <= change_map) & (change_map <= 0.2)))

    st.metric("Vegetation Loss Pixels", vegetation_loss)
    st.metric("Vegetation Gain Pixels", vegetation_gain)
    st.metric("No Significant Change", no_change)

st.success("✅ Chennai land change analysis completed successfully!")
