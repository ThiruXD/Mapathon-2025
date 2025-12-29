import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pystac_client import Client
import planetary_computer as pc

st.set_page_config(page_title="India Land Change Detection", layout="wide")

st.title("🇮🇳 Land Change Detection for India")

# Area: Tamil Nadu (can change)
bbox = [77.9, 12.9, 78.1, 13.1]

def get_ndvi(date_range):
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}}
    )

    item = next(search.get_items())
    item = pc.sign(item)

    red = rasterio.open(item.assets["B04"].href).read(1).astype("float32")
    nir = rasterio.open(item.assets["B08"].href).read(1).astype("float32")

    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

with st.spinner("Processing satellite data..."):
    ndvi_old = get_ndvi("2019-01-01/2019-12-31")
    ndvi_new = get_ndvi("2024-01-01/2024-12-31")

change = ndvi_new - ndvi_old

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 NDVI Change Map")
    fig, ax = plt.subplots()
    im = ax.imshow(change, cmap="RdYlGn")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Statistics")
    st.metric("Vegetation Loss", int(np.sum(change < -0.2)))
    st.metric("Vegetation Gain", int(np.sum(change > 0.2)))

st.success("Automated analysis completed!")
