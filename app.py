import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pystac_client import Client
import planetary_computer as pc
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Chennai Land Change Detection", layout="wide")

st.title("🌍 Advanced Land Change Detection – Chennai, Tamil Nadu")
st.caption("Python-based visual analytics using open Sentinel-2 satellite data")

# ---------------------------------
# CHENNAI BBOX
# ---------------------------------
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# ---------------------------------
# LOAD CHENNAI BOUNDARY (GeoJSON)
# ---------------------------------
@st.cache_data
def load_boundary():
    return gpd.read_file("chennai_boundary.geojson")

chennai_boundary = load_boundary()

# ---------------------------------
# NDVI FUNCTION (CACHED)
# ---------------------------------
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

    item = pc.sign(list(search.get_items())[0])

    with rasterio.open(item.assets["B04"].href) as red_src:
        red = red_src.read(1).astype("float32")
    with rasterio.open(item.assets["B08"].href) as nir_src:
        nir = nir_src.read(1).astype("float32")

    # Speed optimization
    red = red[::4, ::4]
    nir = nir[::4, ::4]

    return (nir - red) / (nir + red + 1e-10)

# ---------------------------------
# YEAR SLIDER (BEFORE / AFTER)
# ---------------------------------
year_before = st.slider("Select BEFORE year", 2018, 2022, 2019)
year_after = st.slider("Select AFTER year", 2023, 2025, 2024)

with st.spinner("Processing satellite data..."):
    ndvi_before = get_ndvi(f"{year_before}-01-01/{year_before}-12-31")
    ndvi_after = get_ndvi(f"{year_after}-01-01/{year_after}-12-31")

change = ndvi_after - ndvi_before

# ---------------------------------
# MAP WITH CHENNAI BOUNDARY
# ---------------------------------
st.subheader("🗺️ Chennai Boundary Overlay")

m = folium.Map(location=[13.05, 80.27], zoom_start=10)
folium.GeoJson(chennai_boundary, name="Chennai Boundary").add_to(m)
st_folium(m, height=400, width=700)

# ---------------------------------
# VISUAL ANALYSIS
# ---------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("NDVI (Before)")
    st.image(ndvi_before, clamp=True)

with col2:
    st.subheader("NDVI (After)")
    st.image(ndvi_after, clamp=True)

with col3:
    st.subheader("NDVI Change Map")
    fig, ax = plt.subplots()
    im = ax.imshow(change, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax)
    ax.axis("off")
    st.pyplot(fig)

# ---------------------------------
# CLASSIFIED CHANGE MAP
# ---------------------------------
st.subheader("📌 Classified Land Change")

classified = np.zeros(change.shape)
classified[change < -0.2] = -1   # Loss
classified[change > 0.2] = 1     # Gain

fig2, ax2 = plt.subplots()
ax2.imshow(classified, cmap="bwr")
ax2.set_title("Red = Loss | Blue = Gain")
ax2.axis("off")
st.pyplot(fig2)

# ---------------------------------
# ANALYTICS
# ---------------------------------
total_pixels = change.size
loss = np.sum(change < -0.2)
gain = np.sum(change > 0.2)
stable = total_pixels - (loss + gain)

st.subheader("📊 Advanced Analytics")

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{(loss/total_pixels)*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{(gain/total_pixels)*100:.2f}%")
c3.metric("Stable Area (%)", f"{(stable/total_pixels)*100:.2f}%")

# Bar chart
st.subheader("📈 Change Distribution")
st.bar_chart({
    "Loss": loss,
    "Gain": gain,
    "Stable": stable
})

st.success("✅ Advanced Chennai land change analysis completed!")
