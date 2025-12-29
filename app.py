import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pystac_client import Client
import planetary_computer as pc
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from rasterio.enums import Resampling

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Chennai Urban & Land Change Detection",
    layout="wide"
)

st.title("🌍 Urban & Land Change Detection – Chennai, Tamil Nadu")
st.caption("Python-based satellite analytics using open Sentinel-2 data")

st.info(
    "⏳ First run may take ~10–20 seconds due to satellite data download. "
    "Results are cached and load instantly afterwards."
)

# -------------------------------------------------
# CHENNAI BOUNDING BOX (SMALL AREA = FAST)
# lon_min, lat_min, lon_max, lat_max
# -------------------------------------------------
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# -------------------------------------------------
# LOAD CHENNAI BOUNDARY (LOCAL FILE)
# -------------------------------------------------
@st.cache_data
def load_boundary():
    return gpd.read_file("chennai_boundary.geojson")

chennai_boundary = load_boundary()

# -------------------------------------------------
# SATELLITE INDEX FUNCTION (NDVI + NDBI)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_indices(date_range):
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
        raise RuntimeError("No satellite scenes found")

    item = pc.sign(items[0])

    # --- Read RED (10m)
    with rasterio.open(item.assets["B04"].href) as src:
        red = src.read(
            1,
            out_shape=(src.height // 2, src.width // 2),
            resampling=Resampling.average
        ).astype("float32")

    # --- Read NIR (10m)
    with rasterio.open(item.assets["B08"].href) as src:
        nir = src.read(
            1,
            out_shape=(src.height // 2, src.width // 2),
            resampling=Resampling.average
        ).astype("float32")

    # --- Read SWIR (20m → resampled to 10m)
    with rasterio.open(item.assets["B11"].href) as src:
        swir = src.read(
            1,
            out_shape=(nir.shape[0], nir.shape[1]),
            resampling=Resampling.bilinear
        ).astype("float32")

    # --- INDICES
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndbi = (swir - nir) / (swir + nir + 1e-10)

    return ndvi, ndbi

# -------------------------------------------------
# YEAR SLIDERS (BEFORE / AFTER)
# -------------------------------------------------
year_before = st.slider("Select BEFORE year", 2018, 2022, 2019)
year_after = st.slider("Select AFTER year", 2023, 2025, 2024)

with st.spinner("Processing Chennai satellite data..."):
    ndvi_before, ndbi_before = get_indices(
        f"{year_before}-01-01/{year_before}-12-31"
    )
    ndvi_after, ndbi_after = get_indices(
        f"{year_after}-01-01/{year_after}-12-31"
    )

veg_change = ndvi_after - ndvi_before
urban_change = ndbi_after - ndbi_before

# -------------------------------------------------
# MAP WITH CHENNAI BOUNDARY
# -------------------------------------------------
st.subheader("🗺️ Chennai Administrative Boundary")

m = folium.Map(location=[13.05, 80.27], zoom_start=10)
folium.GeoJson(
    chennai_boundary,
    name="Chennai Boundary",
    style_function=lambda x: {
        "fillOpacity": 0,
        "color": "blue",
        "weight": 2
    }
).add_to(m)

st_folium(m, height=400, width=700)

# -------------------------------------------------
# VISUAL COMPARISON
# -------------------------------------------------
st.subheader("🖼️ Before / After Visual Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("NDVI (Before)")
    st.image(ndvi_before, clamp=True)

with col2:
    st.caption("NDVI (After)")
    st.image(ndvi_after, clamp=True)

with col3:
    fig, ax = plt.subplots()
    im = ax.imshow(veg_change, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title("Vegetation Change")
    ax.axis("off")
    st.pyplot(fig)

# -------------------------------------------------
# URBAN GROWTH (ROADS + BUILDINGS)
# -------------------------------------------------
st.subheader("🏗️ Urban Expansion (Buildings & Roads – NDBI)")

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(urban_change, cmap="inferno", vmin=-0.3, vmax=0.3)
plt.colorbar(im2, ax=ax2, label="Built-up Change (NDBI)")
ax2.axis("off")
st.pyplot(fig2)

# -------------------------------------------------
# CLASSIFIED MAPS
# -------------------------------------------------
st.subheader("📌 Classified Change Maps")

veg_class = np.zeros(veg_change.shape)
veg_class[veg_change < -0.2] = -1
veg_class[veg_change > 0.2] = 1

urb_class = np.zeros(urban_change.shape)
urb_class[urban_change > 0.2] = 1

col4, col5 = st.columns(2)

with col4:
    fig3, ax3 = plt.subplots()
    ax3.imshow(veg_class, cmap="bwr")
    ax3.set_title("Vegetation Loss / Gain")
    ax3.axis("off")
    st.pyplot(fig3)

with col5:
    fig4, ax4 = plt.subplots()
    ax4.imshow(urb_class, cmap="Reds")
    ax4.set_title("Urban Growth Zones")
    ax4.axis("off")
    st.pyplot(fig4)

# -------------------------------------------------
# ADVANCED ANALYTICS
# -------------------------------------------------
st.subheader("📊 Advanced Analytics")

total_pixels = veg_change.size
veg_loss = np.sum(veg_change < -0.2)
veg_gain = np.sum(veg_change > 0.2)
urban_growth = np.sum(urban_change > 0.2)

c1, c2, c3 = st.columns(3)
c1.metric("Vegetation Loss (%)", f"{(veg_loss/total_pixels)*100:.2f}%")
c2.metric("Vegetation Gain (%)", f"{(veg_gain/total_pixels)*100:.2f}%")
c3.metric("Urban Growth Pixels", int(urban_growth))

st.bar_chart({
    "Vegetation Loss": veg_loss,
    "Vegetation Gain": veg_gain,
    "Urban Growth": urban_growth
})

# -------------------------------------------------
# ISRO / NRSC CITATION
# -------------------------------------------------
st.divider()
st.subheader("📚 Data Sources & Scientific References")

st.markdown("""
**Satellite Data**
- Sentinel-2 Level-2A (ESA Copernicus Programme)

**Indian Remote Sensing Context**
- Indian Space Research Organisation (ISRO)
- National Remote Sensing Centre (NRSC), Hyderabad

**Methodologies**
- NDVI for vegetation monitoring
- NDBI for built-up (roads & buildings) detection
- Standard urban remote sensing techniques
""")

st.success("✅ Full Chennai land & urban change analysis completed!")

