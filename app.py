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
st.set_page_config(
    page_title="Chennai Land & Urban Change Detection",
    layout="wide"
)

st.title("🌍 Land & Urban Change Detection – Chennai, Tamil Nadu")
st.caption("Python-based satellite analytics using open Sentinel-2 data")

st.info(
    "⏳ First run may take ~10–20 seconds due to satellite data download. "
    "Results are cached and load instantly afterwards."
)

# --------------------------------------------------
# CHENNAI BBOX (SMALL AREA = FAST)
# --------------------------------------------------
CHENNAI_BBOX = [80.20, 12.90, 80.35, 13.15]

# --------------------------------------------------
# LOAD CHENNAI BOUNDARY (FOR VISUAL OVERLAY)
# --------------------------------------------------
@st.cache_data
def load_boundary():
    with open("chennai_boundary.geojson") as f:
        return json.load(f)

boundary = load_boundary()

# --------------------------------------------------
# SATELLITE INDICES FUNCTION (NDVI + NDBI)
# --------------------------------------------------
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

    items = list(search.items())
    if not items:
        raise RuntimeError("No satellite data found")

    item = pc.sign(items[0])

    # --- RED (10m)
    with rasterio.open(item.assets["B04"].href) as src:
        red = src.read(
            1,
            out_shape=(src.height // 2, src.width // 2),
            resampling=Resampling.average
        ).astype("float32")

    # --- NIR (10m)
    with rasterio.open(item.assets["B08"].href) as src:
        nir = src.read(
            1,
            out_shape=(src.height // 2, src.width // 2),
            resampling=Resampling.average
        ).astype("float32")

    # --- SWIR (20m → resampled)
    with rasterio.open(item.assets["B11"].href) as src:
        swir = src.read(
            1,
            out_shape=(nir.shape[0], nir.shape[1]),
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
# PROCESS DATA (GUARDED)
# --------------------------------------------------
try:
    with st.spinner("Processing Chennai satellite data..."):
        ndvi_before, ndbi_before = get_indices(
            f"{year_before}-01-01/{year_before}-12-31"
        )
        ndvi_after, ndbi_after = get_indices(
            f"{year_after}-01-01/{year_after}-12-31"
        )

    veg_change = ndvi_after - ndvi_before
    urban_change = ndbi_after - ndbi_before

except Exception:
    st.error("❌ Satellite processing failed. Please refresh once.")
    st.stop()

# --------------------------------------------------
# CHENNAI BOUNDARY OVERLAY (STATIC – SAFE)
# --------------------------------------------------
st.subheader("🗺️ Chennai Administrative Boundary")

fig_map, ax_map = plt.subplots(figsize=(5, 5))
for feature in boundary["features"]:
    coords = feature["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax_map.plot(xs, ys, color="blue")

ax_map.set_title("Chennai Boundary")
ax_map.set_xlabel("Longitude")
ax_map.set_ylabel("Latitude")
st.pyplot(fig_map)

# --------------------------------------------------
# BEFORE / AFTER VISUALS
# --------------------------------------------------
st.subheader("🖼️ Before / After Comparison")

c1, c2, c3 = st.columns(3)

with c1:
    st.caption("NDVI – Before")
    st.image(ndvi_before, clamp=True)

with c2:
    st.caption("NDVI – After")
    st.image(ndvi_after, clamp=True)

with c3:
    fig, ax = plt.subplots()
    im = ax.imshow(veg_change, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title("Vegetation Change")
    ax.axis("off")
    st.pyplot(fig)

# --------------------------------------------------
# URBAN CHANGE (ROADS + BUILDINGS)
# --------------------------------------------------
st.subheader("🏗️ Urban Expansion (Buildings & Roads – NDBI)")

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(urban_change, cmap="inferno", vmin=-0.3, vmax=0.3)
plt.colorbar(im2, ax=ax2, label="Built-up Change (NDBI)")
ax2.axis("off")
st.pyplot(fig2)

# --------------------------------------------------
# CLASSIFIED MAPS
# --------------------------------------------------
st.subheader("📌 Classified Change Maps")

veg_class = np.zeros(veg_change.shape)
veg_class[veg_change < -0.2] = -1
veg_class[veg_change > 0.2] = 1

urb_class = (urban_change > 0.2).astype(int)

c4, c5 = st.columns(2)

with c4:
    fig3, ax3 = plt.subplots()
    ax3.imshow(veg_class, cmap="bwr")
    ax3.set_title("Vegetation Loss / Gain")
    ax3.axis("off")
    st.pyplot(fig3)

with c5:
    fig4, ax4 = plt.subplots()
    ax4.imshow(urb_class, cmap="Reds")
    ax4.set_title("Urban Growth Zones")
    ax4.axis("off")
    st.pyplot(fig4)

# --------------------------------------------------
# ADVANCED ANALYTICS
# --------------------------------------------------
st.subheader("📊 Advanced Analytics")

total = veg_change.size
loss = np.sum(veg_change < -0.2)
gain = np.sum(veg_change > 0.2)
urban = np.sum(urban_change > 0.2)

a1, a2, a3 = st.columns(3)
a1.metric("Vegetation Loss (%)", f"{loss/total*100:.2f}%")
a2.metric("Vegetation Gain (%)", f"{gain/total*100:.2f}%")
a3.metric("Urban Growth Pixels", int(urban))

st.bar_chart({
    "Vegetation Loss": loss,
    "Vegetation Gain": gain,
    "Urban Growth": urban
})

# --------------------------------------------------
# ISRO / NRSC CITATION
# --------------------------------------------------
st.divider()
st.subheader("📚 Data Sources & References")

st.markdown("""
**Satellite Data**
- Sentinel-2 Level-2A (ESA Copernicus Programme)

**Indian Remote Sensing Context**
- Indian Space Research Organisation (ISRO)
- National Remote Sensing Centre (NRSC), Hyderabad

**Scientific Methods**
- NDVI for vegetation change
- NDBI for built-up (roads & buildings) detection
""")

st.success("✅ Chennai land & urban change analysis completed successfully!")
