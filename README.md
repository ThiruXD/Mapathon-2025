# Urban & Environmental Intelligence Dashboard

## Overview
This project is an interactive GIS dashboard that analyzes urban growth and vegetation change
using satellite imagery and ward-level administrative boundaries.

## Data Sources
- **Sentinel-2 (ESA)** via Microsoft Planetary Computer
- **OpenStreetMap** basemap for visualization
- City ward boundaries (GeoJSON)

## Key Features
- NDVI-based vegetation loss and gain
- NDBI-based urban expansion detection
- Ward-wise interactive popups
- Ward ranking tables
- Hotspot detection (top 10%)
- Downloadable analytics (CSV & GeoJSON)
- Mobile-friendly UI

## Cities Supported
- Bangalore
- Chennai
- Hyderabad

## Technology Stack
- Python
- Streamlit
- Leafmap (Folium backend)
- Rasterio
- NumPy
- Pandas

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
