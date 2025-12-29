## 🗺️ Automated Change Mapping for Urban India

### *Mapathon 2025 Submission*

---

## 📌 Problem Statement

### **Automated Change Mapping**

Rapid urbanization in Indian cities leads to significant changes in land use, including **vegetation loss**, **urban expansion**, and **environmental degradation**. However, city-level decision-making often lacks **automated, scalable, and ward-level spatial intelligence** derived from satellite data.

This project addresses the **Automated Change Mapping** challenge by developing an **end-to-end, reproducible system** that detects and analyzes land-use changes using satellite imagery and open-source geospatial tools.

---

## 🎯 Project Objective

To build an **automated, scalable, and reproducible urban change detection system** that:

* Detects **vegetation loss and gain**
* Identifies **urban expansion hotspots**
* Generates **ward-wise analytics**
* Supports **multi-city analysis**
* Produces **decision-ready outputs** for planners and policymakers

---

## 🛰️ Data Sources

### Satellite Data

* **Sentinel-2 (ESA)**
  Accessed via **Microsoft Planetary Computer**

  * Spatial resolution: 10–20 m
  * Used for NDVI & NDBI computation

### Administrative Boundaries

* City ward boundaries in **GeoJSON** format:

  * Bangalore (BBMP)
  * Chennai
  * Hyderabad

### Basemap

* **OpenStreetMap** (default visualization layer)

---

## 🧠 Methodology

### 1️⃣ Automated Satellite Processing

* Fetch cloud-free Sentinel-2 imagery
* Compute:

  * **NDVI** (vegetation index)
  * **NDBI** (built-up index)
* Perform **year-to-year change detection**

### 2️⃣ Ward-Level Spatial Analysis

* Clip raster data using ward boundaries
* Calculate:

  * Vegetation loss (%)
  * Urban expansion (%)
* Auto-generate **ward-wise statistics**

### 3️⃣ Analytics & Insights

* City-level KPIs
* Ward ranking tables
* Hotspot detection (top 10% change zones)
* Visual charts for comparison

---

## ✨ Key Features

### 🗺️ Interactive Map

* OpenStreetMap basemap
* Clickable ward polygons
* Popup analytics per ward

### 📊 Analytics Dashboard

* Average vegetation loss
* Average urban expansion
* Ward-wise rankings
* Hotspot identification

### 📈 Visualization

* Bar charts:

  * Top wards by vegetation loss
  * Top wards by urban expansion

### ⬇️ Data Export

* Download ward analytics as **CSV**
* Download enriched **GeoJSON** with popups

---

## 🏙️ Cities Supported

* **Bangalore**
* **Chennai**
* **Hyderabad**

The system is **city-agnostic** and can be extended to any Indian city with administrative boundary data.

---

## 🧪 Technology Stack

* **Python**
* **Streamlit** – Interactive dashboard
* **Rasterio** – Satellite raster processing
* **Leafmap (Folium backend)** – Web mapping
* **NumPy & Pandas** – Analysis
* **Shapely** – Geometry handling
* **Planetary Computer STAC API**

All tools used are **open-source**.

---

## ⚙️ How to Run the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
mapathon-2025-fathima/
├── app.py
├── requirements.txt
├── README.md
└── geojson/
    ├── BBMP.geojson
    ├── chennai_boundary.geojson
    └── hyderabad.geojson
```

---

## 🚀 Scalability & Reproducibility

* Fully automated pipeline
* Reusable for different cities and years
* Minimal manual intervention
* Designed for **Smart City**, **Urban Planning**, and **Environmental Monitoring** use cases

---

## ⚠️ Disclaimer

This analysis is based on satellite-derived indices and is intended for **planning and decision-support purposes only**. It should not be used for legal or regulatory enforcement without field validation.

