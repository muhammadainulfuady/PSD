# Deployment Dashboard LULC Jawa Timur (Streamlit)

Direktori ini berisi seluruh komponen yang siap di-*deploy* ke **Streamlit Community Cloud** untuk visualisasi interaktif tutupan lahan (LULC) Provinsi Jawa Timur menggunakan model **k-Nearest Neighbors (k-NN Scaled, $k=5$)** yang menyajikan **Dua Versi Pendekatan Klasifikasi** (Versi Poligon & Versi Piksel).

Seluruh peta spasial interaktif dibangun dan dirender murni menggunakan pustaka **Folium**.

## 🌐 Live URL
👉 **[https://lulc-jatim.streamlit.app/](https://lulc-jatim.streamlit.app/)**

---

## 🗺️ Dua Versi Klasifikasi & Peta Folium

| Versi | Unit Analisis | Data Latih / Uji | Akurasi | Peta Interaktif Folium |
| :--- | :--- | :--- | :--- | :--- |
| **Versi 1: Tingkat Poligon** | Centroid 250 Poligon | 175 Train / 75 Test | **80.0%** | `map_lulc.html` (Ground Truth) & `map_lulc_predicted.html` (Evaluasi k-NN) |
| **Versi 2: Tingkat Piksel** | Piksel 10m Citra Sentinel-2 | 10.000 Train (Balanced) / 54.164 Test | **91.0%** | `peta_crawling_lulc_jatim.html` (10,2 Juta Piksel Se-Jawa Timur) |

---

## 📂 Struktur Berkas
- `app.py`: Aplikasi utama Streamlit dengan palet visual Warm Orange, kartu metrik komparasi poligon & piksel, dan tab peta interaktif Folium.
- `requirements.txt`: Dependensi pustaka Python untuk environment Streamlit Cloud.
- `peta_crawling_lulc_jatim.html`: Peta interaktif Folium hasil klasifikasi raster 10,2 juta piksel se-Jawa Timur (5 layer toggleable).
- `map_lulc.html`: Peta interaktif Folium sebaran 250 poligon sampel digitasi acuan.
- `map_lulc_predicted.html`: Peta interaktif Folium evaluasi prediksi spasial model k-NN (Train vs Test).
- `rekapitulasi_lulc_jatim_raster.csv`: Tabel rekapitulasi jumlah piksel, persentase, dan estimasi luas (km²) tutupan lahan Jawa Timur.
- `metrik_evaluasi_knn.csv`, `confusion_matrix.csv`, dll.: Berkas analitik tabular sebagai *Single Source of Truth* (SSOT).

---

## 🚀 Konfigurasi Streamlit Cloud
- **Repository:** `muhammadainulfuady/PSD`
- **Branch:** `main`
- **Main file path:** `uts/deploy/app.py`
- **URL Domain:** `lulc-jatim.streamlit.app`
