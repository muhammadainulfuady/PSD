# Deployment Dashboard LULC Jawa Timur (Streamlit)

Direktori ini berisi seluruh komponen yang siap di-*deploy* ke **Streamlit Community Cloud** untuk visualisasi interaktif tutupan lahan (LULC) Provinsi Jawa Timur menggunakan model **k-Nearest Neighbors (k-NN Scaled, $k=5$)**.

## 🌐 Live URL
👉 **[https://lulc-jatim.streamlit.app/](https://lulc-jatim.streamlit.app/)**

## 📂 Struktur Berkas
- `app.py`: Aplikasi utama Streamlit dengan palet visual Warm Orange, kartu metrik dinamis, dan integrasi peta Folium.
- `requirements.txt`: Dependensi pustaka Python untuk environment Streamlit Cloud.
- `*.csv`: 6 berkas analitik tabular sebagai *Single Source of Truth* (SSOT).
- `*.html`: Peta interaktif Folium (sebaran poligon training, testing, dan evaluasi spasial).

## 🚀 Konfigurasi Streamlit Cloud
- **Repository:** `muhammadainulfuady/PSD`
- **Branch:** `main`
- **Main file path:** `uts/deploy/app.py`
- **URL Domain:** `lulc-jatim.streamlit.app`
