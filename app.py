import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from folium import plugins
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import streamlit.components.v1 as components

# Determine Base Directory of current script
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Streamlit Page Configuration
st.set_page_config(
    page_title="Peta Klasifikasi Land Use Land Cover (LULC) Satelit",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Header Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stApp {
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-title">🛰️ Peta Klasifikasi Land Use and Land Cover (LULC) Jawa Timur</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Visualisasi Multi-Layer Satelit Sentinel-2 (Band 11 SWIR, Band 8 NIR, Band 4 Red) & Pemodelan Machine Learning</div>', unsafe_allow_html=True)

# Colors Mapping for 5 LULC Classes (Matching Satellite Land Cover Standard)
CLASS_COLORS = {
    'Air': '#1f77b4',                # Blue
    'Hutan Mangrove': '#2ca02c',     # Light Green (Flooded Veg)
    'Hutan Non-Mangrove': '#006400', # Dark Green (Trees)
    'Pemukiman': '#d62728',         # Red (Built Area)
    'Sawah': '#bcbd22'               # Yellow / Olive (Crops)
}

COLOR_ORANGE = '#ff7f0e'  # Dedicated color for Misclassified / Mismatch samples

FITUR_LIST = ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI', 'MNDWI', 'NDBI']

# Sidebar Controls
st.sidebar.title("⚙️ Kontrol Layer & Model")

# 1. Model Machine Learning Selector
model_options = {
    "k-Nearest Neighbors (k-NN)": "model_lulc_knn.pkl",
    "Decision Tree Classifier": "model_lulc_dt.pkl",
    "Random Forest Classifier": "model_lulc_rf.pkl",
    "Support Vector Machine (SVM)": "model_lulc_svm.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "🤖 Pilih Model Machine Learning:",
    list(model_options.keys())
)

model_filename = model_options[selected_model_name]

# Robust search for model files across directories
search_model_paths = [
    os.path.join(APP_DIR, "models", model_filename),
    os.path.join(APP_DIR, "uts", "models", model_filename),
    os.path.join(APP_DIR, "..", "models", model_filename),
    os.path.join(APP_DIR, "..", "..", "uts", "models", model_filename),
    os.path.join("uts", "models", model_filename),
    os.path.join("models", model_filename),
    model_filename
]

model_path = next((p for p in search_model_paths if os.path.exists(p)), None)

@st.cache_resource
def load_ml_model(path):
    if path and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"Gagal memuat model: {e}")
            return None
    return None

model = load_ml_model(model_path)

if model is None:
    st.sidebar.warning(f"File model '{model_filename}' belum ditemukan. Pastikan file .pkl ada di folder models.")

# 2. Interactive Layer Checkbox Controls
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Filter Layer Klasifikasi LULC")
st.sidebar.write("Centang kelas tutupan lahan untuk menampilkan warna klasifikasi di atas citra satelit:")

show_air = st.sidebar.checkbox("🔵 Water (Air)", value=True)
show_mangrove = st.sidebar.checkbox("🟢 Flooded Vegetation (Hutan Mangrove)", value=True)
show_non_mangrove = st.sidebar.checkbox("🌲 Trees (Hutan Non-Mangrove)", value=True)
show_pemukiman = st.sidebar.checkbox("🔴 Built Area (Pemukiman)", value=True)
show_sawah = st.sidebar.checkbox("🌾 Crops (Sawah)", value=True)

st.sidebar.markdown("---")
show_mismatch = st.sidebar.checkbox("⚠️ Tampilkan Prediksi Salah (ORANGE)", value=True)

active_classes = []
if show_air: active_classes.append('Air')
if show_mangrove: active_classes.append('Hutan Mangrove')
if show_non_mangrove: active_classes.append('Hutan Non-Mangrove')
if show_pemukiman: active_classes.append('Pemukiman')
if show_sawah: active_classes.append('Sawah')

# Robust Search for GeoJSON and CSV Data Files
@st.cache_data
def load_spatial_data():
    prov_search = [
        os.path.join(APP_DIR, "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join(APP_DIR, "uts", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join(APP_DIR, "..", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join(APP_DIR, "..", "..", "uts", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join("uts", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join("data", "geojson", "jawa_timur_provinsi.geojson")
    ]
    sample_search = [
        os.path.join(APP_DIR, "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join(APP_DIR, "uts", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join(APP_DIR, "..", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join(APP_DIR, "..", "..", "uts", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join("uts", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join("data", "geojson", "jawatimur_5kelas.geojson")
    ]
    csv_search = [
        os.path.join(APP_DIR, "data", "csv", "poligon", "centroid_poligon.csv"),
        os.path.join(APP_DIR, "uts", "data", "csv", "poligon", "centroid_poligon.csv"),
        os.path.join(APP_DIR, "..", "data", "csv", "poligon", "centroid_poligon.csv"),
        os.path.join(APP_DIR, "..", "..", "uts", "data", "csv", "poligon", "centroid_poligon.csv"),
        os.path.join("uts", "data", "csv", "poligon", "centroid_poligon.csv"),
        os.path.join("data", "csv", "poligon", "centroid_poligon.csv")
    ]

    p_path = next((p for p in prov_search if os.path.exists(p)), None)
    s_path = next((p for p in sample_search if os.path.exists(p)), None)
    c_path = next((p for p in csv_search if os.path.exists(p)), None)

    gdf_prov = gpd.read_file(p_path) if p_path else None
    gdf_samples = gpd.read_file(s_path) if s_path else None
    df_centroid = pd.read_csv(c_path) if c_path else None

    return gdf_prov, gdf_samples, df_centroid

gdf_prov, gdf_samples, df_centroid = load_spatial_data()

# Compute ML Predictions on Centroids
if df_centroid is not None and model is not None:
    df_centroid['prediksi_ml'] = model.predict(df_centroid[FITUR_LIST])
    df_centroid['is_correct'] = df_centroid['prediksi_ml'] == df_centroid['label_teks']
    n_mismatch_total = (~df_centroid['is_correct']).sum()
else:
    n_mismatch_total = 0

if gdf_samples is not None:
    type_map = {
        'Air': 'Air',
        'Hutan_mangrove': 'Hutan Mangrove',
        'Hutan_non_mangrove': 'Hutan Non-Mangrove',
        'Pemukiman': 'Pemukiman',
        'Sawah': 'Sawah'
    }
    gdf_samples['Type_Clean'] = gdf_samples['Type'].map(lambda x: type_map.get(x, x))

# Tabs
tab1, tab2, tab3 = st.tabs(["🗺️ Peta Land Cover Satelit", "📊 Evaluasi Model ML", "🔮 Prediksi Nilai Spektral"])

with tab1:
    st.markdown("##### 🗺️ Peta Klasifikasi 10m Global Land Cover Jawa Timur (Prediksi ML vs Mismatch Oranye)")
    st.caption("Titik Hijau/Biru/Merah/Kuning = Prediksi ML Benar | Titik ORANYE = Prediksi Salah (Mismatch)")

    # Build Folium Map
    m = folium.Map(location=[-7.60, 112.60], zoom_start=9, tiles=None)

    # Layer 2: Basemap Satelit (Bottom)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satelit Esri World Imagery',
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer('openstreetmap', name='OpenStreetMap Standard').add_to(m)

    # Batas Provinsi
    if gdf_prov is not None:
        folium.GeoJson(
            gdf_prov,
            name='Batas Provinsi Jawa Timur',
            style_function=lambda x: {'color': '#ff7800', 'weight': 2, 'fillOpacity': 0.05}
        ).add_to(m)

    # Layer 1: Correct ML Predictions Layer
    if df_centroid is not None and model is not None:
        for class_name in active_classes:
            sub_df = df_centroid[(df_centroid['prediksi_ml'] == class_name) & (df_centroid['is_correct'])]
            if not sub_df.empty:
                color = CLASS_COLORS.get(class_name, '#333333')
                fg = folium.FeatureGroup(name=f"Prediksi Benar: {class_name} ({len(sub_df)})", show=True)
                for idx, row in sub_df.iterrows():
                    tooltip_html = f"""
                    <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
                        <b style="font-size: 13px; color: #1E3A8A;">Poligon ID: {row['poligon_id']}</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Hasil Prediksi ML:</b> <span style="color: {color}; font-weight: bold;">{row['prediksi_ml']}</span><br>
                        <b>Label Ground Truth:</b> {row['label_teks']}<br>
                        <b style="color: #2ca02c;">Status: PREDIKSI BENAR (MATCH)</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Nilai Spektral & Indeks:</b><br>
                        - NDVI: <b>{row['NDVI']:.3f}</b> | NDWI: <b>{row['NDWI']:.3f}</b> | NDBI: <b>{row['NDBI']:.3f}</b>
                    </div>
                    """
                    folium.CircleMarker(
                        location=[row['lat_centroid'], row['lon_centroid']],
                        radius=7.5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.85,
                        weight=1.5,
                        popup=folium.Popup(tooltip_html, max_width=250),
                        tooltip=f"Prediksi Benar: {row['prediksi_ml']}"
                    ).add_to(fg)
                fg.add_to(m)

        # Layer ORANGE: Misclassified / Mismatch Samples
        if show_mismatch:
            df_mismatch = df_centroid[~df_centroid['is_correct']]
            if not df_mismatch.empty:
                fg_mismatch = folium.FeatureGroup(name=f"⚠️ Prediksi Salah / Mismatch (ORANGE - {len(df_mismatch)})", show=True)
                for idx, row in df_mismatch.iterrows():
                    tooltip_html = f"""
                    <div style="font-family: sans-serif; font-size: 12px; width: 250px; background-color: #FFF5EE; padding: 5px; border-radius: 5px;">
                        <b style="font-size: 13px; color: #d62728;">⚠️ PREDIKSI SALAH (MISMATCH)</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Poligon ID:</b> {row['poligon_id']}<br>
                        <b>Prediksi Model ML:</b> <span style="color: {COLOR_ORANGE}; font-weight: bold;">{row['prediksi_ml']}</span><br>
                        <b>Seharusnya (Ground Truth):</b> <b>{row['label_teks']}</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Nilai Spektral & Indeks:</b><br>
                        - NDVI: <b>{row['NDVI']:.3f}</b> | NDWI: <b>{row['NDWI']:.3f}</b> | NDBI: <b>{row['NDBI']:.3f}</b><br>
                        - SWIR (B11): {row['B11']:.3f} | NIR (B08): {row['B08']:.3f} | Red (B04): {row['B04']:.3f}
                    </div>
                    """
                    folium.CircleMarker(
                        location=[row['lat_centroid'], row['lon_centroid']],
                        radius=9.5,
                        color='#d62728',
                        fill=True,
                        fill_color=COLOR_ORANGE,
                        fill_opacity=0.95,
                        weight=2.5,
                        popup=folium.Popup(tooltip_html, max_width=270),
                        tooltip=f"⚠️ PREDIKSI SALAH: Model={row['prediksi_ml']} | GroundTruth={row['label_teks']}"
                    ).add_to(fg_mismatch)
                fg_mismatch.add_to(m)

    # Floating Legend Box (Matching 10m Global Land Cover Style in landuseLandCover.png)
    legend_html = f'''
    <div style="
        position: fixed; 
        bottom: 35px; left: 35px; width: 260px; height: auto; 
        border:2px solid #ccc; z-index:99999; font-size:13px;
        background-color:white; opacity: 0.95; padding: 12px;
        border-radius: 8px; font-family: sans-serif;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    ">
        <b style="font-size:14px; color:#1E3A8A;">10m Global Land Cover</b><br>
        <div style="font-size:11px; color:#666; margin-bottom:6px;">Impact Observatory / Sentinel-2 LULC</div>
        <hr style="margin:4px 0 6px 0;">
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Air']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Water (Air)</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Hutan Non-Mangrove']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Trees (Hutan Non-Mangrove)</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Hutan Mangrove']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Flooded Vegetation (Hutan Mangrove)</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Sawah']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Crops (Sawah)</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Pemukiman']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Built Area (Pemukiman)</b>
        </div>
        <hr style="margin:4px 0 6px 0;">
        <div style="display:flex; align-items:center; margin-bottom:2px; color:#d62728;">
            <span style="background:{COLOR_ORANGE}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px; border:1px solid #d62728;"></span>
            <b>⚠️ ORANYE: Prediksi Salah ({n_mismatch_total})</b>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    # Render Map HTML with 700px height
    components.html(m._repr_html_(), height=700, scrolling=False)

with tab2:
    st.subheader("📊 Evaluasi Metrik Model Machine Learning")
    
    if df_centroid is not None and model is not None:
        X = df_centroid[FITUR_LIST]
        y_true = df_centroid['label_teks']
        y_pred = model.predict(X)

        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Machine Learning", selected_model_name)
        col2.metric("Overall Accuracy", f"{acc * 100:.2f}%")
        col3.metric("Cohen's Kappa Score", f"{kappa:.4f}")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_COLORS.keys()))
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=list(CLASS_COLORS.keys()),
                        yticklabels=list(CLASS_COLORS.keys()), ax=ax)
            plt.ylabel("Aktual (Ground Truth)")
            plt.xlabel("Prediksi Model")
            plt.xticks(rotation=30)
            st.pyplot(fig)

        with col_right:
            st.markdown("#### Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose().round(3)
            st.dataframe(df_report, use_container_width=True)
    elif model is None:
        st.error("Model Machine Learning belum dimuat. Silakan periksa keberadaan file model di folder models.")

with tab3:
    st.subheader("🔮 Simulasi Prediksi Nilai Spektral Tunggal")
    st.write("Masukkan nilai reflektansi Band dan Indeks Spektral untuk menguji prediksi model secara langsung:")

    c1, c2, c3 = st.columns(3)
    val_b04 = c1.number_input("Band 04 (Red)", value=0.08)
    val_b08 = c2.number_input("Band 08 (NIR)", value=0.35)
    val_b11 = c3.number_input("Band 11 (SWIR)", value=0.12)

    c4, c5, c6 = st.columns(3)
    val_b02 = c4.number_input("Band 02 (Blue)", value=0.05)
    val_b03 = c5.number_input("Band 03 (Green)", value=0.09)
    val_ndvi = c6.number_input("NDVI Index", value=(val_b08 - val_b04)/(val_b08 + val_b04 + 1e-6))

    c7, c8, c9 = st.columns(3)
    val_ndwi = c7.number_input("NDWI Index", value=(val_b03 - val_b08)/(val_b03 + val_b08 + 1e-6))
    val_mndwi = c8.number_input("MNDWI Index", value=(val_b03 - val_b11)/(val_b03 + val_b11 + 1e-6))
    val_ndbi = c9.number_input("NDBI Index", value=(val_b11 - val_b08)/(val_b11 + val_b08 + 1e-6))

    if st.button("🔮 Jalankan Prediksi LULC"):
        if model is not None:
            input_data = pd.DataFrame([[val_b02, val_b03, val_b04, val_b08, val_b11, val_ndvi, val_ndwi, val_mndwi, val_ndbi]], columns=FITUR_LIST)
            pred_label = model.predict(input_data)[0]
            st.success(f"Hasil Prediksi Model **{selected_model_name}**: **{pred_label}**")
        else:
            st.error("Model Machine Learning belum dimuat. Tidak dapat menjalankan prediksi.")
