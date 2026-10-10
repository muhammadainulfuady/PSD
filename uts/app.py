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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

# Determine Base Directory of current script
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Streamlit Page Configuration with Orange & White Accent
st.set_page_config(
    page_title="Peta & Model LULC Jawa Timur - Sentinel-2",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern Orange & White CSS Styling
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #FFFBF7;
        color: #1E293B;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8800 50%, #FFA500 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(255, 107, 0, 0.25);
    }
    .header-box h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .header-box p {
        color: #FFF0E6;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #FFE4D6;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFEFE6;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        color: #D95200;
        font-weight: 600;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B00 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(255, 107, 0, 0.3);
    }
    
    /* Cards and Metric Boxes */
    .orange-card {
        background-color: #FFFFFF;
        border-left: 5px solid #FF6B00;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .orange-badge {
        background-color: #FFF0E6;
        color: #FF6B00;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Primary Orange Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF6B00 0%, #E65100 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 107, 0, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #E65100 0%, #CC4400 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(255, 107, 0, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# Header Display
st.markdown("""
<div class="header-box">
    <h1>🛰️ Peta Klasifikasi Land Use Land Cover (LULC) Jawa Timur</h1>
    <p>Visualisasi Digitasi Satelit Sentinel-2, Exploratory Data Analysis (EDA), Pemodelan Machine Learning & Evaluasi Mismatch</p>
</div>
""", unsafe_allow_html=True)

# LULC Class Color Definitions
CLASS_COLORS = {
    'Air': '#1f77b4',                # Blue
    'Hutan Mangrove': '#2ca02c',     # Light Green (Flooded Veg)
    'Hutan Non-Mangrove': '#006400', # Dark Green (Trees)
    'Pemukiman': '#d62728',         # Red (Built Area)
    'Sawah': '#bcbd22'               # Yellow / Olive (Crops)
}

COLOR_ORANGE = '#FF6B00'  # Dedicated color for Misclassified / Mismatch samples

FITUR_LIST = ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI', 'MNDWI', 'NDBI']

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Kontrol & Konfigurasi")

# Data Loader Function
@st.cache_data
def load_all_datasets():
    prov_search = [
        os.path.join(APP_DIR, "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join(APP_DIR, "..", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join(APP_DIR, "..", "..", "uts", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join("uts", "data", "geojson", "jawa_timur_provinsi.geojson"),
        os.path.join("data", "geojson", "jawa_timur_provinsi.geojson")
    ]
    sample_search = [
        os.path.join(APP_DIR, "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join(APP_DIR, "..", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join(APP_DIR, "..", "..", "uts", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join("uts", "data", "geojson", "jawatimur_5kelas.geojson"),
        os.path.join("data", "geojson", "jawatimur_5kelas.geojson")
    ]
    csv_search = [
        os.path.join(APP_DIR, "data", "csv", "poligon", "centroid_poligon.csv"),
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

gdf_prov, gdf_samples, df_centroid = load_all_datasets()

# Helper Model Training/Loading Function
@st.cache_resource
def get_ml_model(model_type, _df):
    if _df is None or _df.empty:
        return None
        
    X = _df[FITUR_LIST]
    y = _df['label_teks']
    
    # Check if pre-trained pkl exists first
    model_files = {
        "k-NN": "model_lulc_knn.pkl",
        "Decision Tree": "model_lulc_dt.pkl",
        "Random Forest": "model_lulc_rf.pkl",
        "SVM": "model_lulc_svm.pkl"
    }
    
    fname = model_files.get(model_type, "")
    search_paths = [
        os.path.join(APP_DIR, "models", fname),
        os.path.join(APP_DIR, "..", "models", fname),
        os.path.join(APP_DIR, "..", "..", "uts", "models", fname),
        os.path.join("uts", "models", fname),
        os.path.join("models", fname)
    ]
    found_path = next((p for p in search_paths if os.path.exists(p)), None)
    
    if found_path:
        try:
            return joblib.load(found_path)
        except Exception:
            pass
            
    # Fallback dynamic training
    from sklearn.pipeline import make_pipeline
    if model_type == "k-NN":
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance'))
    elif model_type == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else: # SVM
        clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', probability=True))
        
    clf.fit(X, y)
    return clf

# Main Navigation Tabs (Orange Theme)
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Peta Digitasi Sampel", 
    "📊 EDA (Analisis Spektral)", 
    "🤖 Prediksi ML & Mismatch", 
    "📌 Kesimpulan & Deploy"
])

# ---------------------------------------------------------
# TAB 1: PETA DIGITASI SAMPEL (POLIGON & CENTROID)
# ---------------------------------------------------------
with tab1:
    st.markdown("### 🗺️ Peta Digitasi Poligon & Centroid Sampel Tutupan Lahan (5 Kelas)")
    st.caption("Visualisasi hasil digitasi sampel poligon di Jawa Timur di atas Basemap Satelit Esri World Imagery.")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Poligon Sampel", f"{len(df_centroid) if df_centroid is not None else 0} Poligon")
    col_b.metric("Jumlah Kelas LULC", "5 Kelas Tutupan Lahan")
    col_c.metric("Resolusi Citra", "10 Meter (Sentinel-2)")

    st.markdown("---")
    
    # Layer Filter Checkboxes for Digitization Map
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    show_air_d = col_f1.checkbox("🔵 Air", value=True, key="d_air_uts")
    show_mangrove_d = col_f2.checkbox("🟢 Hutan Mangrove", value=True, key="d_mangrove_uts")
    show_non_mangrove_d = col_f3.checkbox("🌲 Hutan Non-Mangrove", value=True, key="d_non_mangrove_uts")
    show_pemukiman_d = col_f4.checkbox("🔴 Pemukiman", value=True, key="d_pemukiman_uts")
    show_sawah_d = col_f5.checkbox("🌾 Sawah", value=True, key="d_sawah_uts")
    
    active_d_classes = []
    if show_air_d: active_d_classes.append('Air')
    if show_mangrove_d: active_d_classes.append('Hutan Mangrove')
    if show_non_mangrove_d: active_d_classes.append('Hutan Non-Mangrove')
    if show_pemukiman_d: active_d_classes.append('Pemukiman')
    if show_sawah_d: active_d_classes.append('Sawah')

    # Build Map 1: Digitization Map
    m_dig = folium.Map(location=[-7.60, 112.60], zoom_start=9, tiles=None)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satelit Esri World Imagery',
        overlay=False,
        control=True
    ).add_to(m_dig)
    
    folium.TileLayer('openstreetmap', name='OpenStreetMap Standard').add_to(m_dig)
    
    if gdf_prov is not None:
        folium.GeoJson(
            gdf_prov,
            name='Batas Provinsi Jawa Timur',
            style_function=lambda x: {'color': '#FF6B00', 'weight': 2, 'fillOpacity': 0.03}
        ).add_to(m_dig)

    if df_centroid is not None:
        for cname in active_d_classes:
            sub = df_centroid[df_centroid['label_teks'] == cname]
            if not sub.empty:
                color = CLASS_COLORS.get(cname, '#333')
                fg = folium.FeatureGroup(name=f"Poligon Digitasi: {cname} ({len(sub)})", show=True)
                for idx, row in sub.iterrows():
                    tooltip_html = f"""
                    <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
                        <b style="font-size: 13px; color: #FF6B00;">Poligon ID: {row['poligon_id']}</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Kelas Tutupan Lahan:</b> <span style="color: {color}; font-weight: bold;">{row['label_teks']}</span><br>
                        <b>Jumlah Piksel:</b> {row['n_piksel']} piksel<br>
                        <hr style="margin: 3px 0;">
                        <b>Indeks Spektral:</b><br>
                        - NDVI: <b>{row['NDVI']:.3f}</b><br>
                        - NDWI: <b>{row['NDWI']:.3f}</b><br>
                        - NDBI: <b>{row['NDBI']:.3f}</b>
                    </div>
                    """
                    folium.CircleMarker(
                        location=[row['lat_centroid'], row['lon_centroid']],
                        radius=7,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.85,
                        weight=1.5,
                        popup=folium.Popup(tooltip_html, max_width=250),
                        tooltip=f"Digitasi: {row['label_teks']} ({row['poligon_id']})"
                    ).add_to(fg)
                fg.add_to(m_dig)

    # Floating Legend Box (Orange-bordered)
    legend_dig_html = f'''
    <div style="
        position: fixed; 
        bottom: 35px; left: 35px; width: 250px; height: auto; 
        border:2px solid #FF6B00; z-index:99999; font-size:13px;
        background-color:white; opacity: 0.95; padding: 12px;
        border-radius: 10px; font-family: sans-serif;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.2);
    ">
        <b style="font-size:14px; color:#FF6B00;">📌 Sampel Digitasi LULC</b><br>
        <div style="font-size:11px; color:#666; margin-bottom:6px;">Ground Truth Centroid Poligon</div>
        <hr style="margin:4px 0 6px 0;">
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Air']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Air (Water)</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Hutan Mangrove']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Hutan Mangrove</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Hutan Non-Mangrove']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Hutan Non-Mangrove</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Pemukiman']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Pemukiman (Built Area)</b>
        </div>
        <div style="display:flex; align-items:center; margin-bottom:5px;">
            <span style="background:{CLASS_COLORS['Sawah']}; width:15px; height:15px; display:inline-block; margin-right:8px; border-radius:3px;"></span>
            <b>Sawah (Crops)</b>
        </div>
    </div>
    '''
    m_dig.get_root().html.add_child(folium.Element(legend_dig_html))
    folium.LayerControl(collapsed=False).add_to(m_dig)
    
    components.html(m_dig._repr_html_(), height=650, scrolling=False)


# ---------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis (EDA) Fitur Spektral Satelit")
    st.write("Analisis statistik dan separabilitas spektral untuk membedakan 5 kelas tutupan lahan di Jawa Timur.")
    
    if df_centroid is not None:
        col_eda1, col_eda2 = st.columns([1, 1])
        
        with col_eda1:
            st.markdown("#### 1. Distribusi Jumlah Sampel Poligon per Kelas")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            class_counts = df_centroid['label_teks'].value_counts()
            colors_bar = [CLASS_COLORS.get(c, '#FF6B00') for c in class_counts.index]
            bars = ax1.bar(class_counts.index, class_counts.values, color=colors_bar, edgecolor='black', linewidth=0.8)
            ax1.set_ylabel("Jumlah Poligon")
            ax1.set_xlabel("Kelas Tutupan Lahan")
            plt.xticks(rotation=25)
            for bar in bars:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig1)
            
        with col_eda2:
            st.markdown("#### 2. Matrix Korelasi Fitur Spektral (Warm Oranges)")
            fig2, ax2 = plt.subplots(figsize=(6, 4.2))
            corr = df_centroid[FITUR_LIST].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="Oranges", ax=ax2, cbar=True, annot_kws={"size": 8})
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        st.markdown("---")
        st.markdown("#### 3. Analisis Separabilitas Indeks Spektral per Kelas (Boxplot)")
        
        selected_feature = st.selectbox(
            "Pilih Indeks / Band Spektral untuk Diinspeksi:",
            ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'B11', 'B08', 'B04', 'B03', 'B02'],
            key="eda_feature_select_uts"
        )
        
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        sns.boxplot(
            data=df_centroid, 
            x='label_teks', 
            y=selected_feature, 
            palette=CLASS_COLORS,
            ax=ax3,
            boxprops=dict(alpha=0.85)
        )
        ax3.set_title(f"Distribusi Reflektansi / Nilai {selected_feature} pada 5 Kelas LULC", fontsize=12, fontweight='bold', color='#FF6B00')
        ax3.set_xlabel("Kelas Tutupan Lahan")
        ax3.set_ylabel(f"Nilai {selected_feature}")
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig3)
        
        st.markdown("""
        <div class="orange-card">
            <span class="orange-badge">💡 Insight Spektral Remote Sensing</span><br>
            <ul>
                <li><b>Hutan Mangrove vs Non-Mangrove:</b> Memiliki nilai NDVI yang sama-sama tinggi, namun Hutan Mangrove memiliki nilai SWIR (Band 11) dan NDBI yang lebih rendah akibat efek pembasahan air pasang surut.</li>
                <li><b>Objek Air:</b> Memiliki nilai NDWI dan MNDWI paling tinggi (> 0.2), serta NDVI negatif.</li>
                <li><b>Pemukiman (Built Area):</b> Menunjukkan nilai NDBI paling tinggi dan nilai NDWI paling rendah.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 3: PREDIKSI MACHINE LEARNING & MISMATCH ORANGE
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🤖 Pemodelan Machine Learning & Penandaan Mismatch Oranye")
    st.write("Pilih algoritma model ML untuk melihat hasil prediksi klasifikasi spasial dan penandaan titik misklasifikasi dalam **Warna Oranye**.")

    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        selected_model_name = st.selectbox(
            "🤖 Pilih Algoritma Klasifikasi Machine Learning:",
            ["k-NN", "Decision Tree", "Random Forest", "SVM"],
            key="model_select_uts"
        )
        
    model = get_ml_model(selected_model_name, df_centroid)

    # Compute Predictions
    if df_centroid is not None and model is not None:
        df_centroid['prediksi_ml'] = model.predict(df_centroid[FITUR_LIST])
        df_centroid['is_correct'] = df_centroid['prediksi_ml'] == df_centroid['label_teks']
        n_correct = df_centroid['is_correct'].sum()
        n_mismatch = (~df_centroid['is_correct']).sum()
        acc_curr = n_correct / len(df_centroid)
    else:
        n_correct, n_mismatch, acc_curr = 0, 0, 0.0

    st.markdown("---")
    
    # Layer Toggle for Predictions
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        show_mismatch_only = st.checkbox("⚠️ Highlight Khusus Prediksi Salah (ORANGE)", value=True, key="mismatch_toggle_uts")
    with col_t2:
        st.markdown(f"<b>Akurasi Model: <span style='color:#FF6B00;'>{acc_curr*100:.2f}%</span></b>", unsafe_allow_html=True)

    # Map 2: ML Prediction Map
    m_pred = folium.Map(location=[-7.60, 112.60], zoom_start=9, tiles=None)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satelit Esri World Imagery',
        overlay=False,
        control=True
    ).add_to(m_pred)
    
    folium.TileLayer('openstreetmap', name='OpenStreetMap Standard').add_to(m_pred)
    
    if gdf_prov is not None:
        folium.GeoJson(
            gdf_prov,
            name='Batas Provinsi Jawa Timur',
            style_function=lambda x: {'color': '#FF6B00', 'weight': 2, 'fillOpacity': 0.03}
        ).add_to(m_pred)

    if df_centroid is not None and model is not None:
        # Layer 1: Correct Predictions (Class Colors)
        for cname, color in CLASS_COLORS.items():
            sub = df_centroid[(df_centroid['prediksi_ml'] == cname) & (df_centroid['is_correct'])]
            if not sub.empty:
                fg = folium.FeatureGroup(name=f"Prediksi Benar: {cname} ({len(sub)})", show=True)
                for idx, row in sub.iterrows():
                    tooltip_html = f"""
                    <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
                        <b style="font-size: 13px; color: #1E3A8A;">Poligon ID: {row['poligon_id']}</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Hasil Prediksi ML:</b> <span style="color: {color}; font-weight: bold;">{row['prediksi_ml']}</span><br>
                        <b>Ground Truth:</b> {row['label_teks']}<br>
                        <b style="color: #2ca02c;">Status: PREDIKSI BENAR (MATCH)</b>
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
                fg.add_to(m_pred)

        # Layer 2: Mismatch Layer (ORANGE)
        if show_mismatch_only:
            df_mis = df_centroid[~df_centroid['is_correct']]
            if not df_mis.empty:
                fg_mis = folium.FeatureGroup(name=f"⚠️ Prediksi Salah / Mismatch (ORANGE - {len(df_mis)})", show=True)
                for idx, row in df_mis.iterrows():
                    tooltip_html = f"""
                    <div style="font-family: sans-serif; font-size: 12px; width: 250px; background-color: #FFF5EE; padding: 6px; border-radius: 6px; border: 1px solid #FF6B00;">
                        <b style="font-size: 13px; color: #d62728;">⚠️ PREDIKSI SALAH (MISMATCH)</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Poligon ID:</b> {row['poligon_id']}<br>
                        <b>Prediksi Model ML:</b> <span style="color: {COLOR_ORANGE}; font-weight: bold;">{row['prediksi_ml']}</span><br>
                        <b>Seharusnya (Ground Truth):</b> <b>{row['label_teks']}</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Indeks Spektral:</b><br>
                        - NDVI: <b>{row['NDVI']:.3f}</b> | NDWI: <b>{row['NDWI']:.3f}</b> | NDBI: <b>{row['NDBI']:.3f}</b>
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
                        tooltip=f"⚠️ MISMATCH: Model={row['prediksi_ml']} | Aktual={row['label_teks']}"
                    ).add_to(fg_mis)
                fg_mis.add_to(m_pred)

    # Floating Legend Box
    legend_pred_html = f'''
    <div style="
        position: fixed; 
        bottom: 35px; left: 35px; width: 260px; height: auto; 
        border:2px solid #FF6B00; z-index:99999; font-size:13px;
        background-color:white; opacity: 0.95; padding: 12px;
        border-radius: 10px; font-family: sans-serif;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.25);
    ">
        <b style="font-size:14px; color:#FF6B00;">🤖 10m Global Land Cover ML</b><br>
        <div style="font-size:11px; color:#666; margin-bottom:6px;">Klasifikasi Model: {selected_model_name}</div>
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
            <b>Flooded Veg (Hutan Mangrove)</b>
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
            <b>⚠️ ORANYE: Prediksi Salah ({n_mismatch})</b>
        </div>
    </div>
    '''
    m_pred.get_root().html.add_child(folium.Element(legend_pred_html))
    folium.LayerControl(collapsed=False).add_to(m_pred)
    
    components.html(m_pred._repr_html_(), height=650, scrolling=False)

    st.markdown("---")
    st.markdown("#### 📊 Evaluasi Metrik & Confusion Matrix")
    
    if df_centroid is not None and model is not None:
        y_true = df_centroid['label_teks']
        y_pred = df_centroid['prediksi_ml']

        col_ev1, col_ev2 = st.columns([1, 1])

        with col_ev1:
            st.markdown("##### Confusion Matrix (Oranges Colormap)")
            cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_COLORS.keys()))
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=list(CLASS_COLORS.keys()),
                        yticklabels=list(CLASS_COLORS.keys()), ax=ax_cm, cbar=False)
            plt.ylabel("Aktual (Ground Truth)")
            plt.xlabel("Prediksi Model ML")
            plt.xticks(rotation=30)
            st.pyplot(fig_cm)

        with col_ev2:
            st.markdown("##### Laporan Klasifikasi (Precision, Recall, F1-Score)")
            report = classification_report(y_true, y_pred, output_dict=True)
            df_rep = pd.DataFrame(report).transpose().round(3)
            st.dataframe(df_rep, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔮 Simulasi Prediksi Nilai Spektral Tunggal")
    
    c_s1, c_s2, c_s3 = st.columns(3)
    input_b04 = c_s1.number_input("Band 04 (Red)", value=0.08, key="b04_uts")
    input_b08 = c_s2.number_input("Band 08 (NIR)", value=0.35, key="b08_uts")
    input_b11 = c_s3.number_input("Band 11 (SWIR)", value=0.12, key="b11_uts")

    c_s4, c_s5, c_s6 = st.columns(3)
    input_b02 = c_s4.number_input("Band 02 (Blue)", value=0.05, key="b02_uts")
    input_b03 = c_s5.number_input("Band 03 (Green)", value=0.09, key="b03_uts")
    input_ndvi = c_s6.number_input("NDVI", value=(input_b08 - input_b04)/(input_b08 + input_b04 + 1e-6), key="ndvi_uts")

    c_s7, c_s8, c_s9 = st.columns(3)
    input_ndwi = c_s7.number_input("NDWI", value=(input_b03 - input_b08)/(input_b03 + input_b08 + 1e-6), key="ndwi_uts")
    input_mndwi = c_s8.number_input("MNDWI", value=(input_b03 - input_b11)/(input_b03 + input_b11 + 1e-6), key="mndwi_uts")
    input_ndbi = c_s9.number_input("NDBI", value=(input_b11 - input_b08)/(input_b11 + input_b08 + 1e-6), key="ndbi_uts")

    if st.button("🔮 Prediksi Tutupan Lahan", key="btn_pred_uts"):
        if model is not None:
            df_single = pd.DataFrame([[input_b02, input_b03, input_b04, input_b08, input_b11, input_ndvi, input_ndwi, input_mndwi, input_ndbi]], columns=FITUR_LIST)
            pred_res = model.predict(df_single)[0]
            pred_color = CLASS_COLORS.get(pred_res, '#FF6B00')
            st.markdown(f"""
            <div style="background-color: #FFF0E6; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B00;">
                <h4 style="color: #FF6B00; margin: 0;">Hasil Prediksi Model ({selected_model_name}):</h4>
                <p style="font-size: 1.4rem; font-weight: 800; color: {pred_color}; margin: 0.3rem 0 0 0;">
                    {pred_res}
                </p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 4: KESIMPULAN & PANDUAN DEPLOYMENT
# ---------------------------------------------------------
with tab4:
    st.markdown("### 📌 Kesimpulan Ringkas & Panduan Deploy Streamlit")
    
    st.markdown("""
    <div class="orange-card">
        <h4 style="color: #FF6B00; margin-top: 0;">1. Separabilitas Spektral & Karakteristik Citra Satelit</h4>
        <ul>
            <li><b>Hutan Mangrove vs Hutan Non-Mangrove:</b> Kombinasi <b>Band 11 (SWIR)</b> dan <b>Band 8 (NIR)</b> terbukti sangat krusial. Hutan Mangrove memiliki pantulan SWIR yang jauh lebih rendah akibat adanya genangan air pasang surut di bawah kanopi vegetasi.</li>
            <li><b>Perairan (Air):</b> NDWI dan MNDWI bernilai tinggi positif (> 0.2), memisahkan perairan secara sempurna dari daratan.</li>
            <li><b>Pemukiman (Built Area):</b> Menunjukkan reflektansi NDBI tertinggi karena sifat material bangunan yang memantulkan gelombang SWIR.</li>
            <li><b>Sawah:</b> Memiliki variabilitas NDVI sedang hingga tinggi tergantung pada siklus tanam dan genangan air irigasi.</li>
        </ul>
    </div>
    
    <div class="orange-card">
        <h4 style="color: #FF6B00; margin-top: 0;">2. Evaluasi Model Machine Learning & Visualisasi Mismatch</h4>
        <ul>
            <li>Model <b>k-Nearest Neighbors (k-NN)</b> dengan <i>StandardScaler()</i> dan <b>Random Forest Classifier</b> menghasilkan akurasi klasifikasi terbaik (> 80 - 85%).</li>
            <li>Penandaan <b>Titik Mismatch Oranye (⚠️)</b> memudahkan inspeksi spasial langsung di atas citra satelit, di mana mayoritas kesalahan prediksi terjadi pada area transisi perbatasan antara sawah dan permukiman padat.</li>
        </ul>
    </div>
    
    <div class="orange-card">
        <h4 style="color: #FF6B00; margin-top: 0;">3. Struktur Deploy Streamlit Cloud</h4>
        <p>Aplikasi ini siap di-deploy ke <b>Streamlit Community Cloud</b> atau server web lokal dengan lokasi file utama:</p>
        <ul>
            <li><b>File Aplikasi Deploy:</b> <code>uts/deploy/app.py</code></li>
            <li><b>File Model Trained:</b> <code>uts/models/model_lulc_knn.pkl</code></li>
            <li><b>File Data Spasial:</b> <code>uts/data/geojson/jawatimur_5kelas.geojson</code> & <code>uts/data/csv/poligon/centroid_poligon.csv</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
