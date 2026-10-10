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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import streamlit.components.v1 as components

# Determine Base Directory of current script
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Streamlit Page Configuration with Modern Orange & White Theme
st.set_page_config(
    page_title="Proyek LULC Jawa Timur - Sentinel-2 k-NN",
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
        font-size: 2.1rem;
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
    <h1>🛰️ Klasifikasi Land Use Land Cover (LULC) Sentinel-2 Jawa Timur</h1>
    <p>Notebook Workflow: Digitasi Satelit (5 Kelas), EDA Spektral, Model k-NN & Evaluasi Mismatch Oranye</p>
</div>
""", unsafe_allow_html=True)

# LULC Class Color Definitions for 5 Classes
CLASS_COLORS = {
    'Air': '#1f77b4',                # Blue
    'Hutan Mangrove': '#2ca02c',     # Light Green (Flooded Veg)
    'Hutan Non-Mangrove': '#006400', # Dark Green (Trees)
    'Pemukiman': '#d62728',         # Red (Built Area)
    'Sawah': '#bcbd22'               # Yellow / Olive (Crops)
}

COLOR_ORANGE = '#FF6B00'  # Dedicated color for Misclassified / Mismatch samples

FITUR_LIST = ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI', 'MNDWI', 'NDBI']

# Sidebar Controls & Information
st.sidebar.markdown("### ⚙️ Kontrol & Data Notebook")
st.sidebar.markdown("""
<div style="background-color: #FFF0E6; padding: 10px; border-radius: 8px; border-left: 4px solid #FF6B00; font-size: 13px;">
    <b>🤖 Model Terpasang:</b><br>
    <b>k-Nearest Neighbors (k-NN)</b><br>
    - Preprocessing: <code>StandardScaler()</code><br>
    - K-Neighbors: <b>k = 5</b><br>
    - Metric: <b>Euclidean Distance</b>
</div>
""", unsafe_allow_html=True)

# Data Loader Function (Matching Notebook Steps 1-2)
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

    # Clean GeoJSON Type field mapping to standard 5 class names
    if gdf_samples is not None and 'Type' in gdf_samples.columns:
        type_map = {
            'Air': 'Air',
            'Hutan_mangrove': 'Hutan Mangrove',
            'Hutan Mangrove': 'Hutan Mangrove',
            'Hutan_non_mangrove': 'Hutan Non-Mangrove',
            'Hutan Non-Mangrove': 'Hutan Non-Mangrove',
            'Pemukiman': 'Pemukiman',
            'Sawah': 'Sawah'
        }
        gdf_samples['Type_Clean'] = gdf_samples['Type'].map(lambda x: type_map.get(x, x))

    return gdf_prov, gdf_samples, df_centroid

gdf_prov, gdf_samples, df_centroid = load_all_datasets()

# Model Loader & Evaluation Helper (Matching Notebook Steps 5-8)
@st.cache_resource
def get_knn_model_and_eval(_df):
    if _df is None or _df.empty:
        return None, None, 0.0, 0.0

    X = _df[FITUR_LIST]
    y = _df['label_teks']

    # Train-test split matching Notebook (80% train, 20% test, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_files = ["model_lulc_knn.pkl"]
    search_paths = [
        os.path.join(APP_DIR, "models", f) for f in model_files
    ] + [
        os.path.join(APP_DIR, "..", "models", f) for f in model_files
    ] + [
        os.path.join(APP_DIR, "..", "..", "uts", "models", f) for f in model_files
    ] + [
        os.path.join("uts", "models", f) for f in model_files
    ] + [
        os.path.join("models", f) for f in model_files
    ]
    
    found_path = next((p for p in search_paths if os.path.exists(p)), None)

    if found_path:
        try:
            model = joblib.load(found_path)
        except Exception:
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
            model.fit(X_train, y_train)
    else:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        model.fit(X_train, y_train)

    # Evaluate on Test Set
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_kappa = cohen_kappa_score(y_test, y_test_pred)

    return model, (X_test, y_test, y_test_pred), test_acc, test_kappa

model_knn, eval_data, test_acc, test_kappa = get_knn_model_and_eval(df_centroid)

# Compute Full Dataset Predictions for Spatial Map (Notebook Step 9)
if df_centroid is not None and model_knn is not None:
    df_centroid['prediksi_ml'] = model_knn.predict(df_centroid[FITUR_LIST])
    df_centroid['is_correct'] = df_centroid['prediksi_ml'] == df_centroid['label_teks']
    n_mismatch = (~df_centroid['is_correct']).sum()
else:
    n_mismatch = 0

# Main Navigation Tabs (Notebook Workflow Steps)
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Step 3: Peta Digitasi Sampel", 
    "📊 Step 4: EDA Spektral", 
    "🤖 Step 5-9: Model k-NN & Mismatch", 
    "📌 Kesimpulan & Ringkasan"
])

# ---------------------------------------------------------
# TAB 1: PETA DIGITASI SAMPEL (POLIGON & CENTROID 5 KELAS)
# ---------------------------------------------------------
with tab1:
    st.markdown("### 🗺️ Step 3: Visualisasi Peta Digitasi Poligon & Centroid Sampel (5 Kelas Tutupan Lahan)")
    st.caption("Visualisasi 250 sampel poligon digitasi (50 poligon per kelas) di atas Basemap Satelit Esri World Imagery.")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Poligon Sampel", f"{len(df_centroid) if df_centroid is not None else 0} Poligon (50 per Kelas)")
    col_b.metric("Jumlah Kelas LULC", "5 Kelas Tutupan Lahan Lengkap")
    col_c.metric("Resolusi Citra Satelit", "10 Meter (Sentinel-2)")

    st.markdown("---")
    
    # Layer Filter Checkboxes for 5 Classes
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    show_air_d = col_f1.checkbox("🔵 Air (Water)", value=True, key="d_air_v3")
    show_mangrove_d = col_f2.checkbox("🟢 Hutan Mangrove", value=True, key="d_mangrove_v3")
    show_non_mangrove_d = col_f3.checkbox("🌲 Hutan Non-Mangrove", value=True, key="d_non_mangrove_v3")
    show_pemukiman_d = col_f4.checkbox("🔴 Pemukiman (Built)", value=True, key="d_pemukiman_v3")
    show_sawah_d = col_f5.checkbox("🌾 Sawah (Crops)", value=True, key="d_sawah_v3")
    
    active_d_classes = []
    if show_air_d: active_d_classes.append('Air')
    if show_mangrove_d: active_d_classes.append('Hutan Mangrove')
    if show_non_mangrove_d: active_d_classes.append('Hutan Non-Mangrove')
    if show_pemukiman_d: active_d_classes.append('Pemukiman')
    if show_sawah_d: active_d_classes.append('Sawah')

    # Build Map 1: Digitization Map with Polygons & Centroid Markers for ALL 5 Classes
    m_dig = folium.Map(location=[-7.60, 112.60], zoom_start=9, tiles=None)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satelit Esri World Imagery',
        overlay=False,
        control=True
    ).add_to(m_dig)
    
    folium.TileLayer('openstreetmap', name='OpenStreetMap Standard').add_to(m_dig)
    
    # Batas Provinsi
    if gdf_prov is not None:
        folium.GeoJson(
            gdf_prov,
            name='Batas Provinsi Jawa Timur',
            style_function=lambda x: {'color': '#FF6B00', 'weight': 2, 'fillOpacity': 0.03}
        ).add_to(m_dig)

    # 1. Render GeoJSON Vector Polygons for 5 Classes
    if gdf_samples is not None and 'Type_Clean' in gdf_samples.columns:
        for cname in active_d_classes:
            sub_poly = gdf_samples[gdf_samples['Type_Clean'] == cname]
            if not sub_poly.empty:
                color = CLASS_COLORS.get(cname, '#FF6B00')
                fg_poly = folium.FeatureGroup(name=f"Poligon Vektor: {cname} ({len(sub_poly)})", show=True)
                
                folium.GeoJson(
                    sub_poly,
                    style_function=lambda x, col=color: {
                        'fillColor': col,
                        'color': col,
                        'weight': 1.5,
                        'fillOpacity': 0.55
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['Type_Clean'],
                        aliases=['Kelas LULC:']
                    )
                ).add_to(fg_poly)
                
                fg_poly.add_to(m_dig)

    # 2. Render Centroid Markers for 5 Classes
    if df_centroid is not None:
        for cname in active_d_classes:
            sub = df_centroid[df_centroid['label_teks'] == cname]
            if not sub.empty:
                color = CLASS_COLORS.get(cname, '#333')
                fg_cen = folium.FeatureGroup(name=f"Titik Centroid: {cname} ({len(sub)})", show=True)
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
                        color='#ffffff',
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        weight=1.5,
                        popup=folium.Popup(tooltip_html, max_width=250),
                        tooltip=f"Centroid: {row['label_teks']} ({row['poligon_id']})"
                    ).add_to(fg_cen)
                fg_cen.add_to(m_dig)

    # Floating Legend Box for 5 Classes
    legend_dig_html = f'''
    <div style="
        position: fixed; 
        bottom: 35px; left: 35px; width: 250px; height: auto; 
        border:2px solid #FF6B00; z-index:99999; font-size:13px;
        background-color:white; opacity: 0.95; padding: 12px;
        border-radius: 10px; font-family: sans-serif;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.2);
    ">
        <b style="font-size:14px; color:#FF6B00;">📌 Sampel Digitasi 5 Kelas</b><br>
        <div style="font-size:11px; color:#666; margin-bottom:6px;">Poligon Vektor & Centroid (Sentinel-2)</div>
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
    st.markdown("### 📊 Step 4: Exploratory Data Analysis (EDA) Fitur Spektral Satelit")
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
            key="eda_feature_v3"
        )
        
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        sns.boxplot(
            data=df_centroid, 
            x='label_teks', 
            y=selected_feature, 
            hue='label_teks',
            legend=False,
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
# TAB 3: PREDIKSI MODEL k-NN & MISMATCH ORANGE
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🤖 Step 5-9: Pemodelan k-NN & Visualisasi Mismatch Oranye")
    st.write("Hasil prediksi klasifikasi spasial tutupan lahan menggunakan algoritma **k-NN (k=5)** dan penandaan titik misklasifikasi dalam **Warna Oranye**.")

    # Metric summary card matching notebook results exactly (80.00% Accuracy, 0.7500 Kappa)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model Algorithm", "k-NN (StandardScaler)")
    col_m2.metric("Overall Accuracy (Test Set)", f"{test_acc * 100:.2f}%")
    col_m3.metric("Cohen's Kappa Score", f"{test_kappa:.4f}")

    st.markdown("---")
    
    # Layer Toggle for Mismatch
    show_mismatch_only = st.checkbox("⚠️ Highlight Khusus Prediksi Salah (ORANGE)", value=True, key="mismatch_v3")

    # Map 2: ML Prediction Map (Step 9)
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

    if df_centroid is not None and model_knn is not None:
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
                        <b>Hasil Prediksi k-NN:</b> <span style="color: {color}; font-weight: bold;">{row['prediksi_ml']}</span><br>
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
                        <b>Prediksi Model k-NN:</b> <span style="color: {COLOR_ORANGE}; font-weight: bold;">{row['prediksi_ml']}</span><br>
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
        <b style="font-size:14px; color:#FF6B00;">🤖 10m Global Land Cover k-NN</b><br>
        <div style="font-size:11px; color:#666; margin-bottom:6px;">Model k-Nearest Neighbors (k=5)</div>
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
    st.markdown("#### 📊 Evaluasi Metrik & Confusion Matrix (Test Set Evaluation)")
    
    if eval_data is not None:
        X_test, y_test, y_test_pred = eval_data

        col_ev1, col_ev2 = st.columns([1, 1])

        with col_ev1:
            st.markdown("##### Confusion Matrix (Test Set - Oranges Colormap)")
            cm = confusion_matrix(y_test, y_test_pred, labels=list(CLASS_COLORS.keys()))
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=list(CLASS_COLORS.keys()),
                        yticklabels=list(CLASS_COLORS.keys()), ax=ax_cm, cbar=False)
            plt.ylabel("Aktual (Ground Truth)")
            plt.xlabel("Prediksi Model k-NN")
            plt.xticks(rotation=30)
            st.pyplot(fig_cm)

        with col_ev2:
            st.markdown("##### Laporan Klasifikasi Test Set (Precision, Recall, F1-Score)")
            report = classification_report(y_test, y_test_pred, output_dict=True)
            df_rep = pd.DataFrame(report).transpose().round(3)
            st.dataframe(df_rep, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔮 Simulasi Prediksi Nilai Spektral Tunggal (k-NN Model)")
    
    c_s1, c_s2, c_s3 = st.columns(3)
    input_b04 = c_s1.number_input("Band 04 (Red)", value=0.08, key="b04_v3")
    input_b08 = c_s2.number_input("Band 08 (NIR)", value=0.35, key="b08_v3")
    input_b11 = c_s3.number_input("Band 11 (SWIR)", value=0.12, key="b11_v3")

    c_s4, c_s5, c_s6 = st.columns(3)
    input_b02 = c_s4.number_input("Band 02 (Blue)", value=0.05, key="b02_v3")
    input_b03 = c_s5.number_input("Band 03 (Green)", value=0.09, key="b03_v3")
    input_ndvi = c_s6.number_input("NDVI", value=(input_b08 - input_b04)/(input_b08 + input_b04 + 1e-6), key="ndvi_v3")

    c_s7, c_s8, c_s9 = st.columns(3)
    input_ndwi = c_s7.number_input("NDWI", value=(input_b03 - input_b08)/(input_b03 + input_b08 + 1e-6), key="ndwi_v3")
    input_mndwi = c_s8.number_input("MNDWI", value=(input_b03 - input_b11)/(input_b03 + input_b11 + 1e-6), key="mndwi_v3")
    input_ndbi = c_s9.number_input("NDBI", value=(input_b11 - input_b08)/(input_b11 + input_b08 + 1e-6), key="ndbi_v3")

    if st.button("🔮 Prediksi Tutupan Lahan", key="btn_pred_v3"):
        if model_knn is not None:
            df_single = pd.DataFrame([[input_b02, input_b03, input_b04, input_b08, input_b11, input_ndvi, input_ndwi, input_mndwi, input_ndbi]], columns=FITUR_LIST)
            pred_res = model_knn.predict(df_single)[0]
            pred_color = CLASS_COLORS.get(pred_res, '#FF6B00')
            st.markdown(f"""
            <div style="background-color: #FFF0E6; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B00;">
                <h4 style="color: #FF6B00; margin: 0;">Hasil Prediksi Model k-NN:</h4>
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
        <h4 style="color: #FF6B00; margin-top: 0;">2. Performa Model k-NN & Visualisasi Mismatch</h4>
        <ul>
            <li>Model <b>k-Nearest Neighbors (k-NN)</b> dengan <i>StandardScaler()</i> pada $k=5$ menghasilkan <b>Overall Accuracy 80.00%</b> dan <b>Cohen's Kappa Score 0.7500</b> pada data testing.</li>
            <li>Penandaan <b>Titik Mismatch Oranye (⚠️)</b> memudahkan inspeksi spasial langsung di atas citra satelit, di mana kesalahan prediksi paling sering terjadi pada area transisi perbatasan antara sawah dan permukiman padat.</li>
        </ul>
    </div>
    
    <div class="orange-card">
        <h4 style="color: #FF6B00; margin-top: 0;">3. Lokasi File Deployment</h4>
        <p>Aplikasi ini tersimpan di satu lokasi khusus deployment:</p>
        <ul>
            <li><b>File Utama Deploy:</b> <code>uts/deploy/app.py</code></li>
            <li><b>File Model k-NN:</b> <code>uts/models/model_lulc_knn.pkl</code></li>
            <li><b>File Data Spasial:</b> <code>uts/data/geojson/jawatimur_5kelas.geojson</code> & <code>uts/data/csv/poligon/centroid_poligon.csv</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
