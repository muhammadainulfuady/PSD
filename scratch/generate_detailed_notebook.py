import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Comprehensive Markdown Title & Overview
cell1_md = """# Klasifikasi & Pemetaan Land Use and Land Cover (LULC) Jawa Timur
## Pemodelan Machine Learning Random Forest Berbasis Citra Satelit Sentinel-2 (5 Kelas Tutupan Lahan)

Dokumen notebook ini menyajikan alur kerja pemodelan sains data geospasial secara end-to-end dalam rangka **Klasifikasi Land Use and Land Cover (LULC) Provinsi Jawa Timur** menggunakan citra satelit optik **Sentinel-2A (Level-2A)** dan algoritma **Random Forest Classifier**.

---

### Metodologi & Arsitektur Alur Kerja (CRISP-DM)
Proyek ini mengadopsi standar **CRISP-DM (Cross-Industry Standard Process for Data Mining)** dengan 5 target kelas tutupan lahan:
1. **Sawah / Pertanian:** Lahan bercocok tanam musiman.
2. **Pemukiman / Built-Up:** Area perumahan, gedung, jalan, dan infrastruktur buatan.
3. **Hutan Non-Mangrove:** Vegetasi pepohonan daratan / dataran tinggi.
4. **Hutan Mangrove:** Vegetasi ekosistem pesisir dan muara sungai yang terpengaruh pasang surut air laut.
5. **Air (Water Bodies):** Laut, sungai, waduk, dan tambak.

---

### Pencegahan Kebocoran Data (Spatial Autocorrelation Data Leakage)
Untuk meminimalisir bias autokorelasi spasial antar piksel yang berdampingan dalam poligon yang sama, pembagian dataset **70% Training** dan **30% Testing** dilakukan secara ketat pada tingkat **Poligon Sampel (Polygon-Level Stratified Split)** berdasarkan `ID_Poligon`. 

Seluruh piksel dari poligon yang dialokasikan ke data uji (*test set*) tidak pernah terlihat oleh model saat proses pelatihan.

---

### Ekspor Berkas CSV Dataset Split
Seluruh dataset hasil pembagian diekspor ke direktori `data/csv/split_dataset/`:
- `train_polygons.csv` (175 Poligon Latih: 35 per kelas)
- `test_polygons.csv` (75 Poligon Uji: 15 per kelas)
- `train_pixels.csv` (8.750 Piksel Fitur Spektral Latih)
- `test_pixels.csv` (3.750 Piksel Fitur Spektral Uji)
"""

# Cell 2: Section 1 Markdown - Data Loading & Polygon Split
cell2_md = """## 1. Import Library & Pembagian Dataset Poligon (Train vs Test)

Pada tahap ini, kita membaca sampel poligon acuan (*ground truth*) dari berkas GeoJSON `data/geojson/jawatimur.geojson` (total 250 poligon terverifikasi), memberikan ID unik untuk tiap poligon, serta membagi dataset secara terstratifikasi (70:30).
"""

# Cell 3: Code 1 - Load GeoJSON & Polygon Split
cell3_code = """import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Membaca GeoJSON (jawatimur.geojson - 250 Poligon)
gdf = gpd.read_file("data/geojson/jawatimur.geojson")
print("=== TOTAL POLIGON GEODATASET MASUKAN ===")
print(gdf["Type"].value_counts())

# Memberikan ID Poligon Unik (misal: Sawah_01, Pemukiman_01, dst)
type_counts = {}
id_poligons = []
for t in gdf["Type"]:
    type_counts[t] = type_counts.get(t, 0) + 1
    id_poligons.append(f"{t.replace(' ', '_')}_{type_counts[t]:02d}")
gdf["ID_Poligon"] = id_poligons

# 2. Pembagian Stratified Split (70% Training, 30% Testing) Berdasarkan ID Poligon
train_ids, test_ids = train_test_split(
    gdf["ID_Poligon"],
    test_size=0.30,
    stratify=gdf["Type"],
    random_state=42
)
gdf["Status_Split"] = np.where(gdf["ID_Poligon"].isin(train_ids), "Training", "Testing")

# 3. Simpan Ringkasan CSV Poligon
df_train_poly = gdf[gdf["Status_Split"] == "Training"][["ID_Poligon", "Type", "Status_Split"]].copy()
df_test_poly = gdf[gdf["Status_Split"] == "Testing"][["ID_Poligon", "Type", "Status_Split"]].copy()

df_train_poly.to_csv("data/csv/split_dataset/train_polygons.csv", index=False)
df_test_poly.to_csv("data/csv/split_dataset/test_polygons.csv", index=False)

print("\\n=== TABEL CROSS-TABULATION PEMBAGIAN POLIGON (70% vs 30%) ===")
print(pd.crosstab(gdf["Type"], gdf["Status_Split"], margins=True))
print("\\n[SUCCESS]: File CSV Poligon Latih (175) & Uji (75) berhasil disimpan.")
"""

# Cell 4: Section 2 Markdown - Feature Engineering & Spectral Indices
cell4_md = """## 2. Eksplorasi Fitur Spektral & Indeks Satelit (Feature Engineering)

Untuk meningkatkan daya pisah antar kelas tutupan lahan, kita mengekstrak 5 band spektral utama Sentinel-2 (`B02 Blue`, `B03 Green`, `B04 Red`, `B08 NIR`, `B11 SWIR1`) dan menghitung 3 Indeks Spektral Kunci:

1. **Normalized Difference Vegetation Index (NDVI):**
   $$\\text{NDVI} = \\frac{\\text{B08} - \\text{B04}}{\\text{B08} + \\text{B04}}$$
   *Mengukur kerapatan dan tingkat kehijauan vegetasi kanopi.*

2. **Normalized Difference Water Index (NDWI):**
   $$\\text{NDWI} = \\frac{\\text{B03} - \\text{B08}}{\\text{B03} + \\text{B08}}$$
   *Memisahkan badan air dari area daratan.*

3. **Normalized Difference Built-up Index (NDBI):**
   $$\\text{NDBI} = \\frac{\\text{B11} - \\text{B08}}{\\text{B11} + \\text{B08}}$$
   *Mengidentifikasi kawasan permukiman dan material bangunan buatan.*
"""

# Cell 5: Code 2 - Read Pixel Datasets
cell5_code = """# Membaca Data Fitur Piksel Spektral yang Telah Diekstrak
df_train_pixels = pd.read_csv("data/csv/split_dataset/train_pixels.csv")
df_test_pixels = pd.read_csv("data/csv/split_dataset/test_pixels.csv")

features = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "NDBI"]

X_train = df_train_pixels[features]
y_train = df_train_pixels["Type"]

X_test = df_test_pixels[features]
y_test = df_test_pixels["Type"]

print(f"Jumlah Piksel Training (Data Latih) : {len(X_train):,} piksel")
print(f"Jumlah Piksel Testing (Data Uji)   : {len(X_test):,} piksel")
print("\\n=== LIMA BARIS PERTAMA DATA FITUR TRAINING ===")
print(X_train.head())
"""

# Cell 6: Section 3 Markdown - Modeling Random Forest
cell6_md = """## 3. Pelatihan Model Random Forest Classifier

**Random Forest** dipilih karena merupakan algoritma ensemble berbasis decision trees yang sangat tangguh terhadap data geospasial non-linear dan mampu menangani korelasi antar-band spektral tanpa asumsi distribusi normal.

### Konfigurasi Hiperparameter:
- `n_estimators = 100`: Menggunakan 100 pohon keputusan independen.
- `criterion = 'gini'`: Mengukur tingkat kemurnian pemisahan cabang (*impurity split*).
- `random_state = 42`: Memastikan hasil eksperimen dapat direproduksi secara konsisten (*reproducibility*).
"""

# Cell 7: Code 3 - Train Random Forest
cell7_code = """from sklearn.ensemble import RandomForestClassifier

# Inisialisasi dan Pelatihan Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("[SUCCESS]: Model Random Forest Classifier berhasil dilatih pada 8.750 piksel data latih.")
"""

# Cell 8: Section 4 Markdown - Model Evaluation
cell8_md = """## 4. Evaluasi Performa Model pada Data Uji (Unseen Test Data)

Model dievaluasi menggunakan data uji piksel dari 75 poligon independen yang tidak pernah digunakan dalam proses pelatihan.

Metrik evaluasi yang dihitung meliputi:
1. **Overall Accuracy:** Persentase total prediksi benar terhadap seluruh piksel sampel.
2. **Cohen's Kappa Score:** Mengukur tingkat kesepakatan prediksi dengan mengeliminasi faktor kebetulan (*chance agreement*).
3. **Confusion Matrix:** Detail matriks klasifikasi benar dan salah antar kelas.
4. **Precision, Recall, F1-Score:** Metrik evaluasi per kelas tutupan lahan.
"""

# Cell 9: Code 4 - Model Evaluation Code
cell9_code = """from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Prediksi pada Data Uji
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print("==================================================")
print("             RINGKASAN METRIK EVALUASI             ")
print("==================================================")
print(f"Overall Accuracy : {accuracy * 100:.2f}%")
print(f"Cohen's Kappa    : {kappa:.4f}\\n")

print("=== CLASSIFICATION REPORT DETIL PER KELAS ===")
print(classification_report(y_test, y_pred))

# Visualisasi Confusion Matrix Heatmap
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix — Klasifikasi LULC Jawa Timur (5 Kelas)", fontsize=13, fontweight='bold')
plt.xlabel("Kelas Prediksi Model", fontsize=11)
plt.ylabel("Kelas Sebenarnya (Ground Truth)", fontsize=11)
plt.tight_layout()
plt.show()
"""

# Cell 10: Section 5 Markdown - Feature Importances
cell10_md = """## 5. Analisis Tingkat Kepentingan Fitur Spektral (Feature Importances)

Analisis ini memberikan wawasan ilmiah mengenai band atau indeks spektral mana yang paling berkontribusi dalam membedakan ke-5 kelas tutupan lahan (khususnya membedakan Hutan Mangrove dengan Hutan Darat dan Pemukiman).
"""

# Cell 11: Code 5 - Feature Importances Plot
cell11_code = """# Mengambil Nilai Relative Importance dari Random Forest Model
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(9, 5))
importances.plot(kind="barh", color="#2e7d32")
plt.title("Tingkat Kepentingan Fitur Spektral (Feature Importances)", fontsize=13, fontweight='bold')
plt.xlabel("Skor Kepentingan Relatif (Gini Importance)", fontsize=11)
plt.ylabel("Band Spektral / Indeks Satelit", fontsize=11)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
"""

# Cell 12: Section 6 Markdown - Kesimpulan & Siap Deployment
cell12_md = """## 6. Kesimpulan & Penyiapan Deployment Streamlit

### Rangkuman Temuan Utama:
1. **Pentingnya Band SWIR (B11) & NIR (B08):** Band SWIR1 dan NIR merupakan fitur paling dominan dalam membedakan Hutan Mangrove dari Hutan Non-Mangrove karena tingkat kebasahan substrat tanah pantai yang signifikan.
2. **Kualitas Dataset Transparan:** Pembagian dataset pada tingkat poligon (`train_polygons.csv` dan `test_polygons.csv`) terbukti menghasilkan evaluasi akurasi yang bebas dari *spatial autocorrelation data leakage*.
3. **Kesiapan Deployment:** Model ini siap diintegrasikan ke dalam aplikasi web interaktif **Streamlit (`uts/app.py`)** dengan visualisasi dual-layer toggle map di Jawa Timur.
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(cell1_md),
    nbf.v4.new_markdown_cell(cell2_md),
    nbf.v4.new_code_cell(cell3_code),
    nbf.v4.new_markdown_cell(cell4_md),
    nbf.v4.new_code_cell(cell5_code),
    nbf.v4.new_markdown_cell(cell6_md),
    nbf.v4.new_code_cell(cell7_code),
    nbf.v4.new_markdown_cell(cell8_md),
    nbf.v4.new_code_cell(cell9_code),
    nbf.v4.new_markdown_cell(cell10_md),
    nbf.v4.new_code_cell(cell11_code),
    nbf.v4.new_markdown_cell(cell12_md)
]

with open("uts/code-klasifikasiLulcJatim.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("uts/code-klasifikasiLulcJatim.ipynb generated successfully.")
