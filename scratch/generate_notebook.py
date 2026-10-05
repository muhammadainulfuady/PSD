import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown Header
cell1_md = """# Klasifikasi Land Use and Land Cover (LULC) Jawa Timur (5 Kelas)
## Modeling & Evaluasi Random Forest Classifier Berbasis Citra Satelit Sentinel-2

Notebook ini berisi alur kerja pemodelan Machine Learning **Random Forest Classifier** untuk mengklasifikasikan 5 kelas tutupan lahan di Provinsi Jawa Timur:
1. **Sawah / Pertanian**
2. **Pemukiman / Built-Up**
3. **Hutan Non-Mangrove**
4. **Hutan Mangrove**
5. **Air (Water Bodies)**

---

### Transparansi Pembagian Dataset (Data Training & Testing)
Untuk mencegah kebocoran data (*spatial autocorrelation data leakage*), pembagian dataset sebesar **70% Training** dan **30% Testing** dilakukan di tingkat **Poligon Sampel (Group-based Split)**.

Berkas CSV hasil pembagian diekspor ke direktori `uts/data/csv/split_dataset/`:
- `train_polygons.csv` (175 Poligon)
- `test_polygons.csv` (75 Poligon)
- `train_pixels.csv` (8.750 Piksel Data Latih)
- `test_pixels.csv` (3.750 Piksel Data Uji)
"""

# Cell 2: Code 1 - Load GeoJSON & Polygon-Level Split
cell2_code = """import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Membaca Data GeoJSON (jawatimur.geojson - 250 Poligon)
gdf = gpd.read_file("data/geojson/jawatimur.geojson")
print("=== TOTAL POLIGON GEODATASET (5 KELAS) ===")
print(gdf["Type"].value_counts())

# Memberikan ID Poligon Unik
type_counts = {}
id_poligons = []
for t in gdf["Type"]:
    type_counts[t] = type_counts.get(t, 0) + 1
    id_poligons.append(f"{t.replace(' ', '_')}_{type_counts[t]:02d}")
gdf["ID_Poligon"] = id_poligons

# 2. Polygon-Level Stratified Split (70% Training, 30% Testing)
train_ids, test_ids = train_test_split(
    gdf["ID_Poligon"],
    test_size=0.30,
    stratify=gdf["Type"],
    random_state=42
)
gdf["Status_Split"] = np.where(gdf["ID_Poligon"].isin(train_ids), "Training", "Testing")

# 3. Ekspor CSV Poligon
df_train_poly = gdf[gdf["Status_Split"] == "Training"][["ID_Poligon", "Type", "Status_Split"]].copy()
df_test_poly = gdf[gdf["Status_Split"] == "Testing"][["ID_Poligon", "Type", "Status_Split"]].copy()

df_train_poly.to_csv("data/csv/split_dataset/train_polygons.csv", index=False)
df_test_poly.to_csv("data/csv/split_dataset/test_polygons.csv", index=False)

print("\\n=== RINGKASAN PEMBAGIAN POLIGON (TRAIN vs TEST) ===")
print(pd.crosstab(gdf["Type"], gdf["Status_Split"], margins=True))
print("\\n[SUCCESS]: File CSV Poligon Latih & Uji disimpan di 'data/csv/split_dataset/'")
"""

# Cell 3: Code 2 - Read/Generate Pixel Data & Compute Spectral Indices
cell3_code = """# Membaca Data Fitur Piksel Spektral
df_train_pixels = pd.read_csv("data/csv/split_dataset/train_pixels.csv")
df_test_pixels = pd.read_csv("data/csv/split_dataset/test_pixels.csv")

features = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "NDBI"]

X_train = df_train_pixels[features]
y_train = df_train_pixels["Type"]

X_test = df_test_pixels[features]
y_test = df_test_pixels["Type"]

print(f"Jumlah Piksel Training: {len(X_train):,} piksel")
print(f"Jumlah Piksel Testing : {len(X_test):,} piksel")
print("\\n=== SAMPLE FITUR DATA SPEKTRAL ===")
print(X_train.head())
"""

# Cell 4: Code 3 - Train Random Forest Classifier
cell4_code = """from sklearn.ensemble import RandomForestClassifier

# Pelatihan Model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("[SUCCESS]: Random Forest Classifier berhasil dilatih pada 8.750 piksel data latih.")
"""

# Cell 5: Code 4 - Evaluation (Confusion Matrix & Classification Report)
cell5_code = """from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Prediksi pada Data Uji (Unseen Data)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print("=== METRIK EVALUASI MODEL ===")
print(f"Overall Accuracy : {accuracy * 100:.2f}%")
print(f"Cohen's Kappa    : {kappa:.4f}\\n")

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Visualisasi Confusion Matrix Heatmap
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix — Klasifikasi LULC Jawa Timur (5 Kelas)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()
"""

# Cell 6: Code 5 - Feature Importances
cell6_code = """# Visualisasi Tingkat Kepentingan Fitur (Feature Importances)
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="forestgreen")
plt.title("Tingkat Kepentingan Fitur Spektral (Feature Importances)")
plt.xlabel("Relative Importance Score")
plt.ylabel("Spectral Band / Index")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(cell1_md),
    nbf.v4.new_code_cell(cell2_code),
    nbf.v4.new_code_cell(cell3_code),
    nbf.v4.new_code_cell(cell4_code),
    nbf.v4.new_code_cell(cell5_code),
    nbf.v4.new_code_cell(cell6_code)
]

with open("uts/code-uts.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("uts/code-uts.ipynb regenerated successfully.")
