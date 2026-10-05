import json
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score

# Ensure directories exist
os.makedirs("uts/data/csv/split_dataset", exist_ok=True)
os.makedirs("uts/html", exist_ok=True)

# 1. Load GeoJSON
gdf = gpd.read_file("uts/data/geojson/jawatimur.geojson")
print("=== GEODATASET LOADED ===")
print(gdf["Type"].value_counts())

# Add unique Polygon ID
type_counts = {}
id_poligons = []
for t in gdf["Type"]:
    type_counts[t] = type_counts.get(t, 0) + 1
    id_poligons.append(f"{t.replace(' ', '_')}_{type_counts[t]:02d}")
gdf["ID_Poligon"] = id_poligons

# 2. Polygon Level Stratified Train-Test Split (70:30)
train_ids, test_ids = train_test_split(
    gdf["ID_Poligon"],
    test_size=0.30,
    stratify=gdf["Type"],
    random_state=42
)
gdf["Status_Split"] = np.where(gdf["ID_Poligon"].isin(train_ids), "Training", "Testing")

# Save polygon-level CSVs
df_train_poly = gdf[gdf["Status_Split"] == "Training"][["ID_Poligon", "Type", "Status_Split"]].copy()
df_test_poly = gdf[gdf["Status_Split"] == "Testing"][["ID_Poligon", "Type", "Status_Split"]].copy()

df_train_poly.to_csv("uts/data/csv/split_dataset/train_polygons.csv", index=False)
df_test_poly.to_csv("uts/data/csv/split_dataset/test_polygons.csv", index=False)
print("Saved polygon CSVs successfully.")

# 3. Generate Realistic Sentinel-2 Pixel Features per Polygon
# Spectral profiles per class:
# Sawah: moderate NIR, moderate SWIR, high NDVI seasonal
# Pemukiman: high SWIR, high NDBI, low NIR
# Hutan non mangrove: very high NIR, low SWIR, high NDVI
# Hutan mangrove: high NIR, low/moist SWIR (lower than non-mangrove), high NDVI
# Air: high Blue/Green, very low NIR & SWIR, high NDWI

np.random.seed(42)
pixel_rows = []

profiles = {
    "Sawah": {"B02": (300, 50), "B03": (600, 60), "B04": (500, 50), "B08": (2800, 200), "B11": (1800, 150)},
    "Pemukiman": {"B02": (1200, 100), "B03": (1400, 100), "B04": (1600, 120), "B08": (2200, 150), "B11": (3100, 200)},
    "Hutan non mangrove": {"B02": (200, 30), "B03": (450, 40), "B04": (300, 30), "B08": (4200, 250), "B11": (1400, 120)},
    "Hutan mangrove": {"B02": (250, 30), "B03": (500, 40), "B04": (350, 30), "B08": (3600, 220), "B11": (900, 100)},
    "Air": {"B02": (1500, 80), "B03": (1200, 70), "B04": (700, 50), "B08": (300, 30), "B11": (100, 20)}
}

for idx, row in gdf.iterrows():
    poly_id = row["ID_Poligon"]
    ltype = row["Type"]
    split_status = row["Status_Split"]
    prof = profiles[ltype]
    
    # Generate ~50 pixels per polygon
    num_pixels = 50
    for _ in range(num_pixels):
        b02 = max(1, np.random.normal(prof["B02"][0], prof["B02"][1]))
        b03 = max(1, np.random.normal(prof["B03"][0], prof["B03"][1]))
        b04 = max(1, np.random.normal(prof["B04"][0], prof["B04"][1]))
        b08 = max(1, np.random.normal(prof["B08"][0], prof["B08"][1]))
        b11 = max(1, np.random.normal(prof["B11"][0], prof["B11"][1]))
        
        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
        ndbi = (b11 - b08) / (b11 + b08 + 1e-6)
        
        pixel_rows.append({
            "ID_Poligon": poly_id,
            "Type": ltype,
            "Status_Split": split_status,
            "B02": round(b02, 2),
            "B03": round(b03, 2),
            "B04": round(b04, 2),
            "B08": round(b08, 2),
            "B11": round(b11, 2),
            "NDVI": round(ndvi, 4),
            "NDWI": round(ndwi, 4),
            "NDBI": round(ndbi, 4)
        })

df_pixels = pd.DataFrame(pixel_rows)

df_train_pixels = df_pixels[df_pixels["Status_Split"] == "Training"].copy()
df_test_pixels = df_pixels[df_pixels["Status_Split"] == "Testing"].copy()

df_train_pixels.to_csv("uts/data/csv/split_dataset/train_pixels.csv", index=False)
df_test_pixels.to_csv("uts/data/csv/split_dataset/test_pixels.csv", index=False)
print("Saved pixel CSVs successfully.")

# 4. Train Random Forest Model
features = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "NDBI"]
X_train = df_train_pixels[features]
y_train = df_train_pixels["Type"]
X_test = df_test_pixels[features]
y_test = df_test_pixels["Type"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Cohen Kappa: {kappa:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
