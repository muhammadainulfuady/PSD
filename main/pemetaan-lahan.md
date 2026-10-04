# Pemetaan dan Klasifikasi Lahan Sawah dan Pemukiman Menggunakan Citra Sentinel-2A

Bagian ini mendokumentasikan seluruh alur pemetaan lahan, mulai dari penentuan area dan sampel berbasis **GeoJSON**, akuisisi citra satelit **Sentinel-2A (Level-2A)** berformat **GeoTIFF (`.tif`)** melalui **openEO** (Copernicus Data Space Ecosystem), ekstraksi fitur spektral, klasifikasi **Random Forest** untuk membedakan **Sawah** dan **Pemukiman**, validasi akurasi, hingga visualisasi hasil dalam bentuk peta 2D dan 3D.

---

## 1. Gambaran Umum Alur Kerja

| Tahap | Proses                                  | Keluaran                                                     |
| :---: | :-------------------------------------- | :----------------------------------------------------------- |
|   1   | Membaca GeoJSON (area studi dan sampel) | `GeoDataFrame` 101 poligon                                   |
|   2   | Mengunduh citra Sentinel-2A via openEO  | `sentinel2_area_studi.tif`                                   |
|   3   | Mengekstrak nilai spektral per poligon  | `fitur_sampel.csv`                                           |
|   4   | Melatih dan memvalidasi Random Forest   | Confusion matrix, Overall Accuracy, Kappa                    |
|   5   | Memprediksi seluruh area                | `klasifikasi_sawah_pemukiman.tif` dan `peta_klasifikasi.png` |
|   6   | Visualisasi 3D di atas relief (DEM)     | `peta_3d.html`                                               |

---

## 2. Data Masukan: GeoJSON

Seluruh data spasial berasal dari satu berkas, yaitu `data/geojson/input2.geojson`, yang berisi **101 fitur bertipe Polygon**. Atribut kelas disimpan pada kolom `Type`.

| Nilai `Type`  | Jumlah | Peran dalam Analisis                                                                                     |
| :------------ | :----: | :------------------------------------------------------------------------------------------------------- |
| **Daerah**    |   1    | Batas area studi (sekitar 24 km²). Dipakai sebagai _spatial extent_ pengunduhan citra, **bukan** sampel. |
| **Sawah**     |   50   | Sampel pelatihan dan pengujian kelas Sawah                                                               |
| **Pemukiman** |   50   | Sampel pelatihan dan pengujian kelas Pemukiman                                                           |

Beberapa hal yang perlu diperhatikan pada data sampel:

- Sampel berupa **poligon**, bukan titik. Satu poligon dapat mencakup banyak piksel Sentinel-2 (resolusi 10 m), sehingga jumlah data latih jauh lebih banyak dari 100.
- Ukuran poligon bervariasi. Poligon sawah terkecil sekitar 133 m² (kurang dari 2 piksel), sedangkan poligon pemukiman dapat mencapai sekitar 98.000 m². Akibatnya jumlah piksel kedua kelas tidak seimbang, dan hal ini ditangani pada tahap pemodelan.

```python
import openeo, json, os
import geopandas as gpd

gdf = gpd.read_file("../data/geojson/input2.geojson")
print(gdf["Type"].value_counts())

# Poligon "Daerah" dipakai sebagai batas area unduhan citra
daerah = gdf[gdf["Type"] == "Daerah"].geometry.iloc[0]
geom = json.loads(gpd.GeoSeries([daerah]).to_json())["features"][0]["geometry"]
```

---

## 3. Akuisisi Citra Sentinel-2A (`.tif`) melalui openEO

Citra diambil dari koleksi **`SENTINEL2_L2A`** (_Bottom-of-Atmosphere Reflectance_) pada platform Copernicus Data Space Ecosystem melalui openEO. Pemrosesan dilakukan di server, sehingga yang diunduh hanya hasil _clip_ area studi, bukan seluruh _tile_.

**Parameter akuisisi**

| Parameter         | Nilai                            | Keterangan                                  |
| :---------------- | :------------------------------- | :------------------------------------------ |
| Koleksi           | `SENTINEL2_L2A`                  | Reflektansi permukaan terkoreksi atmosfer   |
| Area              | Poligon `Daerah`                 | Berasal dari GeoJSON                        |
| Rentang waktu     | `2025-08-24` sampai `2026-08-23` | Satu tahun penuh untuk menekan tutupan awan |
| Filter awan       | `max_cloud_cover = 30`           | Seleksi tingkat scene                       |
| Band              | B02, B03, B04, B08, B11, SCL     | SCL hanya untuk masking awan                |
| Agregasi temporal | Median                           | Menghasilkan satu komposit bebas awan       |
| Format keluaran   | GeoTIFF                          | `sentinel2_area_studi.tif`                  |

**Masking awan.** Selain filter tingkat scene, dilakukan masking tingkat piksel menggunakan band **SCL** (_Scene Classification Layer_). Piksel dengan kelas 3 (bayangan awan), 8 dan 9 (awan probabilitas sedang dan tinggi), 10 (_cirrus_), dan 11 (salju/es) dikeluarkan sebelum median dihitung.

```python
conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

bands = ["B02", "B03", "B04", "B08", "B11", "SCL"]
cube = conn.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=geom,
    temporal_extent=["2025-08-24", "2026-08-23"],
    bands=bands,
    max_cloud_cover=30,
)

# Mask awan, bayangan awan, dan salju berdasarkan SCL
scl = cube.band("SCL")
mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
cube = cube.mask(mask.resample_cube_spatial(cube)).filter_bands(bands[:-1])

# Komposit median sepanjang dimensi waktu
cube = cube.reduce_dimension(dimension="t", reducer="median")

os.makedirs("../data/tif", exist_ok=True)
cube.download("../data/tif/sentinel2_area_studi.tif", format="GTiff")
```

---

## 4. Fitur Spektral dan Indeks Turunan

Model menggunakan **8 fitur** per piksel: 5 band reflektansi dan 3 indeks spektral. Nilai digital citra dibagi 10.000 untuk mengubahnya menjadi reflektansi.

| Fitur           | Panjang Gelombang | Kegunaan                                                 |
| :-------------- | :---------------- | :------------------------------------------------------- |
| `B02` (_Blue_)  | 490 nm            | Pembeda objek terang, bangunan, dan air                  |
| `B03` (_Green_) | 560 nm            | Pantulan hijau vegetasi, komponen NDWI                   |
| `B04` (_Red_)   | 665 nm            | Penyerapan klorofil, komponen NDVI                       |
| `B08` (_NIR_)   | 842 nm            | Pantulan kuat oleh vegetasi sehat                        |
| `B11` (_SWIR_)  | 1610 nm           | Sensitif terhadap kelembapan tanah dan material bangunan |

**Indeks yang dihitung**

1. _Normalized Difference Vegetation Index_ (kerapatan dan kesehatan vegetasi):

   $$\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$$

2. _Normalized Difference Water Index_ (kandungan air, relevan untuk fase genangan sawah):

   $$\text{NDWI} = \frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}}$$

3. _Normalized Difference Built-up Index_ (area terbangun):

   $$\text{NDBI} = \frac{\text{B11} - \text{B08}}{\text{B11} + \text{B08}}$$

---

## 5. Ekstraksi Nilai Spektral per Poligon

Untuk setiap poligon sampel, piksel yang berada di dalam poligon diambil menggunakan `rasterio.mask`. Parameter `all_touched=True` memastikan poligon kecil tetap menyumbang minimal satu piksel. Piksel yang tertutup awan atau tidak memiliki data (NaN atau bernilai 0) dibuang.

```python
import os, rasterio, numpy as np, pandas as pd
from rasterio.mask import mask as rio_mask

samples = gdf[gdf["Type"].isin(["Sawah", "Pemukiman"])].copy()

with rasterio.open("../data/tif/sentinel2_area_studi.tif") as src:
    samples = samples.to_crs(src.crs)
    rows = []
    for idx, r in samples.iterrows():
        m, _ = rio_mask(src, [r.geometry], crop=True, all_touched=True, filled=False)
        arr = m.astype("float32").filled(np.nan) / 10000   # reflektansi
        b02, b03, b04, b08, b11 = [arr[k].ravel() for k in range(5)]
        ok = ~np.isnan(b04) & ~np.isnan(b08) & ~np.isnan(b03) & ~np.isnan(b11)
        if ok.sum() == 0:
            continue
        # ... hitung NDVI, NDWI, NDBI lalu simpan rata-rata per poligon
```

Ringkasan rata-rata indeks per kelas:

| Kelas     |  NDVI  |  NDWI   |  NDBI   |
| :-------- | :----: | :-----: | :-----: |
| Sawah     | 0,3186 | -0,1987 | -0,2624 |
| Pemukiman | 0,2918 | -0,3310 | 0,0920  |

**Interpretasi.** Indeks yang paling jelas membedakan kedua kelas adalah **NDBI**: pemukiman bernilai positif (0,092) sedangkan sawah bernilai negatif (-0,262). NDWI sawah juga lebih tinggi daripada pemukiman (-0,199 dibanding -0,331), sesuai dengan sifat sawah yang lembap dan sering tergenang.

---

## 6. Klasifikasi Random Forest

### 6.1 Dataset Pelatihan

Pemodelan memakai **data per piksel**, bukan rata-rata per poligon. Total ekstraksi piksel dari 100 poligon sampel menghasilkan **12.037 piksel**:
- **Pemukiman**: 10.198 piksel
- **Sawah**: 1.839 piksel

### 6.2 Pembagian Data Latih dan Uji

Data dibagi **70% latih (9.418 piksel) dan 30% uji (2.619 piksel)** dengan `GroupShuffleSplit`, yaitu pembagian **per poligon** untuk menghindari _data leakage_.

### 6.3 Pelatihan Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             cohen_kappa_score, classification_report)

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr, te = next(gss.split(X, y, grp))

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(X[tr], y[tr])
pred = rf.predict(X[te])

labels = ["Sawah", "Pemukiman"]
print("Confusion matrix (baris=aktual, kolom=prediksi):")
print(confusion_matrix(y[te], pred, labels=labels))
print("Overall Accuracy:", round(accuracy_score(y[te], pred), 4))
print("Kappa           :", round(cohen_kappa_score(y[te], pred), 4))
print(classification_report(y[te], pred))
```

---

## 7. Hasil Validasi dan Evaluasi Model

Validasi dilakukan pada data uji (30% poligon uji = 2.619 piksel).

**Confusion Matrix** (baris = aktual, kolom = prediksi):

|                      | Prediksi Sawah | Prediksi Pemukiman |
| :------------------- | :------------: | :----------------: |
| **Aktual Sawah**     |    **888**     |         12         |
| **Aktual Pemukiman** |     **0**      |      **1.719**     |

**Metrik Akurasi**

| Metrik                | Nilai    | Penjelasan                                                     |
| :-------------------- | :------: | :------------------------------------------------------------- |
| **Overall Accuracy**  | **99,54%** | 2.607 dari 2.619 piksel uji berhasil diklasifikasi dengan benar |
| **Cohen's Kappa**     | **0,9898** | Menunjukkan tingkat kesepakatan yang sangat tinggi             |
| **Precision Sawah**   | 1.00     | Seluruh piksel yang diprediksi Sawah adalah benar-benar Sawah  |
| **Recall Sawah**      | 0.99     | 99% piksel Sawah berhasil terdeteksi                          |
| **F1-Score Sawah**    | 0.99     | Keseimbangan sensitivitas dan presisi untuk kelas Sawah        |
| **Precision Pemukiman**| 0.99    | Presisi tinggi untuk kelas Pemukiman                           |
| **Recall Pemukiman**  | 1.00     | Seluruh piksel Pemukiman berhasil terdeteksi secara sempurna   |

**Urutan Kepentingan Fitur (_Feature Importance_):**

1. `B11` (_SWIR_): **0.368** (Fitur paling dominan membedakan kelembapan & bahan bangunan)
2. `B04` (_Red_): **0.176**
3. `NDBI` (_Built-up Index_): **0.161**
4. `B08` (_NIR_): **0.111**
5. `B02` (_Blue_): **0.098**
6. `NDWI` (_Water Index_): **0.037**
7. `B03` (_Green_): **0.031**
8. `NDVI` (_Vegetation Index_): **0.017**

---

## 8. Visualisasi Peta 2D Klasifikasi Tutupan Lahan

Peta 2D hasil pemetaan tutupan lahan membedakan area **Sawah** dan **Pemukiman** di area studi berdasarkan reflektansi spektral Sentinel-2A dan model Random Forest:

![Peta 2D Klasifikasi Tutupan Lahan Sawah dan Pemukiman](../assets/peta_klasifikasi.png)

---

## 9. Visualisasi Peta 3D Interaktif

Peta hasil klasifikasi tutupan lahan dipetakan ke atas relief **Copernicus DEM (30 m)** untuk menghasilkan peta permukaan 3D interaktif yang dapat diputar, di-zoom, dan digeser langsung di browser.

```{raw} html
<iframe src="../assets/editor/html/peta_3d.html" width="100%" height="750px" style="border:none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

---

## 10. Notebook Interaktif & Berkas Keluaran

Seluruh eksperimen dan eksekusi kode dapat diakses melalui berkas notebook berikut:

- [code-pemetaanLahanKlasifikasi.ipynb](code-pemetaanLahanKlasifikasi.ipynb)
- [code-klasifikasiLahan.ipynb](code-klasifikasiLahan.ipynb)

**Berkas Keluaran Proyek**

| Berkas                                     | Isi                                                     |
| :----------------------------------------- | :------------------------------------------------------ |
| `data/tif/sentinel2_area_studi.tif`        | Citra Sentinel-2A komposit (B02, B03, B04, B08, B11)    |
| `data/tif/klasifikasi_sawah_pemukiman.tif` | Peta klasifikasi (0 = nodata, 1 = Sawah, 2 = Pemukiman) |
| `data/tif/dem_area_studi.tif`              | Model elevasi digital area studi                        |
| `data/csv/fitur_sampel.csv`                | Nilai fitur rata-rata per poligon sampel                |
| `assets/peta_klasifikasi.png`              | Peta 2D klasifikasi tutupan lahan                       |
| `assets/editor/html/peta_3d.html`          | Peta 3D interaktif                                      |
