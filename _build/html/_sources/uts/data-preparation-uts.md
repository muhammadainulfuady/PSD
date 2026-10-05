# Data Preparation — Klasifikasi Land Use and Land Cover (LULC) Jawa Timur

Dokumen ini menjelaskan tahap **Data Preparation** dalam metodologi CRISP-DM untuk proyek UTS LULC Jawa Timur. Tahap ini mencakup ekstraksi fitur spektral, kalkulasi indeks spektral, pembersihan data, serta pembagian dataset (_train-test split_) berbasis polygon ID untuk menghindari _data leakage_.

---

## 3. Data Preparation

### 3.1 Ekstraksi Nilai Spektral & Rekayasa Fitur (Feature Engineering)

Data masukan (_input features_) diambil dari piksel citra satelit Sentinel-2 yang jatuh di dalam poligon sampel.

#### Fitur Spektral yang Diekstrak:

1. **Raw Bands (5 Fitur):**
   - `B02` (Blue)
   - `B03` (Green)
   - `B04` (Red)
   - `B08` (Near-Infrared / NIR)
   - `B11` (Shortwave-Infrared / SWIR1)

2. **Indeks Spektral Kunci (3 Fitur Turunan):**
   - `NDVI` $= \frac{B08 - B04}{B08 + B04}$ (Indeks Kerapatan Vegetasi)
   - `NDWI` $= \frac{B03 - B08}{B03 + B08}$ (Indeks Kebasahan Air)
   - `NDBI` $= \frac{B11 - B08}{B11 + B08}$ (Indeks Area Terbangun / Pemukiman)

---

### 3.2 Pembagian Data Train-Test Berbasis Polygon (Polygon-Level Split)

Untuk mencegah kebocoran data (_spatial autocorrelation data leakage_), pembagian data latih (70%) dan data uji (30%) dilakukan di tingkat **Polygon ID**, bukan di tingkat piksel individu.

- **Aturan Split:** Stratified Group Split berdasarkan `polygon_id` dan kelas `Type`.
- **Proporsi Split:**
  - **Data Latih (Train Set):** 70% polygon (35 polygon per kelas = total 175 polygon).
  - **Data Uji (Test Set):** 30% polygon (15 polygon per kelas = total 75 polygon).

> Piksel dari polygon yang sama **tidak pernah** dipisah ke dalam data latih dan data uji secara bersamaan. Hal ini menjamin evaluasi akurasi model benar-benar objektif pada wilayah yang belum pernah dilihat model (_unseen spatial area_).

---

### 3.3 Ekspor & Struktur Storage CSV

Hasil pra-pemrosesan data diekspor ke dalam folder `uts/data/csv/split_dataset/`:

1. `train_polygons.csv` — Daftar ID polygon latih.
2. `test_polygons.csv` — Daftar ID polygon uji.
3. `train_pixels.csv` — Dataset tingkat piksel untuk pelatihan model Machine Learning.
4. `test_pixels.csv` — Dataset tingkat piksel untuk evaluasi performa model.


