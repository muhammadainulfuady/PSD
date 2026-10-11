# Data Understanding

## 1. Deskripsi Data
Analisis klasifikasi LULC memanfaatkan dua sumber data utama:
1. **Data Vektor Sampel (Ground Truth):** Data spasial poligon terverifikasi bertipe geometri *Polygon* dengan sistem koordinat EPSG:4326 (WGS84).
2. **Data Raster Satelit Sentinel-2 (Level-2A):** Citra satelit optik multiband (*Surface Reflectance*) dengan resolusi spasial 10 m hingga 20 m.

## 2. Karakteristik Sampel Lahan
Data sampel terdiri dari **250 poligon** yang terbagi secara seimbang ke dalam 5 kelas tutupan lahan (masing-masing 50 sampel poligon):

| No | Kelas Tutupan Lahan | Jumlah Sampel | Persentase | Tipe Geometri |
| :-: | :--- | :-: | :-: | :-: |
| 1 | Air | 50 | 20% | Polygon |
| 2 | Hutan Mangrove | 50 | 20% | Polygon |
| 3 | Hutan Non-Mangrove | 50 | 20% | Polygon |
| 4 | Pemukiman | 50 | 20% | Polygon |
| 5 | Sawah | 50 | 20% | Polygon |
| **Total** | **5 Kelas** | **250** | **100%** | **Polygon** |

## 3. Spesifikasi Band Satelit Sentinel-2
Digunakan 5 band spektral kunci dari sensor MSI Sentinel-2 untuk klasifikasi:

| Band Satelit | Nama Spektrum | Panjang Gelombang | Resolusi | Peran Utama |
| :--- | :--- | :---: | :---: | :--- |
| **Band 2 (B02)** | Blue | $\sim 490\text{ nm}$ | 10 m | Komposisi warna alami dan analisis perairan. |
| **Band 3 (B03)** | Green | $\sim 560\text{ nm}$ | 10 m | Pantulan klorofil dan komponen perhitungan NDWI. |
| **Band 4 (B04)** | Red | $\sim 665\text{ nm}$ | 10 m | Penyerapan klorofil vegetasi dan komponen NDVI. |
| **Band 8 (B08)** | NIR (Near-Infrared) | $\sim 842\text{ nm}$ | 10 m | Membedakan kerapatan vegetasi dari air/bangunan. |
| **Band 11 (B11)** | SWIR1 (Shortwave-Infrared) | $\sim 1610\text{ nm}$ | 20 m | Peka kelembapan tanah/kanopi dan membedakan mangrove vs non-mangrove serta pemukiman. |

## 4. Indeks Spektral (Feature Engineering)
Untuk memperkuat pemisahan antar kelas, dihitung 3 indeks spektral utama:

### 4.1 Normalized Difference Vegetation Index (NDVI)
$$\text{NDVI} = \frac{\text{B08 (NIR)} - \text{B04 (Red)}}{\text{B08 (NIR)} + \text{B04 (Red)}}$$

Mengukur kerapatan dan tingkat kehijauan vegetasi.

### 4.2 Normalized Difference Water Index (NDWI)
$$\text{NDWI} = \frac{\text{B03 (Green)} - \text{B08 (NIR)}}{\text{B03 (Green)} + \text{B08 (NIR)}}$$

Mengisolasi dan mengidentifikasi badan air.

### 4.3 Normalized Difference Built-up Index (NDBI)
$$\text{NDBI} = \frac{\text{B11 (SWIR1)} - \text{B08 (NIR)}}{\text{B11 (SWIR1)} + \text{B08 (NIR)}}$$

Mengidentifikasi area terbangun dan pemukiman.

## 5. Respon Spektral per Kelas

| Kelas | Respon NIR (B08) | Respon SWIR (B11) | NDVI | NDWI | NDBI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Air** | Sangat Rendah | Sangat Rendah | Negatif | Positif Tinggi | Negatif |
| **Hutan Mangrove** | Tinggi | Sedang (Lembap) | Positif Tinggi | Sedang | Negatif / Rendah |
| **Hutan Non-Mangrove** | Sangat Tinggi | Rendah / Sedang | Positif Sangat Tinggi | Negatif | Negatif |
| **Pemukiman** | Sedang | Sangat Tinggi | Rendah | Negatif | Positif |
| **Sawah** | Bervariasi | Bervariasi | Positif Sedang | Bervariasi | Bervariasi |


