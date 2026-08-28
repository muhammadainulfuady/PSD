# Data Understanding

Data Understanding adalah tahap untuk **mengumpulkan**, **mengeksplorasi**, dan **menilai kualitas** data yang akan digunakan dalam analisis kualitas udara di Kabupaten Gresik.

---

## 1. Collecting (Mengumpulkan Data)

### 1.1 Sumber Data

Data kualitas udara dikumpulkan dari **Copernicus Data Space** menggunakan layanan **openEO**. Berikut ringkasannya:

| Item | Keterangan |
|------|------------|
| **Sumber** | Copernicus Data Space |
| **Layanan** | openEO |
| **Server** | `openeo.dataspace.copernicus.eu` |
| **Produk / Koleksi** | Sentinel-5P L2 |
| **Polutan** | NO2, CO, SO2, CH4 |
| **Perioda** | 24 Agustus 2025 – 24 Agustus 2026 |
| **Lokasi** | Kabupaten Gresik, Jawa Timur |

### 1.2 Koneksi dan Otentikasi

Langkah pertama adalah menghubungkan ke server openEO dan melakukan otentikasi menggunakan akun Copernicus Data Space.

```python
import openeo

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
```

Saat dijalankan, akan muncul tautan untuk login (menggunakan *device code flow*). Setelah berhasil, tampilan pada openEO Web Editor akan seperti gambar berikut:

```{figure} ../assets/editor/openeo.png
:width: 100%
:align: center

Batch job NO2 Gresik yang telah selesai (status *finished*) pada openEO Web Editor.
```

### 1.3 Penentuan Area of Interest (AOI)

**Lokasi pengamatan** dibatasi pada area di **Kabupaten Gresik**. Area ini digambar sebagai **polygon** di atas peta menggunakan alat seperti [geojson.io](https://geojson.io).

```{figure} ../assets/data/geojson2.png
:width: 100%
:align: center

Polygon Area of Interest (AOI) di Kabupaten Gresik.

```

**Penjelasan koordinat:**

Setiap titik pada polygon memiliki format `[longitude (bujur), latitude (lintang)]`. Dari polygon tersebut, kita memperoleh:

| Atribut | Nilai | Keterangan |
|---------|-------|------------|
| `west` | 112.6409353 | Longitude terkecil (batas kiri) |
| `east` | 112.6589794 | Longitude terbesar (batas kanan) |
| `south` | -7.177427 | Latitude terkecil (batas bawah) |
| `north` | -7.1491012 | Latitude terbesar (batas atas) |

> **Catatan:** Untuk data Sentinel-5P, `spatial_extent` pada `load_collection` menggunakan *bounding box* (kotak batas) yang dibentuk oleh `west`, `south`, `east`, `north`. Sedangkan `aoi` (polygon) digunakan pada tahap `aggregate_spatial` untuk menghitung rata-rata di dalam area tersebut.

### 1.4 Memuat Data (Load Collection)

Data dimuat **per polutan**, karena server Sentinel-5P di openEO hanya mendukung **satu band per proses**.

#### a. Memuat Data NO2

```python
s5_no2 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={"west": 112.6409353,
                    "south": -7.177427,
                    "east": 112.6589794,
                    "north": -7.1491012},
    bands=["NO2"],
)
```

#### b. Memuat Data CO

```python
s5_co = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={"west": 112.6409353,
                    "south": -7.177427,
                    "east": 112.6589794,
                    "north": -7.1491012},
    bands=["CO"],
)
```

#### c. Memuat Data SO2

```python
s5_so2 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={"west": 112.6409353,
                    "south": -7.177427,
                    "east": 112.6589794,
                    "north": -7.1491012},
    bands=["SO2"],
)
```

#### d. Memuat Data CH4

```python
s5_ch4 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={"west": 112.6409353,
                    "south": -7.177427,
                    "east": 112.6589794,
                    "north": -7.1491012},
    bands=["CH4"],
)
```

**Perhatikan:** Kode untuk CO, SO2, dan CH4 **identik** dengan NO2 — hanya berbeda pada parameter `bands` (dan variabel `s5_co`, `s5_so2`, `s5_ch4`).

### 1.5 Definisi AOI (Polygon)

```python
aoi = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [112.6411372, -7.1491012],   # kiri-atas (NW)
                    [112.6589794, -7.1491012],   # kanan-atas (NE)
                    [112.6589794, -7.177427],    # kanan-bawah (SE)
                    [112.6409353, -7.177427],    # kiri-bawah (SW)
                    [112.6411372, -7.1491012]    # kembali ke titik awal
                ]]
            }
        }
    ]
}
```

### 1.6 Agregasi Data

Setiap datacube diagregasi menjadi **rata-rata harian**, lalu dihitung **rata-rata spasial** di dalam polygon AOI untuk menghasilkan deret waktu (time series).

```python
# Agregasi untuk NO2
s5_no2 = s5_no2.aggregate_temporal_period(reducer="mean", period="day")
s5_no2 = s5_no2.aggregate_spatial(reducer="mean", geometries=aoi)

# Agregasi untuk CO
s5_co = s5_co.aggregate_temporal_period(reducer="mean", period="day")
s5_co = s5_co.aggregate_spatial(reducer="mean", geometries=aoi)

# Agregasi untuk SO2
s5_so2 = s5_so2.aggregate_temporal_period(reducer="mean", period="day")
s5_so2 = s5_so2.aggregate_spatial(reducer="mean", geometries=aoi)

# Agregasi untuk CH4
s5_ch4 = s5_ch4.aggregate_temporal_period(reducer="mean", period="day")
s5_ch4 = s5_ch4.aggregate_spatial(reducer="mean", geometries=aoi)
```

### 1.7 Menjalankan Batch Job

Proses dijalankan sebagai **batch job** di server openEO. Hasilnya diunduh sebagai file netCDF (`.nc`).

```python
# Jalankan batch job dan unduh hasil
s5_no2.execute_batch(title="NO2 Gresik", outputfile="polutan_no2_gresik.nc")
s5_co.execute_batch(title="CO Gresik", outputfile="polutan_co_gresik.nc")
s5_so2.execute_batch(title="SO2 Gresik", outputfile="polutan_so2_gresik.nc")
s5_ch4.execute_batch(title="CH4 Gresik", outputfile="polutan_ch4_gresik.nc")
```

Setiap batch job membutuhkan beberapa menit (antre di server). Setelah selesai, file `.nc` tersimpan di folder kerja.

---

## 2. (Lanjutan)

Bagian eksplorasi data, visualisasi peta, grafik, dan identifikasi kualitas data akan dilanjutkan pada tahap berikutnya.
