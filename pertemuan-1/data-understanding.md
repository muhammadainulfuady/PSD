---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Data Understanding

Data Understanding adalah tahap untuk **mengumpulkan**, **mengeksplorasi**, dan **menilai kualitas** data yang akan digunakan dalam analisis kualitas udara di Kabupaten Gresik.

---

## Library Python yang Diperlukan

Berikut adalah library Python beserta kegunaannya untuk mengerjakan proses data understanding ini:

| Library      | Kegunaan                                                                             |
| ------------ | ------------------------------------------------------------------------------------ |
| `openeo`     | Menghubungkan dan memproses data satelit dari server openEO (Copernicus Data Space). |
| `netCDF4`    | Membaca file hasil batch job openEO berformat netCDF (`.nc`).                        |
| `pandas`     | Membaca dan mengolah data tabular (CSV), serta manipulasi deret waktu.               |
| `numpy`      | Komputasi numerik, misalnya untuk perhitungan rata-rata dan statistik.               |
| `folium`     | Membuat visualisasi peta interaktif lokasi pengamatan.                               |

Instalasi dapat dilakukan secara bersamaan:

```python
pip install openeo netCDF4 pandas numpy matplotlib folium
```

---

## 1. Collecting (Mengumpulkan Data)

### 1.1 Sumber Data

Data kualitas udara dikumpulkan dari **Copernicus Data Space** menggunakan layanan **openEO**. Berikut ringkasannya:

| Item                 | Keterangan                        |
| -------------------- | --------------------------------- |
| **Sumber**           | Copernicus Data Space             |
| **Layanan**          | openEO                            |
| **Server**           | `openeo.dataspace.copernicus.eu`  |
| **Produk / Koleksi** | Sentinel-5P L2                    |
| **Polutan**          | NO2, CO, SO2, CH4                 |
| **Perioda**          | 24 Agustus 2025 – 24 Agustus 2026 |
| **Lokasi**           | Kabupaten Gresik, Jawa Timur      |

### 1.2 Koneksi dan Otentikasi

Langkah pertama adalah menghubungkan ke server openEO dan melakukan otentikasi menggunakan akun **Copernicus Data Space**.

```python
import openeo

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
```

Saat dijalankan, akan muncul tautan untuk login (menggunakan _device code flow_). Setelah berhasil melakukan login, akan muncul konfirmasi seperti di bawah ini:

```
Visit (link authentikasi) 📋 to authenticate.
✅ Authorized successfully
Authenticated using device code flow.
```

> **Catatan:** Proses **login/autentikasi sudah berhasil** (✅ Authorized successfully). Ini menandakan koneksi ke server Copernicus Data Space berjalan benar.

Proses crawling data (`code-NO2.ipynb`, `code-CO.ipynb`, `code-SO2.ipynb`, `code-CH4.ipynb`). Notebook tersebut menjalankan **batch job** di server openEO, dan hasilnya dapat **dipantau (monitoring) melalui openEO Web Editor**.

```{figure} ../assets/editor/openeo.png
:width: 100%
:align: center

Pantauan batch job NO2 Gresik yang telah selesai (status *finished*) pada openEO Web Editor. Bisa dilihat langsung di https://editor.openeo.org/.
```

### 1.3 Penentuan Area of Interest (AOI)

**Lokasi pengamatan** dibatasi pada area di **Kabupaten Gresik**. Area ini digambar sebagai **polygon** di atas peta menggunakan alat seperti [geojson.io](https://geojson.io). **Data peta (koordinat lokasi) dapat dilihat langsung dari file geojson** yang dihasilkan.

```{figure} ../assets/data/geojson.png
:width: 100%
:align: center

Polygon Area of Interest (AOI) di Kabupaten Gresik yang digambar pada peta geojson.
```

**Penjelasan koordinat:**

Setiap titik pada polygon memiliki format `[longitude (bujur), latitude (lintang)]`. Dari polygon tersebut, kita memperoleh:

| Atribut | Nilai       | Keterangan                       |
| ------- | ----------- | -------------------------------- |
| `west`  | 112.6193968 | Longitude terkecil (batas kiri)  |
| `east`  | 112.6600158 | Longitude terbesar (batas kanan) |
| `south` | -7.1927923  | Latitude terkecil (batas bawah)  |
| `north` | -7.1514786  | Latitude terbesar (batas atas)   |

> **Catatan:** Untuk data Sentinel-5P, `spatial_extent` pada `load_collection` menggunakan _bounding box_ (kotak batas) yang dibentuk oleh `west`, `south`, `east`, `north`. Sedangkan `aoi` (polygon) digunakan pada tahap `aggregate_spatial` untuk menghitung rata-rata di dalam area tersebut.

### 1.4 Memuat Data (Load Collection)

Data dimuat **per polutan**, karena server Sentinel-5P di openEO hanya mendukung **satu band per proses**. Oleh karena itu dibuat **4 notebook terpisah** (`code-NO2.ipynb`, `code-CO.ipynb`, `code-SO2.ipynb`, `code-CH4.ipynb`), masing-masing untuk satu polutan.

#### a. Memuat Data NO2

```python
s5 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={
        "west": 112.6193968,
        "south": -7.1927923,
        "east": 112.6600158,
        "north": -7.1514786,
    },
    bands=["NO2"],
)
```

#### b. Memuat Data CO

```python
s5 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={
        "west": 112.6193968,
        "south": -7.1927923,
        "east": 112.6600158,
        "north": -7.1514786,
    },
    bands=["CO"],
)
```

#### c. Memuat Data SO2

```python
s5 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={
        "west": 112.6193968,
        "south": -7.1927923,
        "east": 112.6600158,
        "north": -7.1514786,
    },
    bands=["SO2"],
)
```

#### d. Memuat Data CH4

```python
s5 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={
        "west": 112.6193968,
        "south": -7.1927923,
        "east": 112.6600158,
        "north": -7.1514786,
    },
    bands=["CH4"],
)
```

**Perhatikan:** Kode untuk CO, SO2, dan CH4 **identik** dengan NO2 — hanya berbeda pada parameter `bands`.

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
                    [112.6193968, -7.1514786],   # kiri-atas (NW)
                    [112.6600158, -7.1514786],   # kanan-atas (NE)
                    [112.6600158, -7.1927923],   # kanan-bawah (SE)
                    [112.619805,  -7.1927923],   # kiri-bawah (SW)
                    [112.6193968, -7.1514786]    # kembali ke titik awal
                ]]
            }
        }
    ]
}
```

### 1.6 Agregasi Data

Setiap datacube diagregasi menjadi **rata-rata harian**, lalu dihitung **rata-rata spasial** di dalam polygon AOI untuk menghasilkan deret waktu (time series).

```python
# Rata-rata harian, lalu rata-rata di dalam polygon AOI
s5 = s5.aggregate_temporal_period(reducer="mean", period="day")
s5 = s5.aggregate_spatial(reducer="mean", geometries=aoi)
```

### 1.7 Menjalankan Batch Job

Proses dijalankan sebagai **batch job** di server openEO. Hasilnya diunduh sebagai file netCDF (`.nc`).

```python
job = s5.execute_batch(title="NO2 Gresik", outputfile="../data/nc/polutan_NO2_gresik.nc")
```

Setiap batch job membutuhkan beberapa menit (antre di server). Hasilnya disimpan pada folder **`data/nc/`**:

| Polutan | File netCDF                     |
| ------- | ------------------------------- |
| NO2     | `data/nc/polutan_NO2_gresik.nc` |
| CO      | `data/nc/polutan_CO_gresik.nc`  |
| SO2     | `data/nc/polutan_SO2_gresik.nc` |
| CH4     | `data/nc/polutan_CH4_gresik.nc` |

---

## 2. Visualisasi Peta (folium)

Lokasi pengamatan di Kabupaten Gresik divisualisasikan pada **peta interaktif** menggunakan library **`folium`**. Area polygon AOI ditandai pada peta.

```{code-cell}
:tags: [hide-input]
import folium

# Pusat peta di tengah area AOI
lat_c = (-7.1927923 + -7.1514786) / 2
lon_c = (112.6193968 + 112.6600158) / 2

m = folium.Map(location=[lat_c, lon_c], zoom_start=12)

# Tandai area polygon AOI Kabupaten Gresik
folium.Rectangle(
    bounds=[[-7.1927923, 112.6193968], [-7.1514786, 112.6600158]],
    color="red",
    fill=True,
    fill_opacity=0.2,
    tooltip="Area Kabupaten Gresik",
).add_to(m)

m
```

---

## 3. Menampilkan Hasil Data CSV

Setelah data dikonversi menjadi CSV, kita dapat menampilkan isi data menggunakan **pandas** `pd.read_csv` diikuti `.head()` untuk melihat **5 baris paling atas**.

### 3.1 CH4

```{code-cell}
:tags: [hide-input]
import pandas as pd

# Menampilkan 5 data teratas CSV CH4
df_ch4 = pd.read_csv("./../data/csv/CH4_gresik_timeseries.csv")
df_ch4.head()
```

### 3.2 CO

```{code-cell}
:tags: [hide-input]
import pandas as pd

# Menampilkan 5 data teratas CSV CO
df_co = pd.read_csv("./../data/csv/CO_gresik_timeseries.csv")
df_co.head()
```

### 3.3 NO2

```{code-cell}
:tags: [hide-input]
import pandas as pd

# Menampilkan 5 data teratas CSV NO2
df_no2 = pd.read_csv("./../data/csv/NO2_gresik_timeseries.csv")
df_no2.head()
```

### 3.4 SO2

```{code-cell}
:tags: [hide-input]
import pandas as pd

# Menampilkan 5 data teratas CSV SO2
df_so2 = pd.read_csv("./../data/csv/SO2_gresik_timeseries.csv")
df_so2.head()
```

### 3.5 Kenapa Ada Nilai NaN?

Perhatikan pada hasil data di atas, ada beberapa baris yang menunjukkan nilai **NaN** (Not a Number). Artinya, pada tanggal tersebut **tidak ada data pengamatan** yang tercatat.

Penyebab munculnya NaN pada data satelit Sentinel-5P antara lain:

1. **Tidak ada lintasan satelit** — Sentinel-5P tidak melewati lokasi Kabupaten Gresik setiap hari. Satelit memiliki _revisit time_ (jadwal orbit) tertentu, sehingga ada hari-hari tertentu yang tidak terlewati.

2. **Tutupan awan (cloud cover)** — Sentinel-5P menggunakan sensor yang hasilnya dipengaruhi oleh kondisi atmosfer. Jika area tertutup awan tebal, data tidak valid sehingga dianggap kosong.

3. **Validasi kualitas (quality flag)** — data yang kualitasnya buruk (misal nilai ekstrem karena noise instrumen) dibuang oleh proses validasi data, sehingga menghasilkan celah (gap) pada deret waktu.

Karena itu, dari total **365 hari** dalam setahun, tidak semua tanggal memiliki nilai polutan. Hari-hari yang kosong inilah yang tampil sebagai **NaN** — dan ini akan dibahas lebih lanjut pada tahap **identifikasi _missing values_** di bagian selanjutnya.

---

## 4. Identifikasi Kualitas Data

Pada tahap ini dilakukan **identifikasi** (mencatat) masalah-masalah pada data, yaitu **missing values**, **outliers**, dan **noises**. Sesuai prinsip CRISP-DM, tahap Data Understanding hanya **menemukan dan mencatat** masalah tersebut — penanganan (imputasi, menghapus, dsb.) dilakukan pada tahap berikutnya (Data Preparation).

### 4.1 Missing Values

**Missing values** adalah tanggal yang tidak memiliki nilai polutan (NaN). Berikut identifikasinya:

1. ch4

```{code-cell}
:tags: [hide-input]
import pandas as pd
df = pd.read_csv("../data/csv/CH4_gresik_timeseries.csv")
ch4 = df["CH4"]
missingValueCH4 = ch4.isna().sum()
validValueCH4 = ch4.notna().sum()
print(f"Jumlah missing value pada data ch4 : {missingValueCH4}")
print(f"Jumlah data terisi (valid) pada data ch4 : {validValueCH4}")
```

```{figure} ../assets/editor/orange/ch4-msv-orange.png
:width: 100%
:align: center
```

2. co

```{code-cell}
:tags: [hide-input]
df = pd.read_csv("../data/csv/CO_gresik_timeseries.csv")
co = df["CO"]
missingValueCO = co.isna().sum()
validValueCO = co.notna().sum()
print(f"Jumlah missing value pada data co : {missingValueCO}")
print(f"Jumlah data terisi (valid) pada data co : {validValueCO}")
```

```{figure} ../assets/editor/orange/co-msv-orange.png
:width: 100%
:align: center
```

3. no2

```{code-cell}
:tags: [hide-input]
df = pd.read_csv("../data/csv/NO2_gresik_timeseries.csv")
no2 = df["NO2"]
missingValueNO2 = no2.isna().sum()
validValueNO2 = no2.notna().sum()
print(f"Jumlah missing value pada data no2 : {missingValueNO2}")
print(f"Jumlah data terisi (valid) pada data no2 : {validValueNO2}")
```

```{figure} ../assets/editor/orange/no2-msv-orange.png
:width: 100%
:align: center
```

4. so2

```{code-cell}
:tags: [hide-input]
df = pd.read_csv("../data/csv/SO2_gresik_timeseries.csv")
so2 = df["SO2"]
missingValueSO2 = so2.isna().sum()
validValueSO2 = so2.notna().sum()
print(f"Jumlah missing value pada data so2 : {missingValueSO2}")
print(f"Jumlah data terisi (valid) pada data so2 : {validValueSO2}")
```

```{figure} ../assets/editor/orange/so2-msv-orange.png
:width: 100%
:align: center
```

### 4.2 Outliers

**Outliers** (pencilan) adalah nilai pengamatan yang menyimpang secara signifikan dari mayoritas data dalam suatu variabel. Pada dataset ini, deteksi outlier dilakukan menggunakan algoritma **Isolation Forest** dengan tingkat kontaminasi (`contamination`) sebesar **0.05** (5%).

Sebelum deteksi outlier dilakukan, data bernilai kosong (*missing values* / `NaN`) terlebih dahulu dibuang (menggunakan `dropna()` pada Python atau widget `Impute` $\rightarrow$ *Remove instances with unknown values* pada Orange Data Mining) agar populasi perhitungan pencilan selaras.

Berikut adalah hasil identifikasi outlier untuk masing-masing polutan:

1. ch4

```{code-cell}
:tags: [hide-input]
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/CH4_gresik_timeseries.csv")
df_clean = df.dropna(subset=['CH4']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['CH4']])

normal_ch4 = df_clean[df_clean['outlier'] == 1]
outliers_ch4 = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data ch4 : {len(outliers_ch4)}")
print(f"Jumlah tidak outlier (normal) pada data ch4 : {len(normal_ch4)}")

# Visualisasi Grafik Outlier CH4
df_clean['date'] = pd.to_datetime(df_clean['date'])

plt.figure(figsize=(10, 4))
plt.scatter(normal_ch4['date'], normal_ch4['CH4'], color='blue', label='Normal', s=30)
plt.scatter(outliers_ch4['date'], outliers_ch4['CH4'], color='red', label='Outlier', s=50)
plt.title('Deteksi Outlier CH4 (Merah = Outlier, Biru = Normal)')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi CH4')
plt.legend()
plt.grid(True)
plt.show()
```

```{figure} ../assets/editor/orange/ch4-otlr-orange.png
:width: 100%
:align: center
```

2. co

```{code-cell}
:tags: [hide-input]
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/CO_gresik_timeseries.csv")
df_clean = df.dropna(subset=['CO']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['CO']])

normal_co = df_clean[df_clean['outlier'] == 1]
outliers_co = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data co : {len(outliers_co)}")
print(f"Jumlah tidak outlier (normal) pada data co : {len(normal_co)}")

# Visualisasi Grafik Outlier CO
df_clean['date'] = pd.to_datetime(df_clean['date'])

plt.figure(figsize=(10, 4))
plt.scatter(normal_co['date'], normal_co['CO'], color='blue', label='Normal', s=30)
plt.scatter(outliers_co['date'], outliers_co['CO'], color='red', label='Outlier', s=50)
plt.title('Deteksi Outlier CO (Merah = Outlier, Biru = Normal)')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi CO')
plt.legend()
plt.grid(True)
plt.show()
```

```{figure} ../assets/editor/orange/co-otlr-orange.png
:width: 100%
:align: center
```

3. no2

```{code-cell}
:tags: [hide-input]
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/NO2_gresik_timeseries.csv")
df_clean = df.dropna(subset=['NO2']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['NO2']])

normal_no2 = df_clean[df_clean['outlier'] == 1]
outliers_no2 = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data no2 : {len(outliers_no2)}")
print(f"Jumlah tidak outlier (normal) pada data no2 : {len(normal_no2)}")

# Visualisasi Grafik Outlier NO2
df_clean['date'] = pd.to_datetime(df_clean['date'])

plt.figure(figsize=(10, 4))
plt.scatter(normal_no2['date'], normal_no2['NO2'], color='blue', label='Normal', s=30)
plt.scatter(outliers_no2['date'], outliers_no2['NO2'], color='red', label='Outlier', s=50)
plt.title('Deteksi Outlier NO2 (Merah = Outlier, Biru = Normal)')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi NO2')
plt.legend()
plt.grid(True)
plt.show()
```

```{figure} ../assets/editor/orange/no2-otlr-orange.png
:width: 100%
:align: center
```

4. so2

```{code-cell}
:tags: [hide-input]
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/SO2_gresik_timeseries.csv")
df_clean = df.dropna(subset=['SO2']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['SO2']])

normal_so2 = df_clean[df_clean['outlier'] == 1]
outliers_so2 = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data so2 : {len(outliers_so2)}")
print(f"Jumlah tidak outlier (normal) pada data so2 : {len(normal_so2)}")

# Visualisasi Grafik Outlier SO2
df_clean['date'] = pd.to_datetime(df_clean['date'])

plt.figure(figsize=(10, 4))
plt.scatter(normal_so2['date'], normal_so2['SO2'], color='blue', label='Normal', s=30)
plt.scatter(outliers_so2['date'], outliers_so2['SO2'], color='red', label='Outlier', s=50)
plt.title('Deteksi Outlier SO2 (Merah = Outlier, Biru = Normal)')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi SO2')
plt.legend()
plt.grid(True)
plt.show()
```

```{figure} ../assets/editor/orange/so2-otlr-orange.png
:width: 100%
:align: center
```

### 4.3 Noise

BESOK SAJA

