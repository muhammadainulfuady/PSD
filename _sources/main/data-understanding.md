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

Data Understanding adalah tahap untuk **mengumpulkan**, **mengeksplorasi**, dan **menilai kualitas** data yang akan digunakan dalam analisis kualitas udara di Kabupaten Gresik (Area Observasi Baru).

---

## Library Python yang Diperlukan

Berikut adalah library Python beserta kegunaannya untuk mengerjakan proses data understanding ini:

| Library        | Kegunaan                                                                                                                |
| :------------- | :---------------------------------------------------------------------------------------------------------------------- |
| `openeo`       | Menghubungkan dan memproses data satelit dari server openEO (Copernicus Data Space).                                    |
| `netCDF4`      | Membaca file hasil batch job openEO berformat netCDF (`.nc`).                                                           |
| `pandas`       | Membaca dan mengolah data tabular (CSV), serta manipulasi deret waktu (_time-series_).                                  |
| `numpy`        | Komputasi numerik, misalnya perhitungan rata-rata, standar deviasi, dan statistik.                                      |
| `matplotlib`   | Visualisasi grafik data komparatif dual X-axis 4 polutan.                                                               |
| `scikit-learn` | Deteksi pencilan (_outlier detection_) menggunakan algoritma _Isolation Forest_.                                        |
| `folium`       | Membuat visualisasi peta interaktif lokasi pengamatan (AOI Kabupaten Gresik).                                           |

Instalasi dapat dilakukan secara bersamaan:

```bash
pip install openeo netCDF4 pandas numpy matplotlib scikit-learn folium
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

### 1.3 Penentuan Area of Interest (AOI)

**Lokasi pengamatan** dibatasi pada area di **Kabupaten Gresik**. Area ini digambar sebagai **polygon** di atas peta menggunakan alat seperti [geojson.io](https://geojson.io). **Data peta (koordinat lokasi) dapat dilihat langsung dari file geojson** yang dihasilkan.

```{figure} ../assets/data/geojson.png
:width: 100%
:align: center

Polygon Area of Interest (AOI) di Kabupaten Gresik yang digambar pada peta geojson.
```

**Penjelasan koordinat (BBOX Terbaru):**

Setiap titik pada polygon memiliki format `[longitude (bujur), latitude (lintang)]`. Dari polygon tersebut, kita memperoleh *Bounding Box*:

| Atribut | Nilai       | Keterangan                       |
| ------- | ----------- | -------------------------------- |
| `west`  | 112.5058378 | Longitude terkecil (batas kiri)  |
| `east`  | 112.5963518 | Longitude terbesar (batas kanan) |
| `south` | -7.0683846  | Latitude terkecil (batas bawah)  |
| `north` | -7.0230496  | Latitude terbesar (batas atas)   |

> **Catatan:** Untuk data Sentinel-5P, `spatial_extent` pada `load_collection` menggunakan _bounding box_ (kotak batas) yang dibentuk oleh `west`, `south`, `east`, `north`. Sedangkan `aoi` (polygon) digunakan pada tahap `aggregate_spatial` untuk menghitung rata-rata di dalam area tersebut.

### 1.4 Memuat Data (Load Collection)

Data dimuat untuk masing-masing polutan menggunakan rentang koordinat *Bounding Box* terbaru. Proses pemrosesan ini digabungkan secara praktis di dalam notebook `crawling-polutan.ipynb`.

#### a. Memuat Data NO2

```python
s5 = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2025-08-24", "2026-08-24"],
    spatial_extent={
        "west": 112.5058378,
        "south": -7.0683846,
        "east": 112.5963518,
        "north": -7.0230496,
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
        "west": 112.5058378,
        "south": -7.0683846,
        "east": 112.5963518,
        "north": -7.0230496,
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
        "west": 112.5058378,
        "south": -7.0683846,
        "east": 112.5963518,
        "north": -7.0230496,
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
        "west": 112.5058378,
        "south": -7.0683846,
        "east": 112.5963518,
        "north": -7.0230496,
    },
    bands=["CH4"],
)
```

### 1.5 Definisi AOI (Polygon Terbaru)

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
                    [112.5058378, -7.0230496],
                    [112.5963518, -7.0230496],
                    [112.5963518, -7.0683846],
                    [112.5067768, -7.0683846],
                    [112.5058378, -7.0230496]
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

Proses crawling data. Notebook tersebut menjalankan **batch job** di server openEO, dan hasilnya dapat **dipantau (monitoring) melalui openEO Web Editor**.

```{figure} ../assets/editor/openeo.png
:width: 100%
:align: center

Pantauan batch job NO2 Gresik yang telah selesai (status *finished*) pada openEO Web Editor. Bisa dilihat langsung di https://editor.openeo.org/.
```

---

## 2. Visualisasi Peta (folium)

Lokasi pengamatan di Kabupaten Gresik divisualisasikan pada **peta interaktif** menggunakan library **`folium`**. Area polygon AOI ditandai pada peta.

```{code-cell}
:tags: [hide-input]
import folium

# Pusat peta di tengah area AOI
lat_c = (-7.0683846 + -7.0230496) / 2
lon_c = (112.5058378 + 112.5963518) / 2

m = folium.Map(location=[lat_c, lon_c], zoom_start=12)

# Tandai area polygon AOI Kabupaten Gresik
folium.Rectangle(
    bounds=[[-7.0683846, 112.5058378], [-7.0230496, 112.5963518]],
    color="red",
    fill=True,
    fill_opacity=0.2,
    tooltip="Area Pengamatan Kabupaten Gresik (AOI Baru)",
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

### 3.5 polutan_gresik.csv (Gabungan 4 Polutan)

```{code-cell}
:tags: [hide-input]
import pandas as pd

# Menampilkan 5 data teratas CSV gabungan 4 polutan
df_polutan = pd.read_csv("./../data/csv/polutan_gresik.csv")
df_polutan.head()
```

### 3.6 Kenapa Ada Nilai NaN?

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

**Missing values** adalah tanggal yang tidak memiliki nilai polutan (NaN). Berikut identifikasinya melalui kode Python:

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

---

### 4.2 Outliers

**Outliers** (pencilan) adalah nilai pengamatan yang menyimpang secara signifikan dari mayoritas data dalam suatu variabel. Pada dataset ini, deteksi outlier dilakukan menggunakan algoritma **Isolation Forest** dengan tingkat kontaminasi (`contamination`) sebesar **0.05** (5%).

Sebelum deteksi outlier dilakukan, data bernilai kosong (_missing values_ / `NaN`) terlebih dahulu dibuang (menggunakan `dropna()` pada Python) agar populasi perhitungan pencilan selaras.

Berikut adalah hasil identifikasi jumlah outlier untuk masing-masing polutan:

1. ch4

```{code-cell}
:tags: [hide-input]
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/CH4_gresik_timeseries.csv")
df_clean = df.dropna(subset=['CH4']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['CH4']])

normal_ch4 = df_clean[df_clean['outlier'] == 1]
outliers_ch4 = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data ch4 : {len(outliers_ch4)}")
print(f"Jumlah tidak outlier (normal) pada data ch4 : {len(normal_ch4)}")
```

2. co

```{code-cell}
:tags: [hide-input]
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/CO_gresik_timeseries.csv")
df_clean = df.dropna(subset=['CO']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['CO']])

normal_co = df_clean[df_clean['outlier'] == 1]
outliers_co = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data co : {len(outliers_co)}")
print(f"Jumlah tidak outlier (normal) pada data co : {len(normal_co)}")
```

3. no2

```{code-cell}
:tags: [hide-input]
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/NO2_gresik_timeseries.csv")
df_clean = df.dropna(subset=['NO2']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['NO2']])

normal_no2 = df_clean[df_clean['outlier'] == 1]
outliers_no2 = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data no2 : {len(outliers_no2)}")
print(f"Jumlah tidak outlier (normal) pada data no2 : {len(normal_no2)}")
```

4. so2

```{code-cell}
:tags: [hide-input]
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/csv/SO2_gresik_timeseries.csv")
df_clean = df.dropna(subset=['SO2']).copy()

model = IsolationForest(contamination=0.05, random_state=42)
df_clean['outlier'] = model.fit_predict(df_clean[['SO2']])

normal_so2 = df_clean[df_clean['outlier'] == 1]
outliers_so2 = df_clean[df_clean['outlier'] == -1]

print(f"Jumlah outlier pada data so2 : {len(outliers_so2)}")
print(f"Jumlah tidak outlier (normal) pada data so2 : {len(normal_so2)}")
```

---

### 4.3 Noise

**Noise** (derau) adalah fluktuasi acak frekuensi tinggi (_random noise_) pada data pengamatan yang disebabkan oleh kondisi dinamika atmosfer mikro, keterbatasan presisi instrumen satelit, atau interferensi cuaca lokal. Berbeda dari _outlier_ yang berupa pencilan ekstrem tunggal, _noise_ diukur berdasarkan tingkat fluktuasi atau variabilitas relatif data.

Untuk mengukur dan mengidentifikasi _noise_ pada data polutan, digunakan 3 indikator statistik utama:

1. **Rata-rata ($\mu$)**: Nilai rata-rata konsentrasi polutan.
2. **Standar Deviasi ($\sigma$)**: Ukuran sebaran atau besar fluktuasi data dari rata-rata.
3. **Koefisien Variasi ($CV$)**: Rasio fluktuasi relatif terhadap rata-rata ($CV = \frac{\sigma}{\mu} \times 100\%$). Semakin tinggi nilai $CV$, semakin tinggi tingkat derau (_noise_) atau variabilitas relatifnya.

Berikut adalah hasil identifikasi indikator statistik _noise_ untuk masing-masing polutan:

1. ch4

```{code-cell}
:tags: [hide-input]
import pandas as pd

df = pd.read_csv("../data/csv/CH4_gresik_timeseries.csv")
df_clean = df.dropna(subset=['CH4']).copy()

mean_val = df_clean['CH4'].mean()
std_val = df_clean['CH4'].std()
cv_val = (std_val / mean_val) * 100

print(f"Rata-rata CH4 : {mean_val:.4f}")
print(f"Standar Deviasi CH4 : {std_val:.4f}")
print(f"Koefisien Variasi (CV) CH4 : {cv_val:.2f}%")
```

2. co

```{code-cell}
:tags: [hide-input]
import pandas as pd

df = pd.read_csv("../data/csv/CO_gresik_timeseries.csv")
df_clean = df.dropna(subset=['CO']).copy()

mean_val = df_clean['CO'].mean()
std_val = df_clean['CO'].std()
cv_val = (std_val / mean_val) * 100

print(f"Rata-rata CO : {mean_val:.6f}")
print(f"Standar Deviasi CO : {std_val:.6f}")
print(f"Koefisien Variasi (CV) CO : {cv_val:.2f}%")
```

3. no2

```{code-cell}
:tags: [hide-input]
import pandas as pd

df = pd.read_csv("../data/csv/NO2_gresik_timeseries.csv")
df_clean = df.dropna(subset=['NO2']).copy()

mean_val = df_clean['NO2'].mean()
std_val = df_clean['NO2'].std()
cv_val = (std_val / mean_val) * 100

print(f"Rata-rata NO2 : {mean_val:.6f}")
print(f"Standar Deviasi NO2 : {std_val:.6f}")
print(f"Koefisien Variasi (CV) NO2 : {cv_val:.2f}%")
```

4. so2

```{code-cell}
:tags: [hide-input]
import pandas as pd

df = pd.read_csv("../data/csv/SO2_gresik_timeseries.csv")
df_clean = df.dropna(subset=['SO2']).copy()

mean_val = df_clean['SO2'].mean()
std_val = df_clean['SO2'].std()
cv_val = (std_val / mean_val) * 100

print(f"Rata-rata SO2 : {mean_val:.6f}")
print(f"Standar Deviasi SO2 : {std_val:.6f}")
print(f"Koefisien Variasi (CV) SO2 : {cv_val:.2f}%")
```

---

### 4.4 Visualisasi Komparatif 4 Polutan (Style Copernicus)

Untuk membandingkan tren perubahan konsentrasi ke-4 polutan (CH4, CO, NO2, SO2) secara bersamaan sepanjang periode pengamatan di Kabupaten Gresik, dibuat visualisasi **Dual X-Axis Line Plot** dengan gaya visualisasi Copernicus Sentinel-5P.

```{figure} ../assets/editor/copernicus_4polutan_dual_axis.png
:width: 100%
:align: center
```

```{code-cell}
:tags: [hide-input]
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data 4 Polutan
df_no2 = pd.read_csv("../data/csv/NO2_gresik_timeseries.csv").dropna(subset=['NO2'])
df_co  = pd.read_csv("../data/csv/CO_gresik_timeseries.csv").dropna(subset=['CO'])
df_ch4 = pd.read_csv("../data/csv/CH4_gresik_timeseries.csv").dropna(subset=['CH4'])
df_so2 = pd.read_csv("../data/csv/SO2_gresik_timeseries.csv").dropna(subset=['SO2'])

df_no2['date'] = pd.to_datetime(df_no2['date'])
df_co['date']  = pd.to_datetime(df_co['date'])
df_ch4['date'] = pd.to_datetime(df_ch4['date'])
df_so2['date'] = pd.to_datetime(df_so2['date'])

# 2. Smooth data dengan 30-day moving average (seperti Copernicus)
df_no2['NO2_smooth'] = df_no2['NO2'].rolling(window=30, min_periods=1).mean()
df_co['CO_smooth']   = df_co['CO'].rolling(window=30, min_periods=1).mean()
df_ch4['CH4_smooth'] = df_ch4['CH4'].rolling(window=30, min_periods=1).mean()
df_so2['SO2_smooth'] = df_so2['SO2'].rolling(window=30, min_periods=1).mean()

# 3. Normalisasi Min-Max (0 - 1)
for df, col in [(df_no2, 'NO2_smooth'), (df_co, 'CO_smooth'), (df_ch4, 'CH4_smooth'), (df_so2, 'SO2_smooth')]:
    df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# 4. Figure Dual X-Axis (Gaya Copernicus twiny())
fig, ax1 = plt.subplots(figsize=(11, 5), dpi=100)

# Sumbu Bawah: NO2 (Merah) & CO (Hijau)
(line1,) = ax1.plot(df_no2['date'], df_no2['NO2_smooth_norm'], color="r", label="NO2 (Nitrogen Dioksida)", linewidth=1.8)
(line2,) = ax1.plot(df_co['date'],  df_co['CO_smooth_norm'],   color="g", label="CO (Karbon Monoksida)", linewidth=1.8)

ax1.set_xlabel("Periode NO2 & CO (Sumbu Bawah)", color='r', fontsize=10)
ax1.set_ylabel("Skala Ternormalisasi (0 - 1)")
ax1.xaxis.label.set_color("r")
ax1.tick_params(axis="x", colors="r")
ax1.grid(True, linestyle='--', alpha=0.5)

# Sumbu Atas: CH4 (Biru) & SO2 (Oranye)
ax2 = ax1.twiny()
(line3,) = ax2.plot(df_ch4['date'], df_ch4['CH4_smooth_norm'], color="b", label="CH4 (Metana)", linewidth=1.8)
(line4,) = ax2.plot(df_so2['date'], df_so2['SO2_smooth_norm'], color="orange", label="SO2 (Sulfur Dioksida)", linewidth=1.8)

ax2.set_xlabel("Periode CH4 & SO2 (Sumbu Atas)", color='b', fontsize=10)
ax2.xaxis.label.set_color("b")
ax2.tick_params(axis="x", colors="b")

# Combine Legend
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left")

plt.title("Grafik Komparatif Tren 4 Polutan di Kabupaten Gresik", pad=20)
plt.tight_layout()
plt.show()
```

#### Penjelasan Rinci Komponen Grafik:

1. **Struktur Dual X-Axis (Sumbu X Ganda)**:
   - **Sumbu X Bawah (Merah)**: Digunakan sebagai penanda garis waktu tanggal untuk polutan NO2 dan CO.
   - **Sumbu X Atas (Biru)**: Digunakan sebagai penanda garis waktu tanggal untuk polutan CH4 dan SO2.
   - **Fungsi `twiny()`**: Memungkinkan 2 pasangan polutan menggunakan sumbu Y bersama di sebelah kiri, namun memiliki skala waktu horizontal di atas dan bawah untuk menjaga keterbacaan grafik.

2. **Keterangan 4 Warna Polutan**:
   - 🔴 **Garis Merah (NO2)**: Menunjukkan tren emisi gas Nitrogen Dioksida (berasal dari transportasi kendaraan dan pembakaran industri).
   - 🟢 **Garis Hijau (CO)**: Menunjukkan tren emisi gas Karbon Monoksida (berasal dari pembuangan asap pembakaran).
   - 🔵 **Garis Biru (CH4)**: Menunjukkan tren konsentrasi gas Metana (gas rumah kaca dari zona industri/limbah/tambang).
   - 🟠 **Garis Oranye (SO2)**: Menunjukkan tren emisi gas Sulfur Dioksida (berasal dari pembakaran batu bara & pemrosesan industri).

3. **Mengapa Menggunakan Skala Ternormalisasi (0 - 1) di Sumbu Y?**:
   - Nilai asli dari masing-masing polutan memiliki rentang angka yang jauh berbeda (misalnya: CH4 bernilai tinggi sekitar ~1890, sedangkan NO2 bernilai kecil sekitar ~0.00007).
   - Jika diplot tanpa skala bersama, garis NO2, CO, dan SO2 akan terlihat "gepeng/datar" di angka 0.
   - Dengan **Normalisasi Min-Max ($0 - 1$)**, semua garis polutan dibawa ke rentang proporsional yang sama ($0$ = konsentrasi terendah, $1$ = konsentrasi tertinggi), sehingga naik-turunnya pola tren ke-4 polutan dapat dibandingkan secara langsung.

4. **Penghalusan Grafik (_30-Day Moving Average_)**:
   - Garis grafik dihaluskan menggunakan teknik _rolling mean_ 30 hari (`.rolling(window=30).mean()`) persis seperti pada kode acuan Copernicus.
   - Hal ini berfungsi untuk meredam derau (_noise_) harian, sehingga garis grafik terlihat mulus dan pembaca dapat melihat tren kenaikan/penurunan jangka panjang di Kabupaten Gresik secara jernih.

5. **Cara Membaca Arah dan Gerakan Garis Grafik**:
   - 📈 **Garis Naik ke Atas**: Menandakan konsentrasi polutan di Kabupaten Gresik sedang **meningkat / tinggi** (akibat lonjakan emisi industri, volume kendaraan padat, atau musim kemarau di mana emisi terperangkap di atmosfer).
   - 📉 **Garis Turun ke Bawah**: Menandakan kualitas udara sedang **lebih bersih / polusi rendah** (akibat pencucian polutan oleh air hujan / _rain washout_, berkurangnya aktivitas saat hari libur, atau tiupan angin kencang).
   - ➡️ **Garis di Tengah / Datar**: Menandakan konsentrasi polutan berada pada **kondisi rata-rata latar belakang yang stabil**.
   - 〰️ **Garis Fluktuasi (Bergerigi / "Gleot-gleot")**: Menandakan adanya variasi perubahan cuaca harian yang cepat serta derau (_noise_) instrumen pengamatan satelit Sentinel-5P.

---

## 5. Integrasi Cloud Database & Analytics Workflow (Aiven, DBeaver, & KNIME)

Selain pemrosesan secara lokal dengan Python, alur pemahaman data (_Data Understanding_) ini juga diimplementasikan menggunakan infrastruktur _cloud database_ dan perangkat analitik visual: **Aiven PostgreSQL**, **DBeaver**, dan **KNIME Analytics Platform**.

---

### 5.1 Konfigurasi Cloud Database (Aiven PostgreSQL)

**Aiven** digunakan sebagai penyedia layanan _Cloud Database_ terkelola (_managed database_) berbasis PostgreSQL. Langkah-langkah konfigurasinya:

1. Membuat layanan (_service_) baru bertipe **PostgreSQL** pada konsol platform Aiven Cloud.
2. Mengonfigurasi parameter koneksi jaringan, _database name_, _username_, _password_, serta sertifikat SSL (`CA Certificate`).
3. Mencatat kredensial koneksi _Host_ dan _Port_ publik untuk dihubungkan dengan perangkat GUI Client dan KNIME.

```{figure} ../assets/editor/aiven/create-project.png
:width: 100%
:align: center

Konfigurasi Service Cloud Database PostgreSQL pada Console Aiven
```

---

### 5.2 Pengelolaan & Pengujian Data (DBeaver Client)

**DBeaver** digunakan sebagai _Database Management Tool (GUI Client)_ untuk mengelola struktur tabel relasional dan menguji koneksi jaringan ke cloud Aiven:

1. Membuat koneksi baru (_New Database Connection_) berjenis PostgreSQL di DBeaver menggunakan Host, Port, dan kredensial Aiven.
2. Mengimpor dataset `polutan_gresik.csv` ke dalam tabel relasional PostgreSQL bernama `polutan_gresik`.
3. Menjalankan _SQL Query_ `SELECT * FROM polutan_gresik;` untuk memastikan data terstruktur dengan benar di dalam _cloud database_.

```{figure} ../assets/editor/dbeaver/select-tabel.png
:width: 100%
:align: center

Pengelolaan Tabel dan Query Data pada DBeaver Client
```

---

### 5.3 Workflow Pipeline Data (KNIME Analytics Platform)

**KNIME Analytics Platform** digunakan untuk membangun alur kerja pemrosesan data (_Data Pipeline_) secara visual tanpa pengkodean (_low-code_). Node-node yang digunakan dalam workflow ini antara lain:

1. **`PostgreSQL Connector`**: Mengonfigurasi parameter koneksi JDBC (Host, Port, Database Name, Username, Password, dan SSL) untuk menghubungkan KNIME secara langsung ke _cloud database_ Aiven PostgreSQL.
2. **`DB Table Selector`**: Memilih tabel sasaran `polutan_gresik` dari skema database PostgreSQL di cloud.
3. **`DB Reader`**: Mengeksekusi query ekstraksi dan membaca seluruh data dari server _cloud_ Aiven ke dalam format tabel memori KNIME.
4. **`Table View`**: Menampilkan pratinjau isi tabel data observasi secara interaktif.
5. **`Statistics` & `Statistics View`**: Mengekstraksi ringkasan statistik deskriptif dan visualisasi distribusi histogram dari setiap atribut polutan.

```{figure} ../assets/editor/knime/knime.png
:width: 100%
:align: center

Alur Kerja (Workflow Pipeline) Ekstraksi Data PostgreSQL pada KNIME
```

---

### 5.4 Perhitungan Manual Statistik Deskriptif (Rumus & Langkah Kerja)

Sub-bab ini menyediakan rincian perhitungan statistik secara manual menggunakan rumus LaTeX matematis dari nilai minimum hingga konstruksi grafik histogram. **Seluruh kalkulasi dihitung eksklusif hanya pada populasi data valid ($N$) setelah membuang nilai kosong (*missing values* / `NaN`)** dari file CSV pengamatan (`polutan_gresik.csv`).

Untuk memverifikasi kebenaran formula, dilakukan pengujian komparasi antara kalkulasi matematis manual dengan hasil ekstraksi otomatis dari node **Statistics** pada **KNIME Analytics Platform**:

```{figure} ../assets/editor/knime/eda-lengkap.png
:width: 100%
:align: center

Ringkasan Statistik Deskriptif dan Visualisasi Histogram Seluruh Atribut Polutan pada KNIME
```

> [!NOTE]
> **Catatan Pembuktian & Konsistensi Perhitungan**:
> Perhitungan manual yang dilakukan pada lembar kerja (menggunakan fungsi statistik standar `MIN`, `MAX`, `AVERAGE`, `MEDIAN`, `VAR.S`, `STDEV.S`, `SKEW`, dan `KURT`) terbukti **100% konsisten dan identik secara matematis** dengan hasil ekstraksi pada perangkat lunak KNIME. 
> 
> Terdapat sedikit perbedaan jumlah angka di belakang koma (desimal) pada beberapa tampilan antarmuka. Hal ini semata-mata disebabkan oleh **pembulatan tampilan (*cosmetic display rounding*)** otomatis pada antarmuka GUI KNIME. Secara nilai komputasi dasar (*raw precision value*), kedua metode menghasilkan nilai presisi yang sepenuhnya sama.

---

#### 1. Nilai Minimum ($\text{Min}$)

- **Rumus Matematika**:

```{math}
\text{Min} = \min(X_1, X_2, \dots, X_N)
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$ ($N=20$): $\text{Min}_{\text{CH}_4} = \min(X_1, \dots, X_{20}) = 1,821.642090$
  - $\text{CO}$ ($N=192$): $\text{Min}_{\text{CO}} = \min(X_1, \dots, X_{192}) = 0.016968$
  - $\text{NO}_2$ ($N=200$): $\text{Min}_{\text{NO}_2} = \min(X_1, \dots, X_{200}) = -0.000002$
  - $\text{SO}_2$ ($N=235$): $\text{Min}_{\text{SO}_2} = \min(X_1, \dots, X_{235}) = -0.000720$

---

#### 2. Nilai Maksimum ($\text{Max}$)

- **Rumus Matematika**:

```{math}
\text{Max} = \max(X_1, X_2, \dots, X_N)
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$ ($N=20$): $\text{Max}_{\text{CH}_4} = \max(X_1, \dots, X_{20}) = 1,916.421265$
  - $\text{CO}$ ($N=192$): $\text{Max}_{\text{CO}} = \max(X_1, \dots, X_{192}) = 0.045237$
  - $\text{NO}_2$ ($N=200$): $\text{Max}_{\text{NO}_2} = \max(X_1, \dots, X_{200}) = 0.000289$
  - $\text{SO}_2$ ($N=235$): $\text{Max}_{\text{SO}_2} = \max(X_1, \dots, X_{235}) = 0.001380$

---

#### 3. Nilai Rata-Rata (_Mean_ / $\bar{X}$)

- **Rumus Matematika**:

```{math}
\bar{X} = \frac{\text{Overall sum}}{N} = \frac{\sum_{i=1}^{N} X_i}{N}
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$ ($N=20$):
    ```{math}
    \bar{X}_{\text{CH}_4} = \frac{37,745.964540}{20} = 1,887.298227
    ```
  - $\text{CO}$ ($N=192$):
    ```{math}
    \bar{X}_{\text{CO}} = \frac{5.526912}{192} = 0.028786
    ```
  - $\text{NO}_2$ ($N=200$):
    ```{math}
    \bar{X}_{\text{NO}_2} = \frac{0.009800}{200} = 0.000049
    ```
  - $\text{SO}_2$ ($N=235$):
    ```{math}
    \bar{X}_{\text{SO}_2} = \frac{0.015745}{235} = 0.000067
    ```

---

#### 4. Nilai Tengah (_Median_ / $Me$)

- **Rumus Matematika**:

```{math}
Me = \begin{cases} 
X_{\left(\frac{N+1}{2}\right)}, & \text{jika } N \text{ ganjil} \\
\frac{X_{\left(\frac{N}{2}\right)} + X_{\left(\frac{N}{2} + 1\right)}}{2}, & \text{jika } N \text{ genap}
\end{cases}
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$ ($N=20$, genap):
    ```{math}
    Me_{\text{CH}_4} = \frac{X_{(10)} + X_{(11)}}{2} = \frac{1,889.378418 + 1,892.130249}{2} = 1,890.754333
    ```
  - $\text{CO}$ ($N=192$, genap):
    ```{math}
    Me_{\text{CO}} = \frac{X_{(96)} + X_{(97)}}{2} = \frac{0.028821 + 0.028893}{2} = 0.028857
    ```
  - $\text{NO}_2$ ($N=200$, genap):
    ```{math}
    Me_{\text{NO}_2} = \frac{X_{(100)} + X_{(101)}}{2} = 0.000040
    ```
  - $\text{SO}_2$ ($N=235$, ganjil):
    ```{math}
    Me_{\text{SO}_2} = X_{\left(\frac{235+1}{2}\right)} = X_{(118)} = 0.000048
    ```

---

#### 5. Variansi Sampel ($s^2$)

- **Rumus Matematika**:

```{math}
s^2 = \frac{\sum_{i=1}^{N} (X_i - \bar{X})^2}{N - 1}
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$ ($N=20$):
    ```{math}
    s^2_{\text{CH}_4} = \frac{11,640.597950}{20 - 1} = 612.663050
    ```
  - $\text{CO}$ ($N=192$):
    ```{math}
    s^2_{\text{CO}} = \frac{0.003186}{192 - 1} = 1.6682 \times 10^{-5}
    ```
  - $\text{NO}_2$ ($N=200$):
    ```{math}
    s^2_{\text{NO}_2} = \frac{2.3474 \times 10^{-7}}{200 - 1} = 1.1796 \times 10^{-9}
    ```
  - $\text{SO}_2$ ($N=235$):
    ```{math}
    s^2_{\text{SO}_2} = \frac{1.3590 \times 10^{-5}}{235 - 1} = 5.8078 \times 10^{-8}
    ```

---

#### 6. Standar Deviasi Sampel ($s$)

- **Rumus Matematika**:

```{math}
s = \sqrt{s^2}
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$: $s_{\text{CH}_4} = \sqrt{612.663050} = 24.752031$
  - $\text{CO}$: $s_{\text{CO}} = \sqrt{1.6682 \times 10^{-5}} = 0.004084$
  - $\text{NO}_2$: $s_{\text{NO}_2} = \sqrt{1.1796 \times 10^{-9}} = 0.000034$
  - $\text{SO}_2$: $s_{\text{SO}_2} = \sqrt{5.8078 \times 10^{-8}} = 0.000241$

---

#### 7. Kemiringan (_Skewness_ / $S_k$)

- **Rumus Matematika** (Momen Ketiga Sampel):

```{math}
S_k = \frac{N}{(N-1)(N-2)} \sum_{i=1}^{N} \left( \frac{X_i - \bar{X}}{s} \right)^3
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$: $S_{k,\text{CH}_4} = -1.1954$
  - $\text{CO}$: $S_{k,\text{CO}} = +0.2253$
  - $\text{NO}_2$: $S_{k,\text{NO}_2} = +2.9133$
  - $\text{SO}_2$: $S_{k,\text{SO}_2} = +0.7711$

---

#### 8. Keruncingan (_Kurtosis_ / $K$)

- **Rumus Matematika** (Momen Keempat Sampel):

```{math}
K = \left[ \frac{N(N+1)}{(N-1)(N-2)(N-3)} \sum_{i=1}^{N} \left( \frac{X_i - \bar{X}}{s} \right)^4 \right] - \frac{3(N-1)^2}{(N-2)(N-3)}
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$: $K_{\text{CH}_4} = +1.2495$
  - $\text{CO}$: $K_{\text{CO}} = +2.0735$
  - $\text{NO}_2$: $K_{\text{NO}_2} = +13.5417$
  - $\text{SO}_2$: $K_{\text{SO}_2} = +3.6890$

---

#### 9. Aturan Sturges & Lebar Kelas Interval Histogram

- **Rumus Jumlah Kelas Sturges ($k$) & Lebar Kelas ($W$)**:

```{math}
k = 1 + 3.322 \times \log_{10}(N), \quad W = \frac{\text{Max} - \text{Min}}{k}
```

- **Perhitungan Hasil Data Valid**:
  - $\text{CH}_4$ ($N=20$):
    ```{math}
    k = 1 + 3.322 \times \log_{10}(20) = 5.32 \approx 6, \quad W = \frac{1,916.421265 - 1,821.642090}{6} = 17.808867
    ```
  - $\text{CO}$ ($N=192$):
    ```{math}
    k = 1 + 3.322 \times \log_{10}(192) = 8.59 \approx 9, \quad W = \frac{0.045237 - 0.016968}{9} = 0.003293
    ```
  - $\text{NO}_2$ ($N=200$):
    ```{math}
    k = 1 + 3.322 \times \log_{10}(200) = 8.64 \approx 9, \quad W = \frac{0.000289 - (-0.000002)}{9} = 0.000034
    ```
  - $\text{SO}_2$ ($N=235$):
    ```{math}
    k = 1 + 3.322 \times \log_{10}(235) = 8.88 \approx 9, \quad W = \frac{0.001380 - (-0.000720)}{9} = 0.000237
    ```
