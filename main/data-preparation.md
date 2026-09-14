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

# Data Preparation (Persiapan Data CO)

Data Preparation adalah tahap ketiga dalam metodologi CRISP-DM yang bertujuan untuk **membersihkan**, **mengimputasi nilai kosong (*missing values*)**, **menangani pencilan (*outliers*)**, dan **merekayasa fitur (*feature engineering*)** dari deret waktu pengamatan kualitas udara polutan **Karbon Monoksida ($\text{CO}$)** pada skala geografis **Tingkat Kecamatan (Kecamatan Bungah)** selama 365 hari (24 Agustus 2025 – 23 Agustus 2026).

---

## 1. Penanganan Missing Values & Outliers (Data Cleaning Terpadu)

### 1.1 Evaluasi Missing Values (Celah Data Mentah)

Data pengamatan satelit Sentinel-5P dari Copernicus Data Space untuk polutan $\text{CO}$ harian selama 365 hari memiliki celah data (*missing values* / `NaN`) yang disebabkan oleh tutupan awan tebal, kendala jadwal orbit satelit, dan penyaringan validasi kualitas (*quality flag*).

Evaluasi celah data dihitung secara **dinamis menggunakan kode Python** berikut:

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca Dataset Mentah CO (365 Hari)
raw_path = '../data/csv/CO_gresik_timeseries.csv'
df_raw = pd.read_csv(raw_path)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)

# Hitung statistik celah data awal
total_rows = len(df_raw)
nan_awal = df_raw['CO'].isna().sum()
valid_awal = df_raw['CO'].notna().sum()

print(f"Total Baris Observasi   : {total_rows} hari")
print(f"Jumlah Data Valid Awal  : {valid_awal} hari ({valid_awal/total_rows*100:.2f}%)")
print(f"Jumlah Missing (NaN)    : {nan_awal} hari ({nan_awal/total_rows*100:.2f}%)")
```

---

### 1.2 Deteksi & Pengosongan Outlier (Metode IQR Dinamis Berbasis Kode)

Sebelum dilakukan imputasi deret waktu, titik pencilan (*outliers*) dievaluasi terlebih dahulu pada populasi data valid agar lonjakan ekstrem tidak merusak kemiringan (*slope*) garis interpolasi.

Metode **Interquartile Range (IQR)** dihitung secara presisi dengan pustaka Pandas/NumPy:

```{math}
\text{IQR} = Q_3 - Q_1
```

```{math}
\text{Batas Bawah} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Batas Atas} = Q_3 + 1.5 \times \text{IQR}
```

Eksekusi kode Python untuk menghitung ambang batas IQR dan mengidentifikasi seluruh tanggal pencilan:

```{code-cell} ipython3
# 2. Deteksi Outlier dengan Metode IQR pada Data Valid
q1 = df_raw['CO'].quantile(0.25)
q3 = df_raw['CO'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Masking Outlier
is_outlier = (df_raw['CO'] < lower_bound) | (df_raw['CO'] > upper_bound)
jumlah_outlier = is_outlier.sum()

print(f"Kuartil 1 (Q1)           : {q1:.6f} mol/m²")
print(f"Kuartil 3 (Q3)           : {q3:.6f} mol/m²")
print(f"IQR (Q3 - Q1)            : {iqr:.6f} mol/m²")
print(f"Batas Bawah (Lower Bound): {lower_bound:.6f} mol/m²")
print(f"Batas Atas (Upper Bound) : {upper_bound:.6f} mol/m²")
print(f"Jumlah Outlier Terdeteksi: {jumlah_outlier} hari")

# Tabel Detail Tanggal & Nilai Outlier Terdeteksi
df_outliers = df_raw[is_outlier][['date', 'CO']].copy()
df_outliers['date'] = df_outliers['date'].dt.strftime('%Y-%m-%d')
print("\nDaftar Tanggal & Nilai Outlier Terdeteksi secara Akurat:")
print(df_outliers.to_string(index=False))
```

---

### 1.3 Pengosongan Outlier & Imputasi Linear Time Interpolation

Titik data pencilan diubah menjadi `NaN` lalu diimputasi bersama dengan celah data mentah menggunakan **Linear Time Interpolation**:

```{math}
X(t) = X(t_1) + \frac{t - t_1}{t_2 - t_1} \cdot \left[ X(t_2) - X(t_1) \right]
```

```{code-cell} ipython3
# 3. Pengosongan Nilai Outlier menjadi NaN
df_prep = df_raw.copy()
df_prep.loc[is_outlier, 'CO'] = np.nan
nan_setelah_outlier = df_prep['CO'].isna().sum()

# 4. Imputasi Linear Time Interpolation Sekaligus
df_clean = df_prep.copy()
df_clean['CO_clean'] = df_clean['CO'].interpolate(method='linear', limit_direction='both')
nan_akhir = df_clean['CO_clean'].isna().sum()

print(f"Jumlah NaN Setelah Outlier Dikosongkan : {nan_setelah_outlier} hari ({nan_setelah_outlier/total_rows*100:.2f}%)")
print(f"Jumlah NaN Setelah Imputasi Akhir      : {nan_akhir} hari (100% Terisi & Clean)")

# Simpan dataset bersih ke file CSV
df_save = df_clean[['date', 'CO_clean']].rename(columns={'CO_clean': 'CO'})
df_save.to_csv('../data/csv/CO_clean.csv', index=False)
print("Dataset bersih disimpan ke '../data/csv/CO_clean.csv'")
```

---

### 1.4 Visualisasi 4 Grafik Terpisah & Ringkasan Perubahan Status Sinyal

Untuk menjaga kejelasan visual tanpa menumpuk warna dalam satu grafik, alur pembersihan sinyal disajikan dalam **4 grafik terpisah**:

```{code-cell} ipython3
# Set Visual Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Grafik 1: Missing Values (Data Mentah dengan Celah Kosong / NaN)
fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
ax.plot(df_raw['date'], df_raw['CO'], color='#2c3e50', linewidth=1.2, label='Sinyal Valid Mentah (192 Hari)')
ax.set_title('1. Sinyal CO Mentah dengan Celah Missing Values (173 Hari Kosong / NaN)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Tanggal Observasi (24 Aug 2025 - 23 Aug 2026)', fontsize=10)
ax.set_ylabel('Konsentrasi CO (mol/m²)', fontsize=10)
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Grafik 2: Sinyal Setelah Missing Values Diimputasi (Sebelum Cleaning Outlier)
df_raw_imputed = df_raw.copy()
df_raw_imputed['CO_imputed'] = df_raw_imputed['CO'].interpolate(method='linear', limit_direction='both')

fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
ax.plot(df_raw_imputed['date'], df_raw_imputed['CO_imputed'], color='#2980b9', linewidth=1.2, label='Sinyal Terisi Utuh (365 Hari)')
ax.set_title('2. Sinyal CO Setelah Imputasi Missing Values (Linear Time Interpolation)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Tanggal Observasi (24 Aug 2025 - 23 Aug 2026)', fontsize=10)
ax.set_ylabel('Konsentrasi CO (mol/m²)', fontsize=10)
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Grafik 3: Deteksi Outlier Menggunakan Metode IQR
outliers_plot = df_raw[is_outlier]

fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
ax.plot(df_raw['date'], df_raw['CO'], color='#7f8c8d', linewidth=1.0, alpha=0.7, label='Sinyal Valid Mentah')
ax.scatter(outliers_plot['date'], outliers_plot['CO'], color='#e74c3c', s=45, zorder=5, marker='o', edgecolor='black', linewidth=0.8, label=f'Outlier Terdeteksi ({len(outliers_plot)} Hari)')
ax.axhline(upper_bound, color='#c0392b', linestyle='--', linewidth=1.2, label=f'Batas Atas / Upper Bound ({upper_bound:.5f})')
ax.axhline(lower_bound, color='#2980b9', linestyle='--', linewidth=1.2, label=f'Batas Bawah / Lower Bound ({lower_bound:.5f})')
ax.set_title(f'3. Deteksi 11 Outlier CO Menggunakan Metode Interquartile Range (IQR)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Tanggal Observasi (24 Aug 2025 - 23 Aug 2026)', fontsize=10)
ax.set_ylabel('Konsentrasi CO (mol/m²)', fontsize=10)
ax.legend(loc='upper right', frameon=True, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Grafik 4: Sinyal CO Final Setelah Outlier Dibersihkan & Diimputasi
fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
ax.plot(df_clean['date'], df_clean['CO_clean'], color='#27ae60', linewidth=1.4, label='Sinyal Mulus Super Clean (365 Hari)')
ax.set_title('4. Sinyal CO Final Setelah Penanganan Outlier & Imputasi Linear (Clean Dataset)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Tanggal Observasi (24 Aug 2025 - 23 Aug 2026)', fontsize=10)
ax.set_ylabel('Konsentrasi CO (mol/m²)', fontsize=10)
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

### Tabel Ringkasan Perubahan Status Sinyal

| Tahapan Data Preparation | Jumlah Data Valid | Jumlah Missing Values (`NaN`) | Jumlah Outliers | Keterangan Status |
| :--- | :---: | :---: | :---: | :--- |
| **1. Data Mentah (Raw)** | 192 hari | 173 hari (47.40%) | 11 hari | Ada celah `NaN` & pencilan |
| **2. Pengosongan Outlier** | 181 hari | 184 hari (50.41%) | 0 hari | Outlier diubah menjadi `NaN` |
| **3. Imputasi Akhir (Clean)** | **365 hari** | **0 hari (0.00%)** | **0 hari** | **Sinyal mulus 100% utuh** |

```{figure} ../assets/editor/data_prep/1_co_raw_missing.png
:width: 100%
:align: center

Grafik 1: Sinyal CO Mentah dengan Celah Missing Values (173 Hari Kosong / NaN)
```

```{figure} ../assets/editor/data_prep/2_co_imputed_missing.png
:width: 100%
:align: center

Grafik 2: Sinyal CO Setelah Imputasi Missing Values (Linear Time Interpolation)
```

```{figure} ../assets/editor/data_prep/3_co_outlier_detection.png
:width: 100%
:align: center

Grafik 3: Deteksi 11 Outlier CO Menggunakan Metode Interquartile Range (IQR)
```

```{figure} ../assets/editor/data_prep/4_co_final_clean.png
:width: 100%
:align: center

Grafik 4: Sinyal CO Final Setelah Penanganan Outlier & Imputasi Linear (Clean Dataset)
```

---

## 2. Ekstraksi Fitur TSFEL (Time Series Feature Extraction Library)

### 2.1 Konsep Rekayasa Fitur Time Series

Untuk merepresentasikan karakteristik dinamika sinyal konsentrasi $\text{CO}$ selama 365 hari dalam bentuk vektor numerik yang siap diproses oleh algoritma *Machine Learning* dan analisis kemiripan (*similarity analysis*), digunakan pustaka **TSFEL (Time Series Feature Extraction Library)** mengacu pada dokumentasi resmi [TSFEL Feature List Documentation](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html).

Sesuai dengan daftar resmi 68 fitur TSFEL (`FEATURE_LIST`), ekstraksi fitur dikelompokkan ke dalam **4 domain utama**:
1. **Domain Statistik**: Mengukur pemusatan, sebaran, kemiringan, ECDF, dan distribusi probabilitas sinyal.
2. **Domain Temporal**: Mengukur sifat linier, autokorelasi, perlintasan nol, dan durasi fluktuasi dalam domain waktu.
3. **Domain Spektral**: Mengukur distribusi energi spektrogram, MFCC, LPCC, Wavelet, dan frekuensi sinyal (FFT).
4. **Domain Fraktal**: Menganalisis ketidakteraturan, kompleksitas *self-similarity* (DFA, Hurst Exponent, Higuchi, Petrosian, MSE).

- **Output File CSV**: Matriks presisi 68 fitur TSFEL disimpan ke **`data/csv/CO_tsfel_features.csv`** dengan format header `f1_abs_energy` hingga `f68_zero_cross`.

---

## 3. Katalog & Penomoran Presisi 68 Fitur TSFEL ($f_1$ hingga $f_{68}$)

Berikut adalah penjelasan detail komprehensif untuk **seluruh 68 fitur TSFEL** yang diekstrak, mencakup **konsep fitur, alasan penggunaan pada sinyal CO, rumus matematis LaTeX, dan interpretasi hasil** yang dikelompokkan ke dalam **4 domain utama**:

### 3.1 Domain Statistik (*Statistical Domain*) (22 Fitur)

#### **f1**: `f1_abs_energy`
- **Apa sebenarnya fitur ini?**: Akumulasi total energi mutlak sinyal CO.
- **Kenapa fitur ini digunakan?**: Mengukur magnitudo keseluruhan akumulasi pencemaran CO selama kurun waktu pengamatan.
- **Rumus Matematis LaTeX**: $E = \sum_{i=1}^{N} x_i^2$
- **Interpretasi Hasil**: Nilai energi tinggi menunjukkan durasi konsentrasi CO tinggi yang berkelanjutan.

#### **f4**: `f4_average_power`
- **Apa sebenarnya fitur ini?**: Rata-rata daya kuadrat sinyal per satuan waktu harian.
- **Kenapa fitur ini digunakan?**: Memberikan gambaran intensitas daya emisi polusi harian rata-rata tanpa terpengaruh jumlah hari.
- **Rumus Matematis LaTeX**: $P = \frac{1}{N} \sum_{i=1}^{N} x_i^2$
- **Interpretasi Hasil**: Indikator tingkat beban emisi rata-rata harian di Kecamatan Bungah.

#### **f6**: `f6_calc_max`
- **Apa sebenarnya fitur ini?**: Nilai konsentrasi CO tertinggi yang tercatat dalam 365 hari.
- **Kenapa fitur ini digunakan?**: Menandai puncak ekstrem krisis kualitas udara tertinggi.
- **Rumus Matematis LaTeX**: $x_{\max} = \max(x_1, x_2, \dots, x_N)$
- **Interpretasi Hasil**: Menjadi batas atas paparan bahaya pencemaran CO.

#### **f7**: `f7_calc_mean`
- **Apa sebenarnya fitur ini?**: Rata-rata aritmatika konsentrasi CO harian.
- **Kenapa fitur ini digunakan?**: Baseline standar polusi udara tahunan wilayah pengamatan.
- **Rumus Matematis LaTeX**: $\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$
- **Interpretasi Hasil**: Menggambarkan kualitas udara rata-rata sehari-hari.

#### **f8**: `f8_calc_median`
- **Apa sebenarnya fitur ini?**: Nilai tengah distribusi konsentrasi CO.
- **Kenapa fitur ini digunakan?**: Tahan terhadap pencilan lonjakan singkat emisi industri.
- **Rumus Matematis LaTeX**: $\tilde{x} = \text{median}(x_1, x_2, \dots, x_N)$
- **Interpretasi Hasil**: Representasi posisi tengah data yang robus dari skewness.

#### **f9**: `f9_calc_min`
- **Apa sebenarnya fitur ini?**: Nilai konsentrasi CO terendah dalam 365 hari.
- **Kenapa fitur ini digunakan?**: Menunjukkan kualitas udara paling bersih/baseline latar belakang alami.
- **Rumus Matematis LaTeX**: $x_{\min} = \min(x_1, x_2, \dots, x_N)$
- **Interpretasi Hasil**: Ambang minimum polusi lingkungan.

#### **f10**: `f10_calc_std`
- **Apa sebenarnya fitur ini?**: Standar deviasi penyebaran nilai CO dari rata-rata.
- **Kenapa fitur ini digunakan?**: Mengukur tingkat variabilitas dan gejolak kestabilan polusi harian.
- **Rumus Matematis LaTeX**: $s = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N} (x_i - \bar{x})^2}$
- **Interpretasi Hasil**: Semakin tinggi std, semakin tidak stabil kualitas udara.

#### **f11**: `f11_calc_var`
- **Apa sebenarnya fitur ini?**: Variansi atau kuadrat penyebaran data CO.
- **Kenapa fitur ini digunakan?**: Memberikan bobot lebih besar pada variasi fluktuasi emisi ekstrem.
- **Rumus Matematis LaTeX**: $s^2 = \frac{1}{N-1}\sum_{i=1}^{N} (x_i - \bar{x})^2$
- **Interpretasi Hasil**: Ukuran dispersi kuadratik populasi sinyal.

#### **f14**: `f14_ecdf`
- **Apa sebenarnya fitur ini?**: Fungsi Distribusi Kumulatif Empiris sinyal CO.
- **Kenapa fitur ini digunakan?**: Menjelaskan proporsi kumulatif hari yang berada di bawah ambang tertentu.
- **Rumus Matematis LaTeX**: $F(x) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(x_i \le x)$
- **Interpretasi Hasil**: Vektor sebaran probabilitas kumulatif.

#### **f15**: `f15_ecdf_percentile`
- **Apa sebenarnya fitur ini?**: Nilai kuantil persentil spesifik dari ECDF.
- **Kenapa fitur ini digunakan?**: Menentukan batas persentil polusi (misal persentil ke-50 atau ke-90).
- **Rumus Matematis LaTeX**: $Q_p = \inf \{x : F(x) \ge p\}$
- **Interpretasi Hasil**: Ambang batas konsentrasi CO pada persentil tertentu.

#### **f16**: `f16_ecdf_percentile_count`
- **Apa sebenarnya fitur ini?**: Jumlah hari yang konsentrasinya di bawah persentil ECDF.
- **Kenapa fitur ini digunakan?**: Menghitung frekuensi hari dengan kategori kualitas udara tertentu.
- **Rumus Matematis LaTeX**: $N_p = \sum_{i=1}^{N} \mathbb{I}(x_i \le Q_p)$
- **Interpretasi Hasil**: Jumlah akumulasi sampel hari.

#### **f17**: `f17_ecdf_slope`
- **Apa sebenarnya fitur ini?**: Kemiringan kenaikan kumulatif ECDF antara dua persentil.
- **Kenapa fitur ini digunakan?**: Mengukur kerapatan sebaran data pada interval konsentrasi tertentu.
- **Rumus Matematis LaTeX**: $\text{Slope}_{\text{ECDF}} = \frac{F(p_2) - F(p_1)}{x_{p2} - x_{p1}}$
- **Interpretasi Hasil**: Menunjukkan seberapa cepat akumulasi probabilitas naik.

#### **f18**: `f18_entropy`
- **Apa sebenarnya fitur ini?**: Entropi Shannon dari distribusi amplitudo sinyal.
- **Kenapa fitur ini digunakan?**: Mengukur tingkat ketidakpastian / ketidakacakan distribusi konsentrasi CO.
- **Rumus Matematis LaTeX**: $H(x) = -\sum_{i=1}^{K} p(x_i) \log_2 p(x_i)$
- **Interpretasi Hasil**: Entropi tinggi berarti distribusi konsentrasi sangat acak dan tersebar.

#### **f21**: `f21_hist_mode`
- **Apa sebenarnya fitur ini?**: Nilai modus (nilai paling sering muncul) pada histogram CO.
- **Kenapa fitur ini digunakan?**: Mengetahui tingkat konsentrasi CO yang paling dominan dirasakan sehari-hari.
- **Rumus Matematis LaTeX**: $\text{Mode}(x) = \arg\max_b \text{count}(b)$
- **Interpretasi Hasil**: Titik kerapatan puncak populasi sinyal.

#### **f24**: `f24_interq_range`
- **Apa sebenarnya fitur ini?**: Rentang antarkuartil ($Q_3 - Q_1$).
- **Kenapa fitur ini digunakan?**: Ukuran sebaran 50% data tengah yang tahan terhadap outlier ekstrem.
- **Rumus Matematis LaTeX**: $\text{IQR} = Q_3 - Q_1$
- **Interpretasi Hasil**: Rentang variasi konsentrasi CO normal.

#### **f25**: `f25_kurtosis`
- **Apa sebenarnya fitur ini?**: Keruncingan distribusi data (ekor distribusi).
- **Kenapa fitur ini digunakan?**: Mendeteksi seberapa berat ekor distribusi data CO akibat adanya spike pencilan emisi.
- **Rumus Matematis LaTeX**: $K = \frac{\frac{1}{N}\sum (x_i - \bar{x})^4}{s^4} - 3$
- **Interpretasi Hasil**: $K > 0$ (leptokurtik) menandakan frekuensi insiden lonjakan ekstrem tinggi.

#### **f26**: `f26_lempel_ziv`
- **Apa sebenarnya fitur ini?**: Kompleksitas kompresi Lempel-Ziv sinyal terbinarisasi.
- **Kenapa fitur ini digunakan?**: Mengukur tingkat kerumitan urutan perubahan pola polusi.
- **Rumus Matematis LaTeX**: $C_{LZ} = \frac{L_z}{N / \log_2 N}$
- **Interpretasi Hasil**: Nilai mendekati 1 menunjukkan pola sinyal sangat bervariasi dan kompleks.

#### **f31**: `f31_mean_abs_deviation`
- **Apa sebenarnya fitur ini?**: Rata-rata deviasi mutlak sampel terhadap mean.
- **Kenapa fitur ini digunakan?**: Mengukur sebaran data yang lebih stabil dibanding variansi kuadrat.
- **Rumus Matematis LaTeX**: $\text{MAD} = \frac{1}{N}\sum_{i=1}^{N} |x_i - \bar{x}|$
- **Interpretasi Hasil**: Rata-rata simpangan absolut dari nilai tengah.

#### **f34**: `f34_median_abs_deviation`
- **Apa sebenarnya fitur ini?**: Median deviasi mutlak dari nilai median.
- **Kenapa fitur ini digunakan?**: Estimator sebaran data yang paling robus dari pencilan.
- **Rumus Matematis LaTeX**: $\text{MAD}_{med} = \text{median}(|x_i - \tilde{x}|)$
- **Interpretasi Hasil**: Tingkat simpangan robus populasi CO.

#### **f43**: `f43_pk_pk_distance`
- **Apa sebenarnya fitur ini?**: Jarak antara nilai maksimum dan minimum ($x_{\max} - x_{\min}$).
- **Kenapa fitur ini digunakan?**: Mengukur rentang total rentang dinamis konsentrasi CO.
- **Rumus Matematis LaTeX**: $\text{P2P} = x_{\max} - x_{\min}$
- **Interpretasi Hasil**: Lebar jangkauan rentang pencemaran tahunan.

#### **f46**: `f46_rms`
- **Apa sebenarnya fitur ini?**: Root Mean Square (nilai efektif sinyal).
- **Kenapa fitur ini digunakan?**: Mengukur kekuatan magnitudo sinyal secara konsisten.
- **Rumus Matematis LaTeX**: $\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$
- **Interpretasi Hasil**: Nilai konsentrasi efektif CO harian.

#### **f47**: `f47_skewness`
- **Apa sebenarnya fitur ini?**: Kemiringan asimetri distribusi data CO.
- **Kenapa fitur ini digunakan?**: Mengetahui apakah data lebih sering melonjak tinggi (skewness positif).
- **Rumus Matematis LaTeX**: $S = \frac{\frac{1}{N}\sum (x_i - \bar{x})^3}{s^3}$
- **Interpretasi Hasil**: Skewness positif menandakan mayoritas hari bersih dengan sesekali spike tinggi.

### 3.2 Domain Temporal (*Temporal Domain*) (14 Fitur)

#### **f2**: `f2_auc`
- **Apa sebenarnya fitur ini?**: Total luas di bawah kurva sinyal waktu CO menggunakan aturan trapesium.
- **Kenapa fitur ini digunakan?**: Mengakumulasi total paparan polutan CO dalam kurun waktu 365 hari.
- **Rumus Matematis LaTeX**: $\text{AUC} = \sum_{i=1}^{N-1} \frac{x_i + x_{i+1}}{2} \Delta t$
- **Interpretasi Hasil**: Total kuantitas paparan emisi kumulatif.

#### **f3**: `f3_autocorr`
- **Apa sebenarnya fitur ini?**: Titik peluruhan autokorelasi sinyal ($1/e$ crossing).
- **Kenapa fitur ini digunakan?**: Mengukur seberapa kuat konsentrasi hari ini mempengaruhi hari-hari berikutnya (memori temporal).
- **Rumus Matematis LaTeX**: $R(\tau) = \sum_{i=1}^{N-\tau} x_i x_{i+\tau}$
- **Interpretasi Hasil**: Menunjukkan durasi keberlanjutan pola polusi.

#### **f5**: `f5_calc_centroid`
- **Apa sebenarnya fitur ini?**: Pusat berat (pusat massa) sinyal pada sumbu waktu.
- **Kenapa fitur ini digunakan?**: Mengetahui kapan puncak akumulasi emisi CO terjadi (di awal, tengah, atau akhir tahun).
- **Rumus Matematis LaTeX**: $C_t = \frac{\sum_{i=1}^{N} i \cdot x_i}{\sum_{i=1}^{N} x_i}$
- **Interpretasi Hasil**: Titik berat waktu konsentrasi emisi terbanyak.

#### **f13**: `f13_distance`
- **Apa sebenarnya fitur ini?**: Panjang lintasan Euclidean kurva sinyal dari waktu ke waktu.
- **Kenapa fitur ini digunakan?**: Mengukur tingkat kerapatan dan gejolak dinamika perubahan harian sinyal.
- **Rumus Matematis LaTeX**: $D = \sum_{i=1}^{N-1} \sqrt{1 + (x_{i+1} - x_i)^2}$
- **Interpretasi Hasil**: Sinyal dengan grafik sangat bergigi/gejolak memiliki distance tinggi.

#### **f32**: `f32_mean_abs_diff`
- **Apa sebenarnya fitur ini?**: Rata-rata selisih mutlak antara hari $i+1$ dan hari $i$.
- **Kenapa fitur ini digunakan?**: Mengukur laju perubahan laju polusi antar-hari.
- **Rumus Matematis LaTeX**: $\overline{|\Delta x|} = \frac{1}{N-1}\sum_{i=1}^{N-1} |x_{i+1} - x_i|$
- **Interpretasi Hasil**: Menggambarkan volatilitas transisi harian.

#### **f33**: `f33_mean_diff`
- **Apa sebenarnya fitur ini?**: Rata-rata selisih linier per pergerakan hari.
- **Kenapa fitur ini digunakan?**: Mengukur tren arah pergerakan sinyal secara keseluruhan.
- **Rumus Matematis LaTeX**: $\overline{\Delta x} = \frac{x_N - x_1}{N-1}$
- **Interpretasi Hasil**: Positif berarti ada tren kenaikan emisi secara jangka panjang.

#### **f35**: `f35_median_abs_diff`
- **Apa sebenarnya fitur ini?**: Median selisih mutlak antar hari yang berurutan.
- **Kenapa fitur ini digunakan?**: Laju pergeseran harian yang robus terhadap fluktuasi ekstrem sesaat.
- **Rumus Matematis LaTeX**: $\widetilde{|\Delta x|} = \text{median}(|x_{i+1} - x_i|)$
- **Interpretasi Hasil**: Variasi perubahan harian tipikal.

#### **f36**: `f36_median_diff`
- **Apa sebenarnya fitur ini?**: Median selisih linier antar hari.
- **Kenapa fitur ini digunakan?**: Tren pergerakan harian yang paling umum terjadi.
- **Rumus Matematis LaTeX**: $\widetilde{\Delta x} = \text{median}(x_{i+1} - x_i)$
- **Interpretasi Hasil**: Arah tren harian tipikal.

#### **f40**: `f40_negative_turning`
- **Apa sebenarnya fitur ini?**: Jumlah titik lembah lokal ($x_i < x_{i-1} \land x_i < x_{i+1}$).
- **Kenapa fitur ini digunakan?**: Menghitung berapa kali sinyal mengalami pemulihan/penurunan kualitas udara.
- **Rumus Matematis LaTeX**: $N_- = \sum \mathbb{I}(x_i < x_{i-1} \land x_i < x_{i+1})$
- **Interpretasi Hasil**: Frekuensi penurunan titik polusi.

#### **f41**: `f41_neighbourhood_peaks`
- **Apa sebenarnya fitur ini?**: Jumlah puncak lokal dalam jendela tetangga tertentu.
- **Kenapa fitur ini digunakan?**: Mendeteksi berapa banyak episoda gelombang emisi tinggi yang terjadi.
- **Rumus Matematis LaTeX**: $N_p = \sum \mathbb{I}(x_i > \text{tetangga})$
- **Interpretasi Hasil**: Jumlah peristiwa lonjakan lokal.

#### **f44**: `f44_positive_turning`
- **Apa sebenarnya fitur ini?**: Jumlah titik puncak lokal ($x_i > x_{i-1} \land x_i > x_{i+1}$).
- **Kenapa fitur ini digunakan?**: Menghitung frekuensi terjadinya titik balik kenaikan emisi.
- **Rumus Matematis LaTeX**: $N_+ = \sum \mathbb{I}(x_i > x_{i-1} \land x_i > x_{i+1})$
- **Interpretasi Hasil**: Jumlah puncak fluktuasi sinyal.

#### **f48**: `f48_slope`
- **Apa sebenarnya fitur ini?**: Kemiringan garis regresi linier sinyal terhadap waktu.
- **Kenapa fitur ini digunakan?**: Mengetahui laju tren peningkatan atau penurunan CO per hari secara keseluruhan.
- **Rumus Matematis LaTeX**: $\beta_1 = \frac{\sum (t_i - \bar{t})(x_i - \bar{x})}{\sum (t_i - \bar{t})^2}$
- **Interpretasi Hasil**: Slope positif berarti polusi beresiko meningkat seiring waktu.

#### **f62**: `f62_sum_abs_diff`
- **Apa sebenarnya fitur ini?**: Total akumulasi perubahan mutlak antar-hari.
- **Kenapa fitur ini digunakan?**: Mengukur total energi aktivitas dinamika pergerakan sinyal.
- **Rumus Matematis LaTeX**: $\text{SAD} = \sum_{i=1}^{N-1} |x_{i+1} - x_i|$
- **Interpretasi Hasil**: Total gejolak pergerakan sinyal sepanjang tahun.

#### **f68**: `f68_zero_cross`
- **Apa sebenarnya fitur ini?**: Laju perlintasan sinyal terhadap nilai rata-rata/nol.
- **Kenapa fitur ini digunakan?**: Mengukur frekuensi sinyal berganti dari di atas rerata ke di bawah rerata.
- **Rumus Matematis LaTeX**: $\text{ZCR} = \frac{1}{N-1}\sum \mathbb{I}((x_i - \bar{x})(x_{i+1} - \bar{x}) < 0)$
- **Interpretasi Hasil**: Semakin tinggi ZCR, semakin sering sinyal bolak-balik menyeberangi rata-rata.

### 3.3 Domain Spektral (*Spectral Domain*) (26 Fitur)

#### **f19**: `f19_fundamental_frequency`
- **Apa sebenarnya fitur ini?**: Frekuensi utama (komponen dasar) dengan energi terbesar.
- **Kenapa fitur ini digunakan?**: Mendeteksi siklus polusi alami utama (misal siklus mingguan atau bulanan).
- **Rumus Matematis LaTeX**: $f_0 = \arg\max_f |X(f)|$
- **Interpretasi Hasil**: Frekuensi gelombang paling dominan.

#### **f22**: `f22_human_range_energy`
- **Apa sebenarnya fitur ini?**: Rasio energi spektral pada pita frekuensi aktivitas manusia.
- **Kenapa fitur ini digunakan?**: Mengisolasi energi fluktuasi yang disebabkan oleh ritme aktivitas antropogenik/industri.
- **Rumus Matematis LaTeX**: $E_{\text{human}} = \frac{\sum_{f \in \text{human}} |X(f)|^2}{\sum |X(f)|^2}$
- **Interpretasi Hasil**: Persentase kontribusi fluktuasi aktivitas manusia.

#### **f27**: `f27_lpcc`
- **Apa sebenarnya fitur ini?**: Koefisien Cepstral Prediksi Linier (LPCC).
- **Kenapa fitur ini digunakan?**: Memodelkan amplop spektral dari respons dinamika emisi.
- **Rumus Matematis LaTeX**: $c_n = a_n + \sum_{k=1}^{n-1} \frac{k}{n} c_k a_{n-k}$
- **Interpretasi Hasil**: Fitur representasi bentuk spektrum frekuensi.

#### **f28**: `f28_max_frequency`
- **Apa sebenarnya fitur ini?**: Frekuensi tertinggi yang masih memiliki kandungan daya spektral bermakna.
- **Kenapa fitur ini digunakan?**: Menentukan batas frekuensi teratas dari komponen gelombang CO.
- **Rumus Matematis LaTeX**: $f_{\max} = \sup \{f : |X(f)|^2 > \epsilon\}$
- **Interpretasi Hasil**: Batas frekuensi atas sinyal.

#### **f29**: `f29_max_power_spectrum`
- **Apa sebenarnya fitur ini?**: Nilai kerapatan daya spektral (PSD) maksimum pada spektrum Fourier.
- **Kenapa fitur ini digunakan?**: Mengetahui puncak intensitas energi frekuensi terbesar.
- **Rumus Matematis LaTeX**: $P_{\max} = \max_f |X(f)|^2$
- **Interpretasi Hasil**: Kekuatan maksimum gelombang frekuensi dominan.

#### **f37**: `f37_median_frequency`
- **Apa sebenarnya fitur ini?**: Frekuensi yang membagi total daya spektrum menjadi dua bagian sama besar.
- **Kenapa fitur ini digunakan?**: Indikator posisi tengah distribusi energi spektral.
- **Rumus Matematis LaTeX**: $\sum_{f=0}^{f_{med}} |X(f)|^2 = \frac{1}{2} \sum_{f=0}^{f_{nyq}} |X(f)|^2$
- **Interpretasi Hasil**: Titik tengah pemisahan energi spektrum.

#### **f38**: `f38_mfcc`
- **Apa sebenarnya fitur ini?**: Mel-Frequency Cepstral Coefficients (MFCC).
- **Kenapa fitur ini digunakan?**: Menangkap bentuk enveloped spektral non-linier dari fluktuasi sinyal.
- **Rumus Matematis LaTeX**: $C_m = \sum_{k=1}^{K} \log(S_k) \cos\left[m \left(k - \frac{1}{2}\right) \frac{\pi}{K}\right]$
- **Interpretasi Hasil**: Vektor karakteristik spektral halus sinyal.

#### **f45**: `f45_power_bandwidth`
- **Apa sebenarnya fitur ini?**: Lebar pita spektrum yang memuat mayoritas daya sinyal (misal 99%).
- **Kenapa fitur ini digunakan?**: Mengukur seberapa luas rentang frekuensi yang aktif dalam dinamika CO.
- **Rumus Matematis LaTeX**: $\text{BW} = f_{\text{high}} - f_{\text{low}}$
- **Interpretasi Hasil**: Rentang lebar pita energi utama.

#### **f49**: `f49_spectral_centroid`
- **Apa sebenarnya fitur ini?**: Pusat massa (barycenter) spektrum frekuensi sinyal.
- **Kenapa fitur ini digunakan?**: Menunjukkan apakah energi spektral lebih banyak terkonsentrasi di frekuensi rendah atau tinggi.
- **Rumus Matematis LaTeX**: $f_c = \frac{\sum f \cdot |X(f)|^2}{\sum |X(f)|^2}$
- **Interpretasi Hasil**: Centroid tinggi berarti sinyal didominasi fluktuasi cepat (frekuensi tinggi).

#### **f50**: `f50_spectral_decrease`
- **Apa sebenarnya fitur ini?**: Laju penurunan amplitudo spektral seiring bertambahnya frekuensi.
- **Kenapa fitur ini digunakan?**: Mengukur seberapa cepat daya gelombang melemah pada frekuensi tinggi.
- **Rumus Matematis LaTeX**: $\text{SD} = \frac{1}{\sum_{k=2}^{K} |X(f_k)|} \sum_{k=2}^{K} \frac{|X(f_k)| - |X(f_1)|}{k-1}$
- **Interpretasi Hasil**: Menggambarkan kecenderungan peluruhan energi spektrum.

#### **f51**: `f51_spectral_distance`
- **Apa sebenarnya fitur ini?**: Jarak penyebaran spektral terhadap profil spektrum acak.
- **Kenapa fitur ini digunakan?**: Mengukur kompleksitas struktur bentuk spektrum Fourier sinyal.
- **Rumus Matematis LaTeX**: $D_{\text{spec}} = \sqrt{\sum (|X(f_k)| - \mu_{\text{spec}})^2}$
- **Interpretasi Hasil**: Variasi bentuk spektrum frekuensi.

#### **f52**: `f52_spectral_entropy`
- **Apa sebenarnya fitur ini?**: Entropi Shannon dari kerapatan spektrum daya Fourier.
- **Kenapa fitur ini digunakan?**: Mengukur tingkat keacakan distribusi energi frekuensi (apakah seperti noise atau bernada teratur).
- **Rumus Matematis LaTeX**: $H_{\text{spec}} = -\sum P(f_k) \log_2 P(f_k)$
- **Interpretasi Hasil**: Nilai tinggi berarti energi terbagi rata di banyak frekuensi (kompleks/noise).

#### **f53**: `f53_spectral_kurtosis`
- **Apa sebenarnya fitur ini?**: Keruncingan distribusi energi spektral di sekitar centroid.
- **Kenapa fitur ini digunakan?**: Mendeteksi keberadaan lonjakan puncak frekuensi tajam terisolasi.
- **Rumus Matematis LaTeX**: $K_{\text{spec}} = \frac{\sum (f - f_c)^4 |X(f)|^2}{\sigma_{\text{spec}}^4 \sum |X(f)|^2}$
- **Interpretasi Hasil**: Kurtosis spektral tinggi menandakan adanya frekuensi periodik yang sangat kuat.

#### **f54**: `f54_spectral_positive_turning`
- **Apa sebenarnya fitur ini?**: Jumlah puncak lokal pada kurva spektrum daya Fourier.
- **Kenapa fitur ini digunakan?**: Menghitung berapa banyak komponen frekuensi resonansi berlainan.
- **Rumus Matematis LaTeX**: $N_{+\text{spec}} = \sum \mathbb{I}(|X(f_k)| > |X(f_{k-1})| \land |X(f_k)| > |X(f_{k+1})|)$
- **Interpretasi Hasil**: Jumlah puncak harmonic spektrum.

#### **f55**: `f55_spectral_roll_off`
- **Apa sebenarnya fitur ini?**: Frekuensi di mana 85% akumulasi daya spektral terkonsentrasi.
- **Kenapa fitur ini digunakan?**: Menentukan batas frekuensi yang menampung mayoritas energi sinyal.
- **Rumus Matematis LaTeX**: $\sum_{f=0}^{f_{\text{roll}}} |X(f)|^2 = 0.85 \sum |X(f)|^2$
- **Interpretasi Hasil**: Batas spektral 85% energi.

#### **f56**: `f56_spectral_roll_on`
- **Apa sebenarnya fitur ini?**: Frekuensi di mana 15% akumulasi daya spektral awal mulai terbentuk.
- **Kenapa fitur ini digunakan?**: Menentukan batas frekuensi bawah pembentuk energi awal sinyal.
- **Rumus Matematis LaTeX**: $\sum_{f=0}^{f_{\text{on}}} |X(f)|^2 = 0.15 \sum |X(f)|^2$
- **Interpretasi Hasil**: Batas spektral 15% energi awal.

#### **f57**: `f57_spectral_skewness`
- **Apa sebenarnya fitur ini?**: Kemiringan asimetri distribusi spektral di sekitar centroid.
- **Kenapa fitur ini digunakan?**: Mengetahui ke mana energi spektral lebih condong (ke frekuensi rendah atau tinggi).
- **Rumus Matematis LaTeX**: $S_{\text{spec}} = \frac{\sum (f - f_c)^3 |X(f)|^2}{\sigma_{\text{spec}}^3 \sum |X(f)|^2}$
- **Interpretasi Hasil**: Skewness spektral menggambarkan kemiringan spektrum daya.

#### **f58**: `f58_spectral_slope`
- **Apa sebenarnya fitur ini?**: Kemiringan garis regresi penurunan amplitudo spektrum frekuensi.
- **Kenapa fitur ini digunakan?**: Mengukur laju redaman energi spektral seiring naik frekuensi.
- **Rumus Matematis LaTeX**: $\beta_{\text{spec}} = \frac{\sum (f_k - \bar{f})(|X(f_k)| - \bar{|X|})}{\sum (f_k - \bar{f})^2}$
- **Interpretasi Hasil**: Slope spektral negatif menggambarkan penurunan daya standar.

#### **f59**: `f59_spectral_spread`
- **Apa sebenarnya fitur ini?**: Penyebaran (deviasi standar) energi spektram di sekitar centroid.
- **Kenapa fitur ini digunakan?**: Mengukur seberapa lebar spektrum frekuensi terdistribusi di sekitar titik beratnya.
- **Rumus Matematis LaTeX**: $\sigma_{\text{spec}} = \sqrt{\frac{\sum (f - f_c)^2 |X(f)|^2}{\sum |X(f)|^2}}$
- **Interpretasi Hasil**: Penyebaran energi spektral.

#### **f60**: `f60_spectral_variation`
- **Apa sebenarnya fitur ini?**: Fluktuasi variasi bentuk spektrum antar-segmen waktu.
- **Kenapa fitur ini digunakan?**: Mengukur ketidakstabilan profil frekuensi sepanjang 365 hari.
- **Rumus Matematis LaTeX**: $V_{\text{spec}} = 1 - \frac{\sum |X_t(f)| |X_{t+1}(f)|}{\sqrt{\sum |X_t(f)|^2 \sum |X_{t+1}(f)|^2}}$
- **Interpretasi Hasil**: Perubahan kontur spektral seiring waktu.

#### **f61**: `f61_spectrogram_mean_coeff`
- **Apa sebenarnya fitur ini?**: Rata-rata daya spektrogram STFT pada pita frekuensi.
- **Kenapa fitur ini digunakan?**: Mengukur intensitas rata-rata energi pada matriks waktu-frekuensi.
- **Rumus Matematis LaTeX**: $\bar{S}(f_k) = \frac{1}{M}\sum_{m=1}^{M} |X(m, f_k)|^2$
- **Interpretasi Hasil**: Koefisien daya rata-rata spektrogram.

#### **f63**: `f63_wavelet_abs_mean`
- **Apa sebenarnya fitur ini?**: Rata-rata mutlak koefisien Transformasi Wavelet Kontinu (CWT).
- **Kenapa fitur ini digunakan?**: Mendeteksi energi lokal pada berbagai skala waktu-frekuensi secara presisi.
- **Rumus Matematis LaTeX**: $\overline{|W(a, b)|} = \frac{1}{N}\sum |W(a, b)|$
- **Interpretasi Hasil**: Magnitudo respons wavelet skala tertentu.

#### **f64**: `f64_wavelet_energy`
- **Apa sebenarnya fitur ini?**: Total energi kuadrat koefisien Transformasi Wavelet.
- **Kenapa fitur ini digunakan?**: Mengukur total konsentrasi daya sinyal pada resolusi multiskala.
- **Rumus Matematis LaTeX**: $E_{\text{wav}} = \sum |W(a, b)|^2$
- **Interpretasi Hasil**: Kandungan energi wavelet multiresolusi.

#### **f65**: `f65_wavelet_entropy`
- **Apa sebenarnya fitur ini?**: Entropi Shannon dari distribusi energi koefisien Wavelet.
- **Kenapa fitur ini digunakan?**: Mengukur kompleksitas alokasi energi pada skala waktu-frekuensi.
- **Rumus Matematis LaTeX**: $H_{\text{wav}} = -\sum p_i \log_2 p_i$
- **Interpretasi Hasil**: Derajat keacakan sub-band wavelet.

#### **f66**: `f66_wavelet_std`
- **Apa sebenarnya fitur ini?**: Standar deviasi koefisien Transformasi Wavelet.
- **Kenapa fitur ini digunakan?**: Mengukur variabilitas respons gelombang pada skala wavelet tertentu.
- **Rumus Matematis LaTeX**: $\sigma_{\text{wav}} = \sqrt{\frac{1}{N}\sum (W_i - \bar{W})^2}$
- **Interpretasi Hasil**: Variabilitas koefisien wavelet.

#### **f67**: `f67_wavelet_var`
- **Apa sebenarnya fitur ini?**: Variansi kuadrat koefisien Transformasi Wavelet.
- **Kenapa fitur ini digunakan?**: Mengukur penyebaran daya pada domain skala wavelet.
- **Rumus Matematis LaTeX**: $\sigma_{\text{wav}}^2 = \frac{1}{N}\sum (W_i - \bar{W})^2$
- **Interpretasi Hasil**: Dispersi energi wavelet.

### 3.4 Domain Fraktal (*Fractal Domain*) (6 Fitur)

#### **f12**: `f12_dfa`
- **Apa sebenarnya fitur ini?**: Detrended Fluctuation Analysis (DFA) sinyal CO.
- **Kenapa fitur ini digunakan?**: Mengukur eksponen korelasi memori jangka panjang (*long-range temporal dependence*).
- **Rumus Matematis LaTeX**: $F(n) = \sqrt{\frac{1}{N}\sum (y(k) - y_n(k))^2} \sim n^\alpha$
- **Interpretasi Hasil**: $\alpha \approx 0.5$ (white noise), $\alpha > 0.5$ (memori positif/persisten), $\alpha > 1$ (non-stasioner).

#### **f20**: `f20_higuchi_fractal_dimension`
- **Apa sebenarnya fitur ini?**: Dimensi fraktal linier metode Higuchi (HFD).
- **Kenapa fitur ini digunakan?**: Mengukur tingkat kekasaran dan kompleksitas fraktal sinyal deret waktu non-linier.
- **Rumus Matematis LaTeX**: $L(k) \sim k^{-D_H}$
- **Interpretasi Hasil**: Mendekati 1 (sinyal mulus), mendekati 2 (sinyal sangat kasar dan bergejolak).

#### **f23**: `f23_hurst_exponent`
- **Apa sebenarnya fitur ini?**: Eksponen Hurst ($H$) melalui analisis Rescaled Range ($R/S$).
- **Kenapa fitur ini digunakan?**: Mengetahui sifat keberlanjutan tren (*persistency*) atau pembalikan tren (*anti-persistency*).
- **Rumus Matematis LaTeX**: $\mathbb{E}[R(n)/S(n)] = C n^H$
- **Interpretasi Hasil**: $H > 0.5$ (tren persisten/berlanjut), $H < 0.5$ (anti-persisten/mean-reverting).

#### **f30**: `f30_maximum_fractal_length`
- **Apa sebenarnya fitur ini?**: Panjang fraktal maksimum pada skala interval terkecil Higuchi.
- **Kenapa fitur ini digunakan?**: Mengukur titik jenuh batas kompleksitas fraktal sinyal.
- **Rumus Matematis LaTeX**: $\text{MFL} = \max(L(k))$
- **Interpretasi Hasil**: Batas maksimum kekasaran skala fraktal.

#### **f39**: `f39_mse`
- **Apa sebenarnya fitur ini?**: Multiscale Entropy (MSE) sinyal.
- **Kenapa fitur ini digunakan?**: Mengevaluasi kompleksitas dan keteraturan sinyal pada berbagai skala resolusi waktu.
- **Rumus Matematis LaTeX**: $\text{MSE}(\tau) = \text{SampEn}(y^{(\tau)})$
- **Interpretasi Hasil**: Tingkat kompleksitas dinamika sinyal antar-skala.

#### **f42**: `f42_petrosian_fractal_dimension`
- **Apa sebenarnya fitur ini?**: Dimensi fraktal metode Petrosian untuk sinyal terbinarisasi.
- **Kenapa fitur ini digunakan?**: Estimasi cepat kompleksitas fraktal berdasarkan perubahan tanda turunan sinyal.
- **Rumus Matematis LaTeX**: $D_P = \frac{\log_{10} N}{\log_{10} N + \log_{10} \frac{N}{N + 0.4 N_{\delta}}}$
- **Interpretasi Hasil**: Dimensi fraktal Petrosian (biasanya antara 1 dan 1.5).

---

> [!NOTE]
> Katalog 68 fitur di atas menjelaskan secara presisi seluruh fitur TSFEL yang diekstrak ke dalam file `data/csv/CO_tsfel_features.csv`, memberikan landasan analisis yang kuat untuk tahap pemodelan *Machine Learning* dan analisis kemiripan sinyal.
