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

# Data Preparation

Data Preparation adalah tahap ketiga dalam metodologi CRISP-DM yang bertujuan untuk **membersihkan**, **mengimputasi nilai kosong (*missing values*)**, **menangani pencilan (*outliers*)**, dan **merekayasa fitur (*feature engineering*)** dari deret waktu pengamatan kualitas udara polutan **Karbon Monoksida ($\text{CO}$)** pada skala geografis **Tingkat Kecamatan (Kecamatan Bungah)** selama 365 hari (24 Agustus 2025 – 23 Agustus 2026).

---

## 1. Penanganan Missing Values & Outliers

### 1.1 Latar Belakang & Pendekatan Data Cleaning
Data pengamatan satelit Sentinel-5P dari Copernicus Data Space untuk polutan $\text{CO}$ harian memiliki celah data (*missing values* / `NaN`) akibat tutupan awan tebal dan penyaringan validasi kualitas (*quality flag*). Selain itu, terdapat pula beberapa titik data pengamatan yang nilainya melonjak ekstrem (*outlier*) di luar batas wajar akibat interferensi cuaca lokal.

Untuk menghasilkan sinyal deret waktu yang mulus, mulus, dan kontinu 365 hari tanpa pencilan ekstrem maupun celah kosong, digunakan alur **Data Cleaning Terpadu**:
1. **Pemeriksaan Data Mentah**: Mengidentifikasi 173 hari data kosong (`NaN`) awal.
2. **Deteksi Outlier**: Menghitung batas Interquartile Range (IQR) pada data valid untuk mendeteksi 11 titik pencilan ekstrem.
3. **Pengosongan Outlier**: Menghapus/mengosongkan nilai 11 tanggal pencilan tersebut menjadi `NaN` (total missing values menjadi 184 hari).
4. **Imputasi Linear Sekaligus**: Melakukan **Linear Time Interpolation** secara bersamaan untuk 184 titik `NaN` sehingga diperoleh sinyal kontinu mulus 365 hari tanpa celah.

---

### 1.2 Deteksi Outlier Berbasis Interquartile Range (IQR)

Metode **Interquartile Range (IQR)** digunakan untuk menentukan batas wajar observasi:

1. **Kuartil Pertama ($Q_1$)**: Persentil ke-25 data terurut.
2. **Kuartil Ketiga ($Q_3$)**: Persentil ke-75 data terurut.
3. **Rentang Antarkuartil ($\text{IQR}$)**:

```{math}
\text{IQR} = Q_3 - Q_1
```

Batas bawah (*Lower Bound*) dan batas atas (*Upper Bound*) ditentukan melalui formula:

```{math}
\text{Batas Bawah} = Q_1 - 1.5 \times \text{IQR}
```

```{math}
\text{Batas Atas} = Q_3 + 1.5 \times \text{IQR}
```

Titik data $X_i$ dikategorikan sebagai **Outlier** jika:

```{math}
X_i < \text{Batas Bawah} \quad \text{atau} \quad X_i > \text{Batas Atas}
```

Nilai $X_i$ yang terdeteksi sebagai outlier dihapus/dikosongkan menjadi `NaN`:

```{math}
X_{\text{temp}, i} = \begin{cases}
\text{NaN}, & \text{jika } X_i < \text{Batas Bawah} \text{ atau } X_i > \text{Batas Atas} \\
X_i, & \text{lainnya}
\end{cases}
```

---

### 1.3 Formula & Mekanisme Interpolasi Linear

Setelah titik pencilan dikosongkan menjadi `NaN` bersama celah data mentah, dilakukan **Linear Time Interpolation**.

Interpolasi linear mengestimasi nilai sel kosong $X(t)$ pada tanggal $t$ yang berada di antara dua titik observasi valid terdekat $X(t_1)$ dan $X(t_2)$ dengan $t_1 < t < t_2$:

```{math}
X(t) = X(t_1) + \frac{t - t_1}{t_2 - t_1} \cdot \left[ X(t_2) - X(t_1) \right]
```

**Langkah Kerja Imputasi:**
1. Mengurutkan data observasi secara kronologis berdasarkan kolom tanggal `date`.
2. Mengisi seluruh nilai celah deret waktu (184 hari `NaN`) sehingga diperoleh 365 data kontinu tanpa *missing value*.
3. Menyimpan hasil sinyal bersih ke file CSV baru **`data/csv/CO_clean.csv`**.

---

## 2. Ekstraksi Fitur TSFEL (Time Series Feature Extraction Library)

### 2.1 Konsep Rekayasa Fitur Time Series

Untuk merepresentasikan karakteristik dinamika sinyal konsentrasi $\text{CO}$ selama 365 hari dalam bentuk vektor numerik yang siap diproses oleh algoritma *Machine Learning* dan analisis kemiripan (*similarity analysis*), digunakan pustaka **TSFEL (Time Series Feature Extraction Library)** mengacu pada dokumentasi resmi [TSFEL Feature List Documentation](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html).

TSFEL mengekstrak **68 fitur perwakilan** ($f_1$ hingga $f_{68}$) yang terbagi ke dalam 3 domain utama:
1. **Domain Statistik ($f_1 \dots f_{20}$)**: Mengukur karakteristik pemusatan, sebaran, kemiringan, dan distribusi probabilitas sinyal.
2. **Domain Temporal ($f_{21} \dots f_{41}$)**: Mengukur sifat linier, autokorelasi, frekuensi perlintasan nol, dan durasi fluktuasi dalam domain waktu.
3. **Domain Spektral ($f_{42} \dots f_{68}$)**: Mengukur distribusi energi frekuensi sinyal menggunakan transformasi Fourier (*Fast Fourier Transform* / FFT) dan Wavelet CWT.

---

## 3. Katalog & Penomoran Lengkap 68 Fitur TSFEL ($f_1$ hingga $f_{68}$)

Berikut adalah penomoran urut fitur $f_1$ hingga $f_{68}$ beserta fungsi resmi TSFEL, definisi, dan formula matematikalnya:

### 3.1 Domain Statistik ($f_1$ hingga $f_{20}$)

| Kode Fitur | Nama Fungsi TSFEL | Definisi Resmi & Cara Kerja | Cara Menghitung & Rumus Matematika |
| :---: | :--- | :--- | :--- |
| **$f_1$** | `abs_energy` | Menghitung energi mutlak dari keseluruhan sinyal deret waktu. | $E = \sum_{t=1}^{N} \vert X(t) \vert^2$ |
| **$f_2$** | `average_power` | Menghitung rata-rata daya sinyal per satuan waktu sampling harian. | $P = \frac{1}{N} \sum_{t=1}^{N} \vert X(t) \vert^2$ |
| **$f_3$** | `calc_max` | Menghitung nilai puncak maksimum dari sinyal deret waktu $\text{CO}$. | $\text{Max} = \max(X_t)$ |
| **$f_4$** | `calc_mean` | Menghitung nilai rata-rata aritmatika dari sinyal $\text{CO}$. | $\bar{X} = \frac{1}{N} \sum_{t=1}^{N} X_t$ |
| **$f_5$** | `calc_median` | Menghitung nilai median (persentil ke-50) dari sinyal $\text{CO}$. | $Me = X_{((N+1)/2)}$ |
| **$f_6$** | `calc_min` | Menghitung nilai minimum terendah dari sinyal $\text{CO}$. | $\text{Min} = \min(X_t)$ |
| **$f_7$** | `calc_std` | Menghitung nilai standar deviasi (simpangan baku) populasi sinyal. | $s = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(X_t - \bar{X})^2}$ |
| **$f_8$** | `calc_var` | Menghitung nilai variansi sebaran data sinyal dari rata-ratanya. | $s^2 = \frac{1}{N}\sum_{t=1}^{N}(X_t - \bar{X})^2$ |
| **$f_9$** | `ecdf` | Menghitung fungsi distribusi kumulatif empiris sepanjang sumbu waktu. | $F_N(x) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}_{X_i \le x}$ |
| **$f_{10}$** | `ecdf_percentile` | Menghitung nilai persentil ECDF dari distribusi kumulatif. | $Q(p) = \inf \{x : F_N(x) \ge p\}$ |
| **$f_{11}$** | `ecdf_percentile_count` | Menghitung jumlah sampel kumulatif yang nilainya lebih kecil dari persentil. | $C(p) = \sum_{i=1}^{N} \mathbf{1}_{X_i \le Q(p)}$ |
| **$f_{12}$** | `ecdf_slope` | Menghitung kemiringan (gradien) ECDF di antara dua persentil ($p_{\text{init}}, p_{\text{end}}$). | $\text{Slope}_{\text{ECDF}} = \frac{p_{\text{end}} - p_{\text{init}}}{Q(p_{\text{end}}) - Q(p_{\text{init}})}$ |
| **$f_{13}$** | `entropy` | Menghitung tingkat keacakan/ketidakpastian sinyal (Shannon Entropy). | $H(X) = -\sum p(x_i) \log_2 p(x_i)$ |
| **$f_{14}$** | `hist_mode` | Menghitung modus histogram dari pembagian interval bin teratur. | $\text{Mode}_{\text{hist}} = \arg\max_{b} (\text{bin}_b)$ |
| **$f_{15}$** | `interq_range` | Menghitung rentang antarkuartil ($\text{IQR} = Q_3 - Q_1$) dari sinyal. | $\text{IQR} = Q_3 - Q_1$ |
| **$f_{16}$** | `kurtosis` | Menghitung keruncingan puncak distribusi sinyal dibanding distribusi normal. | $K = \frac{\frac{1}{N}\sum(X_t - \bar{X})^4}{s^4} - 3$ |
| **$f_{17}$** | `mean_abs_deviation` | Menghitung rata-rata simpangan mutlak observasi dari rata-ratanya. | $\text{MAD} = \frac{1}{N} \sum_{t=1}^{N} \vert X_t - \bar{X} \vert$ |
| **$f_{18}$** | `median_abs_deviation` | Menghitung median dari selisih mutlak observasi terhadap median sinyal. | $\text{MedAD} = \text{median}(\vert X_t - Me \vert)$ |
| **$f_{19}$** | `rms` | Menghitung *Root Mean Square* (akar rata-rata kuadrat) amplitudo sinyal. | $\text{RMS} = \sqrt{\frac{1}{N}\sum_{t=1}^{N} X_t^2}$ |
| **$f_{20}$** | `skewness` | Menghitung kemiringan (asimetri) distribusi data terhadap rata-rata. | $S_k = \frac{\frac{1}{N}\sum(X_t - \bar{X})^3}{s^3}$ |

---

### 3.2 Domain Temporal ($f_{21}$ hingga $f_{41}$)

| Kode Fitur | Nama Fungsi TSFEL | Definisi Resmi & Cara Kerja | Cara Menghitung & Rumus Matematika |
| :---: | :--- | :--- | :--- |
| **$f_{21}$** | `autocorr` | Menghitung titik perlintasan $1/e$ pertama pada fungsi autokorelasi (ACF). | $R(k) = \frac{\sum (X_t - \bar{X})(X_{t+k} - \bar{X})}{\sum (X_t - \bar{X})^2} \to \text{Cari } k \text{ saat } R(k)=1/e$ |
| **$f_{22}$** | `calc_centroid` | Menghitung titik pusat massa (*barycenter*) sinyal sepanjang sumbu waktu. | $C_{\text{temp}} = \frac{\sum_{t=1}^{N} t \cdot X_t}{\sum_{t=1}^{N} X_t}$ |
| **$f_{23}$** | `dfa` | *Detrended Fluctuation Analysis* untuk mengukur ketergantungan jangka panjang. | $F(n) = \sqrt{\frac{1}{N}\sum_{y}(Y(k) - Y_n(k))^2} \sim n^\alpha$ |
| **$f_{24}$** | `distance` | Menghitung total jarak lintasan akumulatif yang ditempuh oleh sinyal. | $D = \sum_{t=1}^{N-1} \sqrt{1 + (X_{t+1} - X_t)^2}$ |
| **$f_{25}$** | `higuchi_fractal_dimension` | Menghitung dimensi fraktal sinyal menggunakan metode Higuchi (HFD). | $L(m, k) \propto k^{-D_H}$ |
| **$f_{26}$** | `hurst_exponent` | Menghitung eksponen Hurst melalui analisis *Rescaled Range* ($R/S$). | $(R/S)_n \propto n^H$ |
| **$f_{27}$** | `lempel_ziv` | Menghitung indeks kompleksitas Lempel-Ziv (LZ) yang diternormalisasi. | $C_{\text{LZ}} = \frac{c(N)}{N / \log_2(N)}$ |
| **$f_{28}$** | `maximum_fractal_length` | Menghitung panjang fraktal maksimum (MFL) pada skala terkecil. | $\text{MFL} = \text{mean}(L(m, k_{\min}))$ |
| **$f_{29}$** | `mean_abs_diff` | Menghitung rata-rata selisih mutlak antara sampel harian berurutan. | $\text{MAD}_{\text{diff}} = \frac{1}{N-1}\sum_{t=1}^{N-1} \vert X_{t+1} - X_t \vert$ |
| **$f_{30}$** | `mean_diff` | Menghitung rata-rata selisih linier harian antara sampel berurutan. | $\text{M}_{\text{diff}} = \frac{1}{N-1}\sum_{t=1}^{N-1} (X_{t+1} - X_t)$ |
| **$f_{31}$** | `median_abs_diff` | Menghitung median dari selisih mutlak observasi berurutan. | $\text{MedAD}_{\text{diff}} = \text{median}(\vert X_{t+1} - X_t \vert)$ |
| **$f_{32}$** | `median_diff` | Menghitung median dari selisih linier harian observasi berurutan. | $\text{Med}_{\text{diff}} = \text{median}(X_{t+1} - X_t)$ |
| **$f_{33}$** | `mse` | *Multiscale Entropy* yang mengukur kompleksitas pada berbagai skala waktu. | $\text{MSE}(m, r, \tau) = \text{SampleEntropy}(X^\tau, m, r)$ |
| **$f_{34}$** | `negative_turning` | Menghitung jumlah titik balik negatif (lembah lokal sinyal). | $N_{TP-} = \sum \mathbf{1}_{(X_t < X_{t-1} \land X_t < X_{t+1})}$ |
| **$f_{35}$** | `neighbourhood_peaks` | Menghitung jumlah puncak lokal dalam tetangga area dekat $n$. | $N_{\text{peaks}} = \sum \mathbf{1}_{(X_t > \max(X_{t-n..t+n}))}$ |
| **$f_{36}$** | `petrosian_fractal_dimension` | Menghitung dimensi fraktal Petrosian (PFD) dari turunan biner sinyal. | $D_P = \frac{\log_{10} N}{\log_{10} N + \log_{10}\left(\frac{N}{N + 0.4 N_{\Delta}}\right)}$ |
| **$f_{37}$** | `pk_pk_distance` | Menghitung jarak selisih amplitudo dari puncak maksimum ke minimum. | $P_{2P} = \max(X_t) - \min(X_t)$ |
| **$f_{38}$** | `positive_turning` | Menghitung jumlah titik balik positif (puncak lokal sinyal). | $N_{TP+} = \sum \mathbf{1}_{(X_t > X_{t-1} \land X_t > X_{t+1})}$ |
| **$f_{39}$** | `slope` | Menghitung gradien kemiringan regresi linier garis tren sinyal. | $\beta = \frac{\sum (t - \bar{t})(X_t - \bar{X})}{\sum (t - \bar{t})^2}$ |
| **$f_{40}$** | `sum_abs_diff` | Menghitung total akumulasi penjumlahan selisih mutlak harian. | $S_{\text{abs}} = \sum_{t=1}^{N-1} \vert X_{t+1} - X_t \vert$ |
| **$f_{41}$** | `zero_cross` | Menghitung laju perlintasan sinyal melewati angka nol (atau rata-rata). | $\text{ZCR} = \frac{1}{N-1}\sum \mathbf{1}_{(X_t \cdot X_{t+1} < 0)}$ |

---

### 3.3 Domain Spektral ($f_{42}$ hingga $f_{68}$)

| Kode Fitur | Nama Fungsi TSFEL | Definisi Resmi & Cara Kerja | Cara Menghitung & Rumus Matematika |
| :---: | :--- | :--- | :--- |
| **$f_{42}$** | `auc` | Menghitung luas di bawah kurva spektrum menggunakan aturan trapesium. | $\text{AUC} = \int X(t) dt \approx \sum \frac{X_t + X_{t+1}}{2} \Delta t$ |
| **$f_{43}$** | `fundamental_frequency` | Menghitung frekuensi fundamental dasar utama dari spektrum sinyal. | $f_0 = \arg\max_f \vert S(f) \vert$ |
| **$f_{44}$** | `human_range_energy` | Menghitung rasio energi spektral pada pita rentang aktivitas harian. | $E_{\text{human}} = \frac{\sum_{f \in \text{human}} \vert S(f) \vert^2}{\sum \vert S(f) \vert^2}$ |
| **$f_{45}$** | `lpcc` | Menghitung koefisien *Linear Prediction Cepstral* (LPCC). | $c_n = a_n + \sum_{k=1}^{n-1} \frac{k}{n} c_k a_{n-k}$ |
| **$f_{46}$** | `max_frequency` | Menghitung frekuensi tertinggi yang memiliki respon energi spektral. | $f_{\max} = \max \{f : \vert S(f) \vert > 0\}$ |
| **$f_{47}$** | `max_power_spectrum` | Menghitung nilai puncak kerapatan spektrum daya (*Power Spectrum Density*). | $\text{PSD}_{\max} = \max \vert S(f) \vert^2$ |
| **$f_{48}$** | `median_frequency` | Menghitung frekuensi yang membagi dua total energi spektrogram. | $\sum_{f=0}^{f_{\text{med}}} \vert S(f) \vert^2 = \frac{1}{2} \sum \vert S(f) \vert^2$ |
| **$f_{49}$** | `mfcc` | Menghitung koefisien spektral *Mel-Frequency Cepstral Coefficients* (MFCC). | $\text{MFCC}_m = \sum_{k=1}^{K} \log(S_k) \cos\left[ m \left(k - \frac{1}{2}\right) \frac{\pi}{K}\right]$ |
| **$f_{50}$** | `power_bandwidth` | Menghitung lebar pita frekuensi (*bandwidth*) dari spektrum daya. | $\text{BW} = f_{\text{high}} - f_{\text{low}}$ |
| **$f_{51}$** | `spectral_centroid` | Menghitung titik pusat massa (*barycenter*) dari spektrum frekuensi Fourier. | $C_{\text{spec}} = \frac{\sum f \cdot \vert S(f) \vert}{\sum \vert S(f) \vert}$ |
| **$f_{52}$** | `spectral_decrease` | Menghitung tingkat penurunan amplitudo spektrum frekuensi. | $S_{\text{dec}} = \frac{1}{\sum_{k=2}^{K} S(k)} \sum_{k=2}^{K} \frac{S(k) - S(1)}{k - 1}$ |
| **$f_{53}$** | `spectral_distance` | Menghitung jarak deviasi spektral energi relatif terhadap frekuensi. | $D_{\text{spec}} = \sqrt{\sum (S(f_k) - \bar{S})^2}$ |
| **$f_{54}$** | `spectral_entropy` | Menghitung entropi spektral berbasis transformasi Fourier. | $H_{\text{spec}} = -\sum p_k \log_2 p_k, \quad p_k = \frac{\vert S(f_k) \vert^2}{\sum \vert S(f_k) \vert^2}$ |
| **$f_{55}$** | `spectral_kurtosis` | Menghitung keruncingan sebaran bentuk spektrum frekuensi di sekitar rata-rata. | $K_{\text{spec}} = \frac{\sum (f_k - C_{\text{spec}})^4 S(f_k)}{\sigma_{\text{spec}}^4 \sum S(f_k)}$ |
| **$f_{56}$** | `spectral_positive_turning` | Menghitung jumlah titik balik positif dari magnitudo sinyal FFT. | $N_{TP+,\text{fft}} = \sum \mathbf{1}_{(\vert S_k \vert > \vert S_{k-1} \vert \land \vert S_k \vert > \vert S_{k+1} \vert)}$ |
| **$f_{57}$** | `spectral_roll_off` | Menghitung frekuensi di mana 95% energi spektral terkonsentrasi. | $\sum_{f=0}^{f_{\text{roll}}} \vert S(f) \vert^2 = 0.95 \sum \vert S(f) \vert^2$ |
| **$f_{58}$** | `spectral_roll_on` | Menghitung frekuensi di mana 5% awal energi spektral mulai terkonsentrasi. | $\sum_{f=0}^{f_{\text{roll-on}}} \vert S(f) \vert^2 = 0.05 \sum \vert S(f) \vert^2$ |
| **$f_{59}$** | `spectral_skewness` | Menghitung asimetri distribusi spektrum daya frekuensi. | $S_{k,\text{spec}} = \frac{\sum (f_k - C_{\text{spec}})^3 S(f_k)}{\sigma_{\text{spec}}^3 \sum S(f_k)}$ |
| **$f_{60}$** | `spectral_slope` | Menghitung gradien kemiringan garis regresi pada spektrum frekuensi. | $\text{Slope}_{\text{spec}} = \frac{K \sum f_k S(f_k) - \sum f_k \sum S(f_k)}{K \sum f_k^2 - (\sum f_k)^2}$ |
| **$f_{61}$** | `spectral_spread` | Menghitung simpangan sebaran spektrum di sekitar titik centroid spektral. | $\sigma_{\text{spec}}^2 = \frac{\sum (f_k - C_{\text{spec}})^2 S(f_k)}{\sum S(f_k)}$ |
| **$f_{62}$** | `spectral_variation` | Menghitung tingkat variasi perubahan pola spektrum sepanjang waktu. | $V_{\text{spec}} = 1 - \frac{\sum S_t(f) S_{t-1}(f)}{\sqrt{\sum S_t^2(f) \sum S_{t-1}^2(f)}}$ |
| **$f_{63}$** | `spectrogram_mean_coeff` | Menghitung rata-rata kerapatan spektral daya (PSD) dari spektrogram. | $\bar{P}(f) = \frac{1}{T}\sum_{t=1}^{T} \text{PSD}(f, t)$ |
| **$f_{64}$** | `wavelet_abs_mean` | Menghitung rata-rata mutlak koefisien *Continuous Wavelet Transform* (CWT). | $\text{CWT}_{\text{abs}} = \frac{1}{N}\sum \vert W(a, b) \vert$ |
| **$f_{65}$** | `wavelet_energy` | Menghitung total energi koefisien wavelet pada setiap skala *wavelet*. | $E_{\text{wavelet}}(a) = \sum_{b} \vert W(a, b) \vert^2$ |
| **$f_{66}$** | `wavelet_entropy` | Menghitung entropi spektral energi dari hasil dekode *wavelet*. | $H_{\text{wavelet}} = -\sum p_a \log_2 p_a$ |
| **$f_{67}$** | `wavelet_std` | Menghitung standar deviasi dari koefisien skala *wavelet* CWT. | $s_{\text{wavelet}}(a) = \text{std}(W(a, b))$ |
| **$f_{68}$** | `wavelet_var` | Menghitung variansi dari koefisien skala *wavelet* CWT. | $s^2_{\text{wavelet}}(a) = \text{var}(W(a, b))$ |

---

> [!NOTE]
> Penomoran urut kode $f_1$ hingga $f_{68}$ di atas memetakan seluruh modul fitur bawaan TSFEL secara presisi untuk memudahkan identifikasi kolom saat hasil matriks fitur diekstrak ke dalam format tabel CSV.
