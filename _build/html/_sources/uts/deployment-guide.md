# Panduan Deployment Aplikasi Streamlit: Klasifikasi LULC Jawa Timur

Dokumentasi ini menjelaskan langkah-langkah komprehensif implementasi dan deployment aplikasi web interaktif **Klasifikasi Spasial Tutupan Lahan (LULC) Jawa Timur** menggunakan platform **Streamlit Community Cloud** yang terhubung langsung dengan repositori **GitHub**.

---

## 🌐 Tautan Aplikasi & Repositori

- **URL Aplikasi Live (Streamlit Cloud):** [https://lulc-jatim.streamlit.app/](https://lulc-jatim.streamlit.app/)
- **Repositori GitHub:** [https://github.com/muhammadainulfuady/PSD](https://github.com/muhammadainulfuady/PSD)
- **Path Berkas Aplikasi Utama:** [`uts/deploy/app.py`](https://github.com/muhammadainulfuady/PSD/blob/main/uts/deploy/app.py)

---

## 🏗️ Arsitektur Sistem & *Single Source of Truth (SSOT)*

Aplikasi dirancang menggunakan paradigma **Single Source of Truth (SSOT)** berbasis berkas CSV:
1. **Model Machine Learning & Analisis Python:** Dijalankan di notebook [`uts/code/landuseCover.ipynb`](file:///d:/SEMESTER%205%20TIF%202026/PSD/mynewbook/uts/code/landuseCover.ipynb).
2. **Ekspor Data Analitik:** Seluruh metrik evaluasi diekspor dalam format tabular ke direktori `uts/deploy/`:
   - `metrik_evaluasi_knn.csv`: Akurasi, Precision, Recall, F1-Score per kelas lahan.
   - `confusion_matrix.csv`: Distribusi prediksi vs aktual pada 75 poligon testing.
   - `ringkasan_model_knn.csv`: Ringkasan konfigurasi model ($k=5$, bobot jarak, metrik Euclidean).
   - `kepentingan_fitur.csv`: Urutan kontribusi fitur (*Permutation Feature Importance*).
   - `kurva_validasi_k.csv`: Evaluasi akurasi cross-validation nilai $k \in [1, 15]$.
   - `prediksi_data_uji.csv`: Koordinat, label aktual, label prediksi, dan status akurasi tiap poligon.
3. **Aplikasi Streamlit (`app.py`):** Tidak menggunakan gambar statis, melainkan membaca data tabular CSV secara dinamis dan merender grafik secara *real-time* menggunakan Seaborn/Matplotlib dengan palet warna *Warm Orange*.
4. **Peta Interaktif Spasial:** Menyematkan berkas Folium interaktif (`map_lulc_predicted.html` dan `map_lulc.html`) yang menyajikan sebaran 250 poligon sampel di seluruh Jawa Timur.

---

## 📋 Prasyarat Berkas Deployment

Agar aplikasi dapat berjalan mulus pada platform *cloud*, direktori `uts/deploy/` memiliki berkas-berkas berikut:

| Nama Berkas | Fungsi / Deskripsi |
| :--- | :--- |
| `app.py` | Kode antarmuka pengguna Streamlit, visualisasi dinamis, dan kontrol interaktif. |
| `requirements.txt` | Daftar pustaka Python yang wajib diinstal oleh server cloud. |
| `*.csv` (6 Berkas) | Data hasil analisis model k-NN sebagai sumber visualisasi dinamis. |
| `map_lulc_predicted.html` | Peta spasial interaktif evaluasi poligon training dan testing. |
| `map_lulc.html` | Peta spasial dasar sebaran centroid poligon 5 kelas lahan. |

Isi pustaka pada `uts/deploy/requirements.txt`:
```text
streamlit>=1.30.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
```

---

## 🚀 Panduan Langkah-demi-Langkah Deployment

### Langkah 1: Sinkronisasi dan Unggah ke GitHub

Pastikan seluruh berkas kode, dependensi, dan dataset CSV telah di-commit serta di-*push* ke repositori GitHub:

```bash
git add uts/deploy/
git commit -m "feat: finalize streamlit app and deploy assets"
git push origin main
```

### Langkah 2: Akses Streamlit Community Cloud

1. Buka peramban (*browser*) dan kunjungi situs resmi: [share.streamlit.io](https://share.streamlit.io/).
2. Masuk (*Sign in*) menggunakan akun **GitHub** Anda (`muhammadainulfuady`).

### Langkah 3: Membuat Deployment Aplikasi Baru

1. Pada halaman utama Streamlit Cloud, klik tombol **"Create app"** atau **"Deploy a public app from GitHub"**.
2. Masukkan parameter konfigurasi berikut:
   - **Repository:** `muhammadainulfuady/PSD`
   - **Branch:** `main`
   - **Main file path:** `uts/deploy/app.py`
   - **App URL:** `lulc-jatim` (sehingga domain menjadi `lulc-jatim.streamlit.app`)
3. Klik tombol **"Deploy!"**.

### Langkah 4: Proses Build & Verifikasi Cloud

1. Server Streamlit Cloud akan secara otomatis membaca berkas `requirements.txt` di dalam direktori `uts/deploy/`, memasang semua pustaka yang diperlukan (`streamlit`, `pandas`, `matplotlib`, `seaborn`, `pillow`).
2. Server menjalankan `app.py` dan memvalidasi koneksi ke seluruh berkas CSV.
3. Setelah beberapa detik, aplikasi aktif (*Running*) dan dapat diakses secara publik di seluruh dunia melalui tautan resmi:
   👉 **`https://lulc-jatim.streamlit.app/`**

---

## 🔄 Siklus Pembaharuan Otomatis (CI/CD Continuous Deployment)

Sistem deployment Streamlit Community Cloud terhubung langsung dengan *branch* `main` GitHub:
- Apabila Anda menjalankan eksperimen baru di Jupyter Notebook dan memperbarui nilai pada berkas CSV di folder `uts/deploy/`, cukup lakukan `git commit` dan `git push origin main`.
- Streamlit Cloud akan secara otomatis mendeteksi perubahan tersebut dan me-refresh tampilan dashboard tanpa perlu konfigurasi ulang.
