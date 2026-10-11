# Business Understanding

## 1. Latar Belakang
Provinsi Jawa Timur memiliki lanskap geografis yang beragam, meliputi area pesisir pantai, pertanian, perkotaan/pemukiman, hingga pegunungan dan hutan. Pemantauan perubahan tutupan lahan (*Land Use and Land Cover* / LULC) secara berkala penting untuk perencanaan tata ruang, konservasi ekosistem pesisir (hutan mangrove), dan pemantauan ketahanan pangan (sawah).

## 2. Definisi LULC
- **Land Cover (Penutupan Lahan):** Kenampakan fisik biofisik permukaan bumi yang terekam sensor satelit (contoh: vegetasi, badan air, bangunan).
- **Land Use (Penggunaan Lahan):** Pemanfaatan permukaan bumi oleh aktivitas manusia (contoh: pertanian/sawah, permukiman, kawasan industri).

## 3. Tujuan Proyek
1. **Pemetaan Otomatis:** Memetakan dan mengklasifikasikan tutupan lahan di Jawa Timur menggunakan citra satelit optik Sentinel-2.
2. **Diferensiasi Hutan Mangrove:** Membedakan vegetasi Hutan Mangrove dari Hutan Non-Mangrove dan Sawah berdasarkan respon spektral band SWIR (B11), NIR (B08), dan Red (B04).
3. **Verifikasi Visual Overlay:** Menampilkan hasil klasifikasi dalam bentuk *layer overlay* interaktif di atas citra satelit dasar (*basemap*).
4. **Deployment Aplikasi Web:** Membangun aplikasi web interaktif menggunakan Streamlit untuk menampilkan hasil peta, metrik evaluasi model, dan statistik luas area.

## 4. Kelas Tutupan Lahan
Klasifikasi dilakukan pada 5 kelas tutupan lahan utama dengan total 250 sampel poligon (50 sampel per kelas):

| No | Kelas Tutupan Lahan | Deskripsi & Respon Spektral |
| :-: | :--- | :--- |
| 1 | **Air** | Laut, sungai, waduk, dan tambak. Penyerapan kuat pada NIR dan SWIR (NDWI tinggi). |
| 2 | **Hutan Mangrove** | Vegetasi pesisir dan muara sungai. NIR tinggi dan SWIR (B11) peka kelembapan. |
| 3 | **Hutan Non-Mangrove** | Vegetasi daratan/pepohonan tinggi. Reflektansi NIR (B08) sangat tinggi. |
| 4 | **Pemukiman** | Perumahan, gedung, dan infrastruktur. Respon SWIR (B11) dan NDBI tinggi. |
| 5 | **Sawah** | Lahan vegetasi musiman dan tanah pertanian. NDVI bervariasi sesuai fase tanam. |

## 5. Kriteria Keberhasilan Proyek
- **Akurasi Model:** Mencapai *Overall Accuracy* (OA) dan *Kappa Index* yang memadai.
- **Kualitas Visualisasi:** Overlay transparan yang dapat di-toggle interaktif di atas basemap satelit.
- **Analisis Luas Area:** Menyajikan estimasi persentase dan luas area ($km^2$) untuk setiap kelas lahan.

