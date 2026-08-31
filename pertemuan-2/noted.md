# Catatan Data Understanding & Tipe Atribut Data

## 1. Pemahaman Data (_Data Understanding_)

Dalam analisis data dan _data mining_, pemahaman terhadap bentuk dan karakteristik data merupakan langkah awal yang krusial:

- **Tipe Data**: Pengelompokan jenis informasi (misalnya: numerik, teks, tanggal, data spasial).
- **Record Data**: Jumlah entitas atau baris objek individual di dalam berkas/dataset.
- **Data Terstruktur (_Structured Data_)**: Data yang tersusun rapi dalam bentuk tabel (baris dan kolom) berformat baku (contoh: berkas CSV, database SQL).
- **Data Tidak Terstruktur (_Unstructured Data_)**: Data yang tidak memiliki skema tabel baku (contoh: dokumen teks bebas, gambar, rekaman suara, video).

---

## 2. Tipe Atribut Data (_Data Attributes_)

Atribut (variabel/fitur) merepresentasikan sifat atau karakteristik dari suatu objek data. Atribut dibedakan berdasarkan skala pengukuran dan kriteria nilainya:

### A. Berdasarkan Skala Pengukuran

1. **Nominal**:
   - Kategori atau label nama tanpa memiliki urutan/tingkatan tertentu.
   - _Contoh_: Nama Polutan (`CH4`, `CO`, `NO2`, `SO2`), Nama Kota (`Gresik`, `Surabaya`).

2. **Biner (_Binary_)**:
   - Atribut nominal khusus yang hanya memiliki **2 pilihan nilai** (misalnya: $0 / 1$ atau _True / False_).
   - _Contoh_: Status Outlier (_Outlier_ / _Normal_), Status Validasi Data (_Valid_ / _Missing_).

3. **Ordinal**:
   - Nilai kategori yang memiliki **urutan atau tingkatan berjenjang** yang jelas, namun selisih kuantitatif antar tingkatannya tidak dapat dihitung secara pasti.
   - _Contoh_: Kategori Tingkat Polusi Udara (_Rendah_ $\rightarrow$ _Sedang_ $\rightarrow$ _Tinggi_ $\rightarrow$ _Sangat Tinggi_).

4. **Numerik (_Numeric_)**:
   - Nilai kuantitatif berupa angka riil yang dapat diukur dan dihitung secara matematis.
   - _Contoh_: Konsentrasi CH4 ($1892.85\text{ ppb}$), Konsentrasi CO ($0.0291\text{ mol}/m^2$).

---

### B. Berdasarkan Diskrit vs Kontinu (_Discrete vs Continuous_)

1. **Atribut Diskrit (_Discrete Attributes_)**:
   - Memiliki jumlah nilai terbatas (_finite_) atau dapat dihitung (_countable_). Biasanya berupa bilangan bulat.
   - _Contoh_: Jumlah total baris observasi ($366$), Jumlah _missing values_ ($157$).

2. **Atribut Kontinu (_Continuous Attributes_)**:
   - Memiliki jumlah nilai tak terbatas dalam suatu rentang interval, biasanya berupa bilangan riil/desimal.
   - _Contoh_: Nilai hasil pengukuran sensor satelit ($0.00007321\text{ mol}/m^2$).

nanti mungkin ada soal mengenai positive skewed dan negative skewed menghitung mean modus dan median nya

variance : 1 var
co variance : lebih dari 1 sigma
standart devaisai sigma 1
korelasi :

praktekan menghitung korelasi, nilai korelasi dari 1 sampai -1

jika data mendekati 0 itu tidak berpengaruh pada korelasi

sialhkan membuaka aiven
untuk mengetahui statistik

tugas pindah data ke cloud
tarik data ke knime
dan tampilkan statistik viw nya

