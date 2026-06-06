# 1.3 Alur Implementasi

Implementasi MinHash dijalankan dalam Jupyter Notebook `STKI_MinHash.ipynb` dengan alur kerja yang dibagi menjadi dua tahap berurutan, yaitu persiapan korpus dan pembentukan shingle (Subbab 1.3.1) serta konstruksi MinHash signature dan estimasi Jaccard similarity (Subbab 1.3.2). Penjabaran rumus matematis yang dipakai dibahas terpisah pada Subbab 1.4 — Rumus pada MinHash, sehingga subbab ini hanya berfokus pada urutan proses, input-output tiap langkah, dan parameter konfigurasi yang dipilih.

Korpus penelitian terdiri atas tiga jurnal bertema perpustakaan dan temu kembali informasi. Dokumen pertama (D1) adalah "Perkembangan dan Peran OPAC pada Aplikasi CIP — Tridinanti Palembang" karya Elsadantia (2023) dari berkas `Jurnal1.pdf`. Dokumen kedua (D2) adalah "Pemanfaatan STKI menggunakan OPAC — Poltekkes Sorong" karya Nanlohy dkk. (2023) dari `Jurnal2.pdf`. Dokumen ketiga (D3) adalah "Strategi Pustakawan dalam Shelving Koleksi — SMAN 2 Trenggalek" karya Fatoni dan Handayani (2024) dari `Jurnal3.pdf`. Sesuai instruksi dosen, setiap jurnal utuh diperlakukan sebagai satu dokumen — bukan dipecah per paragraf — karena tujuan akhirnya adalah mengukur kemiripan antar-jurnal.

## 1.3.1 Persiapan Korpus dan Pembentukan Shingle

Tahap ini menerima tiga file PDF mentah dan menghasilkan dict `shingle_set = {D1, D2, D3}` yang siap-bandingkan. Lima sub-langkah dijalankan secara berurutan, mulai dari ekstraksi teks PDF, pemotongan bab dengan RegEx, pembentukan korpus dokumen, preprocessing lima tahap, hingga pembentukan shingle.

### 1.3.1.1 Ekstraksi Teks PDF

Tiga berkas jurnal dibaca menggunakan library `pdfplumber`. Karena library tersebut memecah teks per baris visual, struktur paragraf direkonstruksi melalui posisi indentasi horizontal `x0`, di mana baris dengan `x0 > 125` dianggap sebagai awal paragraf baru. Filter generik diterapkan untuk membuang baris yang panjangnya kurang dari sepuluh karakter dan baris yang isinya hanya berupa angka, yang umumnya merupakan nomor halaman atau footer numerik. Output langkah ini adalah dict berisi raw text per dokumen dengan separator `\n\n` antar paragraf.

### 1.3.1.2 Pemotongan Bab dengan RegEx

Tiap jurnal memakai konvensi penutup bab yang berbeda, sehingga RegEx mencoba empat pola secara berurutan dan memakai pola pertama yang berhasil match. Empat pola tersebut adalah `PENDAHULUAN…KESIMPULAN`, `PENDAHULUAN…PENUTUP`, `PENDAHULUAN…SIMPULAN`, dan `PENDAHULUAN…DAFTAR PUSTAKA`. Bila keempat pola gagal, sistem memakai mekanisme *fallback* dengan mengambil seluruh raw text apa adanya. Dengan strategi ini, bagian Abstrak yang berada di atas PENDAHULUAN dan Daftar Pustaka yang berada setelah penanda penutup idealnya terbuang otomatis.

### 1.3.1.3 Korpus Dokumen

Hasil potongan RegEx dari masing-masing jurnal langsung menjadi dokumen utuh D1, D2, dan D3, tanpa pemecahan paragraf lebih lanjut. Pendekatan ini sengaja dipilih agar konsisten dengan paradigma perbandingan antar-jurnal, bukan antar-paragraf dalam satu jurnal.

### 1.3.1.4 Preprocessing Lima Tahap

Pipeline standar Bahasa Indonesia diterapkan secara identik pada tiap dokumen, terdiri dari lima tahap berurutan. Tahap pertama adalah *case folding* melalui `text.lower()` yang menyamakan huruf besar dan kecil. Tahap kedua adalah *cleaning* menggunakan ekspresi reguler `re.sub(r'[^a-z\s]', ' ', text)` untuk membuang angka dan tanda baca. Tahap ketiga adalah *tokenizing* dengan `text.split()` berbasis whitespace. Tahap keempat adalah *stopword removal* memakai PySastrawi `StopWordRemoverFactory` yang membuang kata umum tanpa makna pembeda. Tahap terakhir adalah *stemming* dengan PySastrawi `StemmerFactory` untuk mereduksi kata berimbuhan menjadi kata dasar. Konsistensi pipeline ini wajib dijaga karena shingle antar-dokumen harus dibandingkan pada ruang token yang sama.

### 1.3.1.5 Pembentukan Shingle

Token hasil preprocessing dikonversi menjadi himpunan *k-shingle* dengan parameter `K_SHINGLE = 2` atau bigram. Setiap shingle direpresentasikan sebagai *tuple* yang bersifat *immutable* agar dapat dimasukkan ke struktur `set` Python. Output akhir Subbab 1.3.1 adalah `shingle_set = {D1: {…}, D2: {…}, D3: {…}}` yang menjadi input langsung tahap selanjutnya.

## 1.3.2 Konstruksi MinHash Signature dan Estimasi Jaccard Similarity

Tahap ini menerima `shingle_set` dari Subbab 1.3.1 dan menghasilkan matriks estimasi Jaccard berukuran $3 \times 3$ beserta analisis akurasinya. Lima sub-langkah dijalankan berurutan, dimulai dari penghitungan ground truth, pembangkitan parameter fungsi hash, pembangunan signature matrix, estimasi Jaccard via signature, hingga validasi empiris terhadap jumlah fungsi hash.

### 1.3.2.1 Perhitungan Jaccard Eksak sebagai Ground Truth

Matriks Jaccard $3 \times 3$ antar D1, D2, dan D3 dihitung langsung dari himpunan shingle berdasarkan Rumus 1 yang dipaparkan pada Subbab 1.4. Nilai eksak ini berperan sebagai referensi untuk mengevaluasi seberapa akurat estimasi MinHash yang dihitung pada langkah-langkah berikutnya.

### 1.3.2.2 Pembangkitan Parameter Fungsi Hash

Sistem membangkitkan sebanyak `K = N_HASH` pasang bilangan acak $(a, b)$ dengan syarat $a \neq 0$, lalu memasangkannya dengan bilangan prima besar `p = 4_294_967_311` (kira-kira $2^{32}$) sebagai modulus. Pasangan-pasangan ini menghasilkan $K$ fungsi hash linear universal yang independen sesuai Rumus 2 pada Subbab 1.4. Sebelum proses hashing, tiap shingle terlebih dahulu dipetakan ke indeks integer melalui *id mapping* yang konsisten antar-dokumen agar dapat di-hash secara seragam.

### 1.3.2.3 Pembangunan Signature Matrix

Untuk tiap dokumen $S$ dan tiap fungsi hash $h_k$, sistem mengambil nilai minimum dari $h_k$ pada seluruh elemen $S$ sesuai Rumus 3 pada Subbab 1.4. Hasilnya disusun dalam *signature matrix* berukuran $K \times N$, dengan baris merepresentasikan fungsi hash dan kolom merepresentasikan dokumen. Signature inilah sidik jari ringkas yang menggantikan himpunan shingle berukuran besar, sebuah kompresi yang menjadi nilai jual utama MinHash.

### 1.3.2.4 Estimasi Jaccard via Signature

Untuk tiap pasangan dokumen, sistem menghitung proporsi posisi signature yang cocok berdasarkan Rumus 4 pada Subbab 1.4. Output langkah ini adalah matriks estimasi Jaccard $3 \times 3$ yang langsung dibandingkan dengan ground truth dari Subbab 1.3.2.1. Selisih absolut antar keduanya menjadi metrik kualitas signature yang diperoleh.

### 1.3.2.5 Analisis Akurasi terhadap Jumlah Fungsi Hash

Sebagai validasi empiris atas Rumus 5 mengenai varians estimator yang dipaparkan pada Subbab 1.4, langkah Subbab 1.3.2.2 hingga 1.3.2.4 diulang untuk berbagai nilai $K$, misalnya 16, 64, 256, dan 1024. Rata-rata error absolut estimasi terhadap ground truth kemudian diplot sebagai fungsi $K$. Hasilnya konsisten dengan teori, yakni error berkurang berbanding terbalik dengan $\sqrt{K}$ dan menampilkan pola *diminishing returns* yang khas estimator MinHash.

Output akhir Subbab 1.3 secara keseluruhan terdiri atas matriks Jaccard eksak $3 \times 3$, matriks Jaccard estimasi $3 \times 3$, selisih absolut antara keduanya, serta kurva konvergensi akurasi estimasi terhadap jumlah fungsi hash $K$.
