# Sistem Temu Kembali Informasi (STKI) — TF-IDF & Vector Space Model

Implementasi dan analisis enam metode fundamental **Sistem Temu Kembali Informasi (Information Retrieval)** untuk korpus **Bahasa Indonesia**, lengkap dengan teori, rumus, dan eksperimen pada jurnal nyata. Dikerjakan sebagai tugas UTS mata kuliah STKI.

## ✨ Cakupan Metode

| # | Metode | Inti | Notebook |
|---|--------|------|----------|
| 1 | **Boolean Retrieval** | Inverted index + operasi himpunan (AND/OR/NOT) | `STKI_Boolean_Inverted.ipynb` |
| 2 | **Jaccard Similarity** | Kemiripan berbasis himpunan term | `STKI_Jaccard.ipynb` |
| 3 | **TF-IDF** | Pembobotan term (frekuensi lokal × kelangkaan global) | `STKI_TFIDF_VSM_Final.ipynb` |
| 4 | **Vector Space Model (VSM)** | Representasi vektor + cosine similarity | `STKI_TFIDF_VSM_Final.ipynb` |
| 5 | **Extractive Summarization** | Pemilihan kalimat skor TF-IDF tertinggi | `STKI_TFIDF_Summarization.ipynb` |
| 6 | **MinHash & LSH** | Estimasi Jaccard probabilistik untuk near-duplicate detection | `STKI_MinHash.ipynb` |

## 🔧 Pipeline Preprocessing (5 Tahap)

Seluruh metode memakai pipeline yang konsisten untuk Bahasa Indonesia:

```
case folding → cleaning → tokenizing → stopword removal → stemming
```

- **Stopword removal** — NLTK (757 stopword ID) & Sastrawi (~126 entri)
- **Stemming** — Sastrawi, algoritma **Nazief & Adriani** (confix-stripping berbasis kamus)

## 📁 Struktur Proyek

```
stki-tfidf-vsm/
└── lsi-colab/
    ├── STKI_Boolean_Inverted.ipynb          # Bab 1 — Boolean Retrieval
    ├── STKI_Jaccard.ipynb                    # Bab 2 — Jaccard Similarity
    ├── STKI_TFIDF_VSM_Final.ipynb           # Bab 3 & 4 — TF-IDF + VSM
    ├── STKI_TFIDF_Summarization.ipynb        # Bab 5 — Extractive Summarization
    ├── STKI_TFIDF_Summarization_Penjelasan.ipynb
    ├── STKI_MinHash.ipynb                    # Bab 6 — MinHash & LSH
    ├── Jurnal1.pdf, Jurnal2.pdf, Jurnal3.pdf # Korpus uji (jurnal STKI)
    ├── *.png                                 # Gambar hasil eksperimen
    └── docs/
        ├── LAPORAN_UTS_STKI.md              # Laporan lengkap (teori + hasil)
        ├── JAWABAN_SOAL_STKI.md             # Jawaban soal analisis
        ├── laporan_minhash_1.3.md
        └── rumus.md                          # Rangkuman rumus MinHash
```

## 🚀 Cara Menjalankan

> **Catatan:** semua notebook dijalankan dari dalam direktori `lsi-colab/` karena merujuk file PDF/PNG dengan path relatif.

```bash
# 1. Buat & aktifkan virtual environment
python3 -m venv lsi-colab/.venv
source lsi-colab/.venv/bin/activate     # Windows: lsi-colab\.venv\Scripts\activate

# 2. Install dependency
pip install -r requirements.txt

# 3. Unduh data NLTK (stopword Bahasa Indonesia)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Jalankan Jupyter
cd lsi-colab
jupyter lab        # atau: jupyter notebook
```

## 📦 Dependency Utama

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.12 | Runtime |
| numpy | 2.4 | Komputasi vektor/matriks |
| pandas | 3.0 | Tabulasi hasil |
| scikit-learn | 1.8 | Utilitas TF-IDF/VSM |
| PySastrawi | 1.2 | Stopword & stemming Bahasa Indonesia |
| nltk | — | Stopword Bahasa Indonesia |
| pdfplumber | 0.11 | Ekstraksi teks PDF jurnal |
| matplotlib | 3.10 | Visualisasi akurasi & ranking |
| jupyterlab | 4.5 | Lingkungan notebook |

## 📊 Korpus Uji

Tiga jurnal bertema perpustakaan & temu kembali informasi:

1. **Jurnal1.pdf** — OPAC pada Aplikasi CIP, Tridinanti Palembang (Elsadantia, 2023)
2. **Jurnal2.pdf** — Pemanfaatan STKI via OPAC, Poltekkes Sorong (Nanlohy dkk., 2023)
3. **Jurnal3.pdf** — Strategi Shelving Koleksi, SMAN 2 Trenggalek (Fatoni & Handayani, 2024)

## 📖 Dokumentasi

Penjelasan teori lengkap, rumus, dan analisis hasil tiap metode ada di **[`lsi-colab/docs/LAPORAN_UTS_STKI.md`](lsi-colab/docs/LAPORAN_UTS_STKI.md)**.

## 👥 Tim

- Ravi Arnan Irianto (2305551076)
- Richad Krishnadana Primantara (2305551151)
- Putu Satria Arya Putra (2305551122)

## 📚 Referensi

- Manning, Raghavan & Schütze (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Rajaraman & Ullman (2011). *Mining of Massive Datasets*, Bab 3 (Finding Similar Items).
- Nazief & Adriani (1996). *Confix-Stripping: Approach to Stemming Algorithm for Bahasa Indonesia*. Universitas Indonesia.
- Septiani & Isabela (2022). *Analisis TF-IDF dalam Temu Kembali Informasi pada Dokumen Teks*. SINTESIA Vol. 01 No. 2.
