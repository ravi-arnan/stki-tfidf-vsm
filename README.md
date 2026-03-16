# STKI TF-IDF vs VSM (Bahasa Indonesia)

Proyek ini adalah tugas **Sistem Temu Kembali Informasi (STKI)** untuk membandingkan dua metode perankingan dokumen:
- **TF-IDF (manual)**: skor dokumen dihitung dari penjumlahan bobot term query.
- **VSM (Vector Space Model)**: skor dokumen dihitung dengan **cosine similarity** antara vektor query dan vektor dokumen berbasis TF-IDF.

Notebook utama: `STKI_TFIDF_VSM.ipynb`

## Isi Proyek Singkat
- Korpus 5 dokumen pendek Bahasa Indonesia (topik berbeda)
- Preprocessing teks (case folding, cleaning regex, tokenizing, stopword removal, stemming)
- Perhitungan **TF, IDF, TF-IDF manual** (tanpa `sklearn.TfidfVectorizer`)
- Ranking dokumen dengan metode TF-IDF dan VSM
- Tabel perbandingan hasil ranking dan analisis singkat

---

## Setup di Lokal

### 1) Clone repository
```bash
git clone https://github.com/ravi-arnan/stki-tfidf-vsm.git
cd stki-tfidf-vsm
```

### 2) Install dependency
```bash
pip install --upgrade pip
pip install PySastrawi pandas numpy
```

### 3) Buka notebook di VS Code (extension Colab)
Buka file `STKI_TFIDF_VSM.ipynb` di VS Code, lalu jalankan cell dari atas ke bawah.

---

## Cara Menjalankan Notebook
1. Jalankan cell dari atas ke bawah secara berurutan.
2. Cell pertama akan install `PySastrawi`.
3. Perhatikan output tabel:
   - Tabel TF-IDF lengkap (term × dokumen)
   - Ranking metode TF-IDF
   - Ranking metode VSM
   - Tabel perbandingan kedua metode

---

## Struktur Repository
```text
stki-tfidf-vsm/
├── STKI_TFIDF_VSM.ipynb
└── README.md
```

---

## Catatan
- Proyek ini ditulis untuk kebutuhan pembelajaran STKI.
- Jika ingin memakai data sendiri, ubah variabel korpus pada section **Dokumen Korpus** di notebook.
