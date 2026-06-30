# OVERVIEW — Asisten Hukum Pajak (RAG: IndoBERT + GPT)

Dokumen ini menjelaskan **keseluruhan** subsistem RAG (metode ke-7 proyek STKI) secara
end-to-end: arsitektur, tech stack, alur data, sampai siklus deploy ke **Hugging Face
Spaces**. Untuk panduan ringkas pakai [`README.md`](README.md); untuk laporan akademik
bertahap pakai [`laporan_rag_1.0.md`](laporan_rag_1.0.md).

- **Demo live (cloud):** <https://huggingface.co/spaces/raviarnan/asisten-hukum-pajak-rag>
- **Repo proyek:** bagian dari repo STKI (metode 1–6 di `lsi-colab/`)

---

## 1. Apa ini & kenapa beda

Metode 1–6 STKI (Boolean, Jaccard, TF-IDF, VSM, Summarization, MinHash/LSH) hanya
**me-ranking** dokumen terhadap query. Subsistem RAG menambah satu LLM di ujung pipeline
sehingga sistem **menjawab pertanyaan** dalam Bahasa Indonesia, lengkap dengan **sitasi
dokumen sumber `[D#]`** — bukan sekadar daftar dokumen relevan.

Korpus juga berbeda: bukan 3 jurnal ilmu perpustakaan (metode 1–6), melainkan **10 PDF
hukum pajak** (PBB + Pajak Kendaraan Bermotor) di `corpus_pajak/`.

---

## 2. Arsitektur end-to-end

```
                         ┌──────────────────── INDEXING (sekali, lalu di-cache) ───────────────────┐
                         │                                                                          │
 10 PDF hukum pajak ─► ekstraksi per-halaman ─► chunking (≈4 kalimat / maks 160 kata)               │
 (corpus_pajak/)         (pdfplumber)             + filter noise (buang tabel tarif, WORDY_MIN)      │
                                                        │                                            │
                                          700-an passage berlabel                                    │
                                          {doc_id, sumber, halaman, teks}                            │
                                                        │                                            │
                                          embed IndoBERT (768-d, normalized)                         │
                                                        │                                            │
                                          simpan ke cache: hf_cache/rag_index_v2.npz                 │
                         └──────────────────────────────┼───────────────────────────────────────────┘
                                                        │
 ╔══════════════════════════════════ QUERY (tiap pertanyaan) ═══════════════════════════════════════╗
 ║  query pengguna                                                                                    ║
 ║       │                                                                                            ║
 ║       ├─────────────────────────────┬──────────────────────────────────                           ║
 ║       ▼                             ▼                                                              ║
 ║  BM25 (leksikal)            IndoBERT dense (cosine)                                                ║
 ║  kata/singkatan/angka persis  makna semantik                                                       ║
 ║  "PPnBM", "2025"             "pajak mobil mewah"                                                   ║
 ║       └──────────► Reciprocal Rank Fusion (RRF, K=60) ◄──────────┘                                 ║
 ║                              │                                                                     ║
 ║                   top-5 passage paling relevan                                                     ║
 ║                              ▼                                                                     ║
 ║          GPT gpt-4o-mini (via OpenRouter) + system prompt grounding                                ║
 ║                              ▼                                                                     ║
 ║       Jawaban Bahasa Indonesia + sitasi [D#] + panel chunk konteks (Gradio)                        ║
 ╚════════════════════════════════════════════════════════════════════════════════════════════════╝
```

**Kenapa retrieval hybrid?** Dense (IndoBERT) saja gagal pada query yang bergantung token
persis — mis. "PPnBM" tidak punya tetangga semantik dekat di IndoBERT umum. BM25 menangkap
kecocokan leksikal itu (termasuk label sumber seperti tahun & jenis dokumen, untuk
membedakan D6/D7/D8 yang nyaris identik tapi beda tahun). RRF menggabungkan kedua ranking
tanpa perlu menyetel bobot skala skor yang berbeda.

---

## 3. Tech stack

| Lapisan | Komponen | Peran |
|---------|----------|-------|
| Bahasa & runtime | **Python 3.12**, venv + pip | Semua kode subsistem |
| Ekstraksi PDF | **pdfplumber** | Ekstraksi teks per-halaman dari 10 PDF |
| Retrieval leksikal | **rank-bm25** (BM25Okapi) | Cocokkan kata/singkatan/angka persis |
| Retrieval dense | **sentence-transformers** + **torch** (CPU) — **IndoBERT** `firqaaa/indo-sentence-bert-base` (768-d) | Cocokkan makna semantik |
| Fusi ranking | RRF (K=60) — **numpy** | Gabung BM25 + dense tanpa tuning bobot |
| Generasi jawaban | **GPT** `openai/gpt-4o-mini` via **OpenRouter** (SDK `openai`) | Susun jawaban + sitasi `[D#]` |
| Konfigurasi rahasia | **python-dotenv** (`.env` → `OPENROUTER_API_KEY`) | Muat kunci API; tidak di-hardcode |
| Frontend | **Gradio** (`gr.ChatInterface`) | Chat + panel chunk konteks |
| Cache | `numpy.savez` (`rag_index_v2.npz`) + `HF_HOME` model cache | Hindari re-embed & re-download |
| Deploy | **Hugging Face Spaces** (SDK Gradio 6.x) | Demo cloud publik |

Versi pasti dependency ada di `requirements.txt` (root repo, untuk lokal) dan
`hf-space/requirements.txt` (untuk Space). `torch` di-set CPU-only; thread BLAS dibatasi 4
(`OMP/MKL/OPENBLAS_NUM_THREADS`) agar mesin tetap responsif.

---

## 4. Struktur file

| Berkas | Peran |
|--------|-------|
| `rag_pipeline.py` | **Inti**: `build_korpus()`, `embed()`, `load_index()`, `retrieve()` (hybrid), `jawab()` |
| `app.py` | Frontend chat Gradio (lokal) + panel "chunk yang dipakai sebagai konteks" |
| `STKI_RAG_LLM.ipynb` | Notebook penjelasan/laporan (eksperimen bertahap) |
| `laporan_rag_1.0.md` | Laporan akademik alur implementasi + justifikasi parameter |
| `README.md` | Panduan ringkas subsistem |
| `OVERVIEW.md` | Dokumen ini — gambaran menyeluruh end-to-end |
| `corpus_pajak/` | 10 PDF korpus (D1–D10) |
| `hf_cache/` | Cache embedding (`rag_index_v2.npz`) + model IndoBERT — **gitignored** |
| `.env` | `OPENROUTER_API_KEY=...` — **gitignored** |
| `hf-space/` | Salinan staging yang di-upload ke Hugging Face Space |

### Korpus (D1–D10)

Sepuluh PDF hukum pajak resmi (tema PBB + Pajak Kendaraan Bermotor). Semua sumber
adalah dokumen publik dari portal hukum resmi pemerintah (JDIH Kemenkeu, peraturan.bpk.go.id)
dan repositori Universitas Terbuka.

| ID | Sumber | Sumber resmi (terverifikasi) |
|----|--------|------------------------------|
| D1 | UU 12/1985 — PBB (file `85uu012.pdf`) | [bphn.go.id — 85uu012.pdf](https://bphn.go.id/data/documents/85uu012.pdf) |
| D2 | UU 12/1985 — PBB (file lain, naskah UU yang sama) | [jdih.kemenkeu.go.id — UU 12/1985](https://jdih.kemenkeu.go.id/fulltext/1985/12TAHUN~1985UU.HTM) |
| D3 | PMK 85/2024 — Penilaian NJOP PBB-P2 | [jdih.kemenkeu.go.id — PMK 85/2024](https://jdih.kemenkeu.go.id/dok/pmk-85-tahun-2024/summary) |
| D4 | PMK 234/PMK.03/2022 — klasifikasi & penetapan NJOP PBB (perubahan PMK 186/2019) | [jdih.kemenkeu.go.id — PMK 234/2022](https://jdih.kemenkeu.go.id/dok/234-pmk-03-2022/summary) |
| D5 | Modul 1 PBB (PAJA3233) — Universitas Terbuka | [pustaka.ut.ac.id — PAJA3233 M1](https://pustaka.ut.ac.id/lib/wp-content/uploads/pdfmk/PAJA323304-M1.pdf) |
| D6 | Permendagri 7/2025 — dasar pengenaan PKB, BBN-KB & PAB 2025 | [peraturan.bpk.go.id — Permendagri 7/2025](https://peraturan.bpk.go.id/Details/321612/permendagri-no-7-tahun-2025) |
| D7 | Permendagri 8/2024 — dasar pengenaan PKB, BBN-KB & PAB 2024 | [peraturan.bpk.go.id — Permendagri 8/2024](https://peraturan.bpk.go.id/Details/300076/permendagri-no-8-tahun-2024) |
| D8 | Permendagri 6/2023 — dasar pengenaan PKB, BBN-KB & PAB 2023 | [peraturan.bpk.go.id — Permendagri 6/2023](https://peraturan.bpk.go.id/Details/252380/permendagri-no-6-tahun-2023) |
| D9 | PMK 8/2024 — PPN DTP kendaraan listrik (EV) TA 2024 | [jdih.kemenkeu.go.id — PMK 8/2024](https://jdih.kemenkeu.go.id/dok/pmk-8-tahun-2024) |
| D10 | PMK 5/PMK.010/2022 — PPnBM DTP kendaraan bermotor TA 2022 | [jdih.kemenkeu.go.id — PMK 5/2022](https://jdih.kemenkeu.go.id/api/download/d4c58cfc-2eda-491a-a46f-a0aa52c67877/5~PMK.010~2022Per.pdf) |

> D1 & D2 adalah **dua file PDF berbeda** (ukuran/hash beda) yang memuat **naskah UU
> 12/1985 yang sama** — sistem memperlakukannya sebagai dua dokumen terpisah. Nama file
> D1 (`85uu012.pdf`) cocok dengan pola BPHN; asal unduh persis D2 tak bisa dipastikan dari
> namanya, jadi ditautkan ke naskah resmi UU yang sama di JDIH Kemenkeu.
>
> Judul resmi D6–D8 mencakup PKB, BBN-KB **dan Pajak Alat Berat (PAB)** — cakupan
> sedikit lebih luas dari label ringkasnya. UU/PMK/Permendagri di atas adalah dokumen
> hukum publik; Modul UT (D5) mencantumkan sumbernya pada portal resmi Universitas Terbuka.

---

## 5. Menjalankan lokal

```bash
# Dari root repo: pasang dependency
source lsi-colab/.venv/bin/activate          # atau venv lain dengan requirements.txt terpasang

# Kunci OpenRouter (tidak di-commit)
echo "OPENROUTER_API_KEY=sk-or-v1-..." > rag/.env

cd rag
python app.py                                # buka http://127.0.0.1:7860
```

Saat pertama dijalankan, model IndoBERT (~500 MB) diunduh ke `rag/hf_cache/` dan index
embedding korpus dibangun lalu di-cache (`rag_index_v2.npz`, ~2–4 menit di CPU). Run
berikutnya langsung memakai cache. `app.py` melakukan *warm-up* encoder saat startup agar
pesan pertama tidak lambat.

Uji cepat dari terminal tanpa frontend:

```bash
python rag_pipeline.py "Kendaraan bermotor apa saja yang dikenai PPnBM?"
```

---

## 6. Hugging Face Spaces (cloud)

### 6.1 Bagaimana Space dibentuk

`hf-space/` adalah salinan staging yang di-upload ke Space. Isinya: `app.py` (varian
cloud), `rag_pipeline.py` (identik dengan versi lokal), `requirements.txt`, `README.md`
(berisi metadata Space), dan `corpus_pajak/` (10 PDF). `hf-space/.gitignore` mengecualikan
`.env` dan `hf_cache/`.

`hf-space/README.md` diawali blok *front-matter* YAML yang dibaca Hugging Face untuk
mengonfigurasi Space:

```yaml
---
title: Asisten Hukum Pajak RAG
emoji: 📑
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
---
```

### 6.2 Perbedaan `app.py` lokal vs cloud

| Aspek | Lokal (`app.py`) | Cloud (`hf-space/app.py`) |
|-------|------------------|---------------------------|
| Kunci API | dari `.env` (python-dotenv) | dari **Space secret** `OPENROUTER_API_KEY` |
| Bind server | `demo.launch()` (127.0.0.1:7860) | `server_name="0.0.0.0"`, `server_port=$PORT` |
| Berkas `.env` | ada (gitignored) | tidak ada — secret dari dashboard |

`rag_pipeline.py` **identik** di kedua sisi — tetap membaca `OPENROUTER_API_KEY` dari
environment, dan `load_dotenv()` hanya no-op kalau `.env` tidak ada (di Space env sudah
diisi secret).

### 6.3 Cold start

Cache (`hf_cache/`) di-*gitignore* di Space, jadi **tidak** ikut di-upload. Saat Space
cold start: model IndoBERT diunduh + index korpus dibangun (~2–4 menit di CPU), lalu
di-cache selama Space aktif. Setelah Space idle dan dimatikan HF, cold start berikutnya
membangun ulang.

### 6.4 Secret

Set di Space: **Settings → Variables and secrets** → tambahkan
`OPENROUTER_API_KEY = sk-or-v1-...`. **Tidak pernah** di-hardcode atau di-commit.

### 6.5 Menyinkronkan perubahan ke Space

> **Penting:** `hf-space/rag_pipeline.py` adalah salinan dari `rag/rag_pipeline.py`.
> Saat mengubah logika pipeline, perbarui **kedua** berkas lalu upload ulang ke Space.

Alur update tipikal:

1. Ubah `rag/rag_pipeline.py` (dan/atau `rag/app.py`).
2. Salin perubahan pipeline ke `rag/hf-space/rag_pipeline.py` (jaga tetap identik).
   Untuk `app.py`, terapkan perubahan ke varian cloud (pertahankan beda secret + bind di 6.2).
3. Upload isi `hf-space/` ke Space (via `git push` ke remote Space, atau UI upload HF).
4. Space rebuild otomatis; cek log untuk memastikan cold start sukses.

---

## 7. Konfigurasi (konstanta di `rag_pipeline.py`)

| Konstanta | Nilai | Arti |
|-----------|-------|------|
| `MODEL_EMBED` | `firqaaa/indo-sentence-bert-base` | Encoder IndoBERT (Sentence-BERT) |
| `MODEL_LLM` | `openai/gpt-4o-mini` | Generator jawaban (via OpenRouter) |
| `TOP_K` | 5 | Jumlah passage konteks |
| `KALIMAT_PER_CHUNK` / `MAX_KATA_CHUNK` | 4 / 160 | Ukuran chunk |
| `WORDY_MIN` | 0.60 | Ambang filter noise (rasio kata "wajar") |
| `RRF_K` | 60 | Konstanta Reciprocal Rank Fusion |

---

## 8. Menambah atau mengganti dokumen korpus

Operasi paling mungkin dilakukan orang lain. Langkahnya:

1. **Taruh PDF** di `corpus_pajak/`. Pastikan PDF bisa diekstrak teksnya (bukan hasil
   pindai gambar). Kalau hasil scan, OCR dulu — lihat D7 (`Permendagri No 8 Tahun 2024_OCR.pdf`).
2. **Daftarkan** di `PETA_DOKUMEN` (`rag_pipeline.py`):

   ```python
   "namafile.pdf": ("D11", "Label sumber - jenis & tahun", "D11_Nama_Tampilan.pdf"),
   ```

   - `doc_id` (`D11`) → dipakai LLM untuk sitasi `[D#]`.
   - **Label sumber** → ikut diindeks BM25, jadi **sertakan jenis & tahun dokumen**
     (mis. "Permendagri 9/2026 - PKB & BBN-KB 2026") agar bisa dibedakan dari dokumen
     serupa beda tahun.
   - **Nama tampilan** → muncul di daftar referensi/panel konteks.
3. **Bangun ulang index** — cache lama tidak otomatis tahu ada dokumen baru:

   ```python
   import rag_pipeline as rag
   rag.load_index(rebuild=True)        # ekstrak + chunk + embed ulang, lalu simpan cache
   ```

   Atau hapus `hf_cache/rag_index_v2.npz` lalu jalankan apa pun yang memanggil `load_index()`.
4. **Untuk Space:** salin PDF ke `hf-space/corpus_pajak/` dan samakan `PETA_DOKUMEN` di
   `hf-space/rag_pipeline.py`, lalu upload ulang. Cache di-rebuild otomatis saat cold start.

> **Catatan:** filter noise (`WORDY_MIN = 0.60`) membuang passage yang didominasi angka/tabel.
> Dokumen tarif yang padat angka bisa kehilangan sebagian isi — turunkan `WORDY_MIN` bila perlu.

---

## 9. Troubleshooting

| Gejala | Penyebab | Solusi |
|--------|----------|--------|
| `RuntimeError: OPENROUTER_API_KEY tidak ditemukan` | `.env` belum ada / salah lokasi | Buat `rag/.env` berisi `OPENROUTER_API_KEY=sk-or-v1-...`; di Space, set **Space secret** dengan nama sama |
| Instalasi `torch` besar / error CUDA | pip mengunduh build CUDA | Pasang CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Query pertama sangat lambat (~2–4 menit) | Cold start: download model IndoBERT (~500 MB) + bangun index | Wajar sekali di awal; run berikutnya pakai cache. `app.py` sudah warm-up encoder saat startup |
| Jawaban "informasi tidak ditemukan" padahal ada | Passage relevan tak masuk `TOP_K`, atau dibuang filter noise | Naikkan `TOP_K`; cek apakah pertanyaan komparatif lintas-dokumen (lihat Keterbatasan) |
| Korpus baru tak muncul di jawaban | Cache lama masih dipakai | `load_index(rebuild=True)` (lihat bagian 8) |
| Space "Sleeping" / lambat saat dibuka | HF mematikan Space yang idle | Buka ulang untuk membangunkan; tunggu cold start selesai |
| Hang saat startup (headless / tanpa internet) | Telemetri Gradio | Sudah dinonaktifkan via `GRADIO_ANALYTICS_ENABLED=False` di `app.py` |

---

## 10. Contoh tanya-jawab

Bentuk keluaran `python rag_pipeline.py "<pertanyaan>"` (struktur dari fungsi `__main__`).
Teks jawaban di bawah bersifat **ilustratif** — kata persis bervariasi karena LLM
(`temperature=0.2`), dan nomor halaman/skor bergantung hasil retrieval aktual.

```text
PERTANYAAN: Bagaimana cara penilaian NJOP untuk PBB-P2?

JAWABAN:
 Penilaian NJOP untuk PBB-P2 dilakukan secara massal maupun individual sesuai
 ketentuan PMK 85/2024 [D3]. NJOP ditetapkan berdasarkan ... [D3]. Bila objek
 pajak tidak memenuhi kriteria penilaian massal, dilakukan penilaian individual [D3].

REFERENSI:
  [1] D3_PMK_85_2024_NJOP.pdf - hal. 4 (skor 0.61)
  [2] D3_PMK_85_2024_NJOP.pdf - hal. 6 (skor 0.55)
  [3] D5_Modul_PBB_UT.pdf     - hal. 12 (skor 0.48)
```

Di frontend Gradio (`app.py`), jawaban yang sama tampil di gelembung chat, dan daftar
`REFERENSI` muncul sebagai panel lipat **"Lihat chunk yang dipakai sebagai konteks"**
berisi teks penuh tiap passage (nama file, halaman, skor). Pertanyaan NJOP di atas memang
mengutip PMK 85/2024 `[D3]` sebagaimana dicatat di `laporan_rag_1.0.md`.

---

## 11. Keamanan & biaya

- Kunci `OPENROUTER_API_KEY` hanya di `.env` (lokal, gitignored) dan Space secret —
  tidak pernah di-hardcode atau di-commit.
- Space bersifat **publik**: pengunjung memakai kredit OpenRouter pemilik. Disarankan
  memasang batas kredit di dashboard OpenRouter atau menjadikan Space privat.
- System prompt mewajibkan LLM menjawab **hanya** dari konteks dan jujur bila informasi
  tak ada — mengurangi risiko halusinasi angka/pasal.

---

## 12. Keterbatasan

- Jawaban terbatas pada isi 10 dokumen; fakta di luar korpus tidak tersedia — sistem
  menyatakannya jujur, bukan mengarang.
- Query komparatif lintas-tahun (mis. "beda tarif 2023 vs 2025") bergantung pada
  ketersediaan kedua dokumen di top-k.
- Inference jalan di CPU; latensi query pertama setelah cold start lebih tinggi.
