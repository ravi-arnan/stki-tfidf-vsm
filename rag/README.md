# RAG — Asisten Hukum Pajak (IndoBERT + GPT)

Metode ke-7 proyek STKI. Berbeda dengan metode 1-6 yang hanya **me-ranking** dokumen,
subsistem ini **menjawab pertanyaan** dalam Bahasa Indonesia dengan menyertakan sitasi
dokumen sumber `[D#]` — sebuah pipeline **RAG (Retrieval-Augmented Generation)**.

**Demo live (cloud):** <https://huggingface.co/spaces/raviarnan/asisten-hukum-pajak-rag>

## Tech stack

| Lapisan | Komponen | Peran |
|---------|----------|-------|
| Bahasa & runtime | **Python 3.12**, venv + pip | Semua kode subsistem |
| Ekstraksi PDF | **pdfplumber** | Ekstraksi teks per-halaman dari 10 PDF korpus |
| Retrieval leksikal | **rank-bm25** (BM25Okapi) | Cocokkan kata/singkatan/angka persis (mis. "PPnBM", "2025") |
| Retrieval dense | **sentence-transformers** + **torch** (CPU) — encoder **IndoBERT** `firqaaa/indo-sentence-bert-base` (768-d) | Cocokkan makna semantik query↔passage |
| Fusi ranking | Reciprocal Rank Fusion (RRF, K=60) — **numpy** | Gabung ranking BM25 + dense tanpa tuning bobot |
| Generasi jawaban | **GPT** `openai/gpt-4o-mini` via **OpenRouter** (SDK `openai`) | Susun jawaban Bahasa Indonesia + sitasi `[D#]` |
| Konfigurasi rahasia | **python-dotenv** (`.env` → `OPENROUTER_API_KEY`) | Muat kunci API; tidak di-hardcode |
| Frontend | **Gradio** (`gr.ChatInterface`) | Chat lokal + panel chunk konteks |
| Cache | `numpy.savez` (`rag_index_v2.npz`) + `HF_HOME` cache model | Hindari re-embed & re-download tiap run |
| Deploy | **Hugging Face Spaces** (runtime Gradio) | Demo cloud publik |

> `torch` di-set CPU-only dan thread BLAS dibatasi 4 (`OMP/MKL/OPENBLAS_NUM_THREADS`) agar
> mesin tetap responsif. Encoder di-load *lazy* dan di-*warm-up* saat startup `app.py`.
> Versi pasti dependency ada di `requirements.txt` di root repo.

## Cara kerja singkat

```
PDF hukum pajak ──► ekstraksi per-halaman ──► chunking (≈4 kalimat / maks 160 kata)
                                                   │  + filter noise (buang tabel tarif)
                                                   ▼
                                    700-an passage berlabel {doc_id, sumber, halaman, teks}
                                                   │
         ┌─────────────────────────────────────────┴───────────────────────────────────┐
         ▼                                                                               ▼
  BM25 (leksikal)                                                        IndoBERT (dense, 768-d)
  cocok kata/singkatan/angka persis                                      cocok makna semantik
  mis. "PPnBM", "2025"                                                   mis. "pajak mobil mewah"
         └───────────────────────────► Reciprocal Rank Fusion (RRF, K=60) ◄─────────────┘
                                                   │
                                       top-5 passage paling relevan
                                                   ▼
                            GPT (gpt-4o-mini via OpenRouter) + system prompt grounding
                                                   ▼
                          Jawaban Bahasa Indonesia + sitasi [D#] + panel chunk konteks
```

Retrieval **hybrid** dipakai karena dense saja gagal pada query yang bergantung pada
token persis (mis. "PPnBM" tidak punya tetangga semantik dekat di IndoBERT umum). BM25
menangkap kecocokan leksikal itu; RRF menggabungkan kedua ranking tanpa perlu menyetel
bobot skala skor yang berbeda.

## Korpus

Sepuluh PDF hukum pajak di `corpus_pajak/` (tema PBB + Pajak Kendaraan Bermotor).
Semua dari portal hukum resmi pemerintah (JDIH Kemenkeu, peraturan.bpk.go.id) dan
repositori Universitas Terbuka:

| ID | Sumber | Sumber resmi |
|----|--------|--------------|
| D1 | UU 12/1985 — PBB (`85uu012.pdf`) | [bphn.go.id](https://bphn.go.id/data/documents/85uu012.pdf) |
| D2 | UU 12/1985 — PBB (file lain, UU sama) | [jdih.kemenkeu.go.id](https://jdih.kemenkeu.go.id/fulltext/1985/12TAHUN~1985UU.HTM) |
| D3 | PMK 85/2024 — Penilaian NJOP PBB-P2 | [jdih.kemenkeu.go.id](https://jdih.kemenkeu.go.id/dok/pmk-85-tahun-2024/summary) |
| D4 | PMK 234/2022 — klasifikasi & penetapan NJOP PBB | [jdih.kemenkeu.go.id](https://jdih.kemenkeu.go.id/dok/234-pmk-03-2022/summary) |
| D5 | Modul PBB (PAJA3233) — Universitas Terbuka | [pustaka.ut.ac.id](https://pustaka.ut.ac.id/lib/wp-content/uploads/pdfmk/PAJA323304-M1.pdf) |
| D6 | Permendagri 7/2025 — PKB, BBN-KB & PAB 2025 | [peraturan.bpk.go.id](https://peraturan.bpk.go.id/Details/321612/permendagri-no-7-tahun-2025) |
| D7 | Permendagri 8/2024 — PKB, BBN-KB & PAB 2024 | [peraturan.bpk.go.id](https://peraturan.bpk.go.id/Details/300076/permendagri-no-8-tahun-2024) |
| D8 | Permendagri 6/2023 — PKB, BBN-KB & PAB 2023 | [peraturan.bpk.go.id](https://peraturan.bpk.go.id/Details/252380/permendagri-no-6-tahun-2023) |
| D9 | PMK 8/2024 — PPN Kendaraan Listrik (EV) | [jdih.kemenkeu.go.id](https://jdih.kemenkeu.go.id/dok/pmk-8-tahun-2024) |
| D10 | PMK 5/2022 — PPnBM Kendaraan Bermotor | [jdih.kemenkeu.go.id](https://jdih.kemenkeu.go.id/api/download/d4c58cfc-2eda-491a-a46f-a0aa52c67877/5~PMK.010~2022Per.pdf) |

## Isi folder

| Berkas | Keterangan |
|--------|------------|
| `rag_pipeline.py` | Modul inti: `build_korpus()`, `embed()`, `load_index()`, `retrieve()` (hybrid), `jawab()` |
| `app.py` | Frontend chat Gradio (lokal), lengkap dengan panel "chunk yang dipakai sebagai konteks" |
| `STKI_RAG_LLM.ipynb` | Notebook penjelasan/laporan (eksperimen bertahap) |
| `laporan_rag_1.0.md` | Laporan alur implementasi + justifikasi parameter |
| `corpus_pajak/` | 10 PDF korpus (tidak di-commit bila besar) |
| `hf_cache/` | Cache embedding (`rag_index_v2.npz`) + model IndoBERT — **gitignored** |
| `.env` | `OPENROUTER_API_KEY=...` — **gitignored**, tidak boleh di-commit |
| `hf-space/` | Salinan staging untuk deploy ke Hugging Face Spaces |

## Menjalankan lokal

```bash
# Dari root repo: pasang dependency (lihat requirements.txt utama)
source lsi-colab/.venv/bin/activate

# Kunci OpenRouter (tidak di-commit)
echo "OPENROUTER_API_KEY=sk-or-v1-..." > rag/.env

cd rag
python app.py        # buka http://127.0.0.1:7860
```

Saat pertama dijalankan, model IndoBERT (~500 MB) diunduh ke `rag/hf_cache/` dan
index embedding korpus dibangun lalu di-cache (`rag_index_v2.npz`). Menjalankan
berikutnya langsung memakai cache. `app.py` melakukan *warm-up* encoder saat startup
agar pesan pertama tidak lambat.

Uji cepat dari terminal tanpa frontend:

```bash
python rag_pipeline.py "Kendaraan bermotor apa saja yang dikenai PPnBM?"
```

## Deploy ke Hugging Face Spaces

`hf-space/` adalah salinan yang di-upload ke Space (`app.py`, `rag_pipeline.py`,
`requirements.txt`, `README.md` metadata, korpus, dan cache). Kunci API disimpan sebagai
**Space secret** `OPENROUTER_API_KEY`, bukan di dalam berkas.

> Saat mengubah logika di `rag_pipeline.py`, sinkronkan juga `hf-space/rag_pipeline.py`
> lalu upload ulang agar versi cloud ikut diperbarui.

## Konfigurasi (konstanta di `rag_pipeline.py`)

| Konstanta | Nilai | Arti |
|-----------|-------|------|
| `MODEL_EMBED` | `firqaaa/indo-sentence-bert-base` | Encoder IndoBERT (Sentence-BERT) |
| `MODEL_LLM` | `openai/gpt-4o-mini` | Generator jawaban (via OpenRouter) |
| `TOP_K` | 5 | Jumlah passage konteks |
| `KALIMAT_PER_CHUNK` / `MAX_KATA_CHUNK` | 4 / 160 | Ukuran chunk |
| `WORDY_MIN` | 0.60 | Ambang filter noise (rasio kata "wajar") |
| `RRF_K` | 60 | Konstanta Reciprocal Rank Fusion |

## Keterbatasan

- Jawaban dibatasi pada isi 10 dokumen; fakta di luar korpus (mis. rincian jenis
  kendaraan pada PP 73/2019) memang tidak tersedia — sistem akan menyatakannya jujur,
  bukan mengarang.
- Query komparatif lintas-tahun (mis. "beda tarif 2023 vs 2025") bergantung pada
  ketersediaan kedua dokumen di top-k.

## Keamanan

Kunci `OPENROUTER_API_KEY` hanya berada di `.env` (lokal, gitignored) dan sebagai Space
secret — **tidak pernah** di-hardcode atau di-commit. Space bersifat publik, sehingga
pengunjung memakai kredit OpenRouter pemilik; disarankan memasang batas kredit di
dashboard OpenRouter atau menjadikan Space privat.
