# Tanya-Jawab RAG — Persiapan Sidang/Presentasi

Catatan tanya-jawab untuk mempertahankan (defend) subsistem RAG "Asisten Hukum Pajak"
(metode ke-7 STKI). Semua jawaban dipetakan langsung dari kode `rag/rag_pipeline.py`
dan `rag/app.py`. Untuk gambaran menyeluruh lihat [`OVERVIEW.md`](OVERVIEW.md);
diagram alur ada di [`diagram_alur_rag.png`](diagram_alur_rag.png).

---

## 1. File apa yang benar-benar berjalan? Notebook masih dipakai?

**Yang berjalan (runtime) = `app.py` + `rag_pipeline.py`.**
- `app.py` (baris 18) `import rag_pipeline as rag`, lalu memanggil `rag.jawab()`,
  `rag.load_index()`, `rag.embed()`. Jadi `app.py` = frontend Gradio (UI chat +
  panel konteks), `rag_pipeline.py` = otak RAG (chunking, retrieve hybrid, jawab GPT).
- **Notebook `STKI_RAG_LLM.ipynb` bukan bagian runtime.** Ia **tidak** mengimpor
  `rag_pipeline`; sebaliknya mendefinisikan ulang sendiri `embed/retrieve/jawab` di
  dalam selnya. Fungsinya sebagai **laporan/walkthrough mandiri**, tidak dipanggil app.
- Versi cloud menjalankan **salinan** di `rag/hf-space/` (`hf-space/app.py` +
  `hf-space/rag_pipeline.py`), bukan file di `rag/` langsung.

---

## 2. BM25, IndoBERT, dan RRF itu untuk apa?

### BM25 — pencarian leksikal (kata persis)
- Algoritma ranking klasik (pengembangan TF-IDF): menilai kecocokan **kata demi kata**,
  meredam kata umum, menormalisasi panjang dokumen.
- **Untuk:** menangkap token/singkatan/angka yang harus **persis** — mis. "PPnBM", "2025".
- **Contoh:** query "PKB 2024" → mengarah ke Permendagri 8/2024 (D7) karena memuat
  token `pkb` **dan** `2024`.
- **Lemah jika sendirian:** buta makna ("pajak mobil mewah" tak cocok dengan "PPnBM").

### IndoBERT — pencarian dense/semantik (makna)
- Model IndoBERT (Sentence-BERT, `firqaaa/indo-sentence-bert-base`) mengubah teks jadi
  **vektor 768 dimensi**; kemiripan diukur **cosine similarity**.
- **Untuk:** menangkap **makna** walau beda kata.
- **Contoh:** "pajak mobil mewah" → nyambung ke "PPnBM kendaraan bermotor" (D10).
- **Lemah jika sendirian:** meleset pada token persis/singkatan/angka.

### RRF — penggabung dua ranking (Reciprocal Rank Fusion)
- Menggabungkan dua daftar hasil: tiap dokumen dapat `1/(K+rank)` dari **tiap** daftar,
  lalu dijumlahkan (`RRF_K = 60`).
- **Untuk:** BM25 dan cosine punya **skala skor beda jauh**; RRF memakai **peringkat**,
  bukan nilai, jadi tak perlu menormalisasi skala atau menyetel bobot.
- Dokumen yang tinggi di **kedua** metode terangkat paling atas.

### Kenapa digabung ("kenapa tidak salah satu saja?")
> Dense kuat di makna tapi lemah di token persis; BM25 kebalikannya. Digabung, keduanya
> saling menambal. RRF menyatukannya tanpa menyetel bobot karena hanya pakai peringkat.
> Inilah **retrieval hybrid**.

---

## 3. Berapa nilai minimum confidence agar AI tidak menjawab (tidak relevan)?

**Tidak ada ambang numerik sama sekali di kode.**
- `retrieve()` **selalu** mengembalikan top-5 (`TOP_K=5`), berapa pun skornya; `jawab()`
  **selalu** mengirimnya ke GPT. Tidak ada baris `if skor < X: tolak`.
- Keputusan "tidak menjawab" **murni dikerjakan GPT** lewat **system prompt** grounding:
  *"Jika informasi tidak ada di dalam konteks, katakan dengan jujur bahwa informasi tidak
  ditemukan... Jangan mengarang angka atau pasal."* Ini disebut **LLM-based abstention**.
- **Jangan tertukar:** `WORDY_MIN=0.60` dan `MIN_CHARS=40` adalah **filter noise saat
  indexing** (buang chunk sampah/tabel tarif), **bukan** ambang relevansi query.

> Kelemahan desain ini: jika semua passage tidak relevan, sistem tetap mengirim ke GPT dan
> bergantung pada kejujuran LLM. Jika ingin threshold sungguhan (mis. gerbang cosine),
> angkanya harus **dikalibrasi dari data**, bukan ditebak.

---

## 4. Berapa chunk per PDF di korpus?

Total **pas 700 passage** dari 10 dokumen (dihitung dari `rag_index_v2.npz`):

| ID | Chunk | Halaman | Sumber |
|----|------:|--------:|--------|
| D1 | 72 | 25 | UU 12/1985 — PBB |
| D2 | 73 | 22 | UU 12/1985 — PBB (salinan) |
| D3 | **146** | 41 | PMK 85/2024 — Penilaian NJOP PBB-P2 |
| D4 | 70 | 19 | PMK 234/2022 — PBB pertambangan/kehutanan |
| D5 | **137** | 40 | Modul PBB Universitas Terbuka |
| D6 | 52 | 16 | Permendagri 7/2025 — PKB & BBN-KB 2025 |
| D7 | 36 | 12 | Permendagri 8/2024 — PKB & BBN-KB 2024 |
| D8 | 34 | 11 | Permendagri 6/2023 — PKB & BBN-KB 2023 |
| D9 | 44 | 12 | PMK 8/2024 — PPN Kendaraan Listrik (EV) |
| D10 | 36 | 15 | PMK 5/2022 — PPnBM Kendaraan Bermotor |
| **Total** | **700** | 213 | 10 dokumen |

- Rata-rata **70 chunk/dokumen**, tapi timpang (34–146).
- **Kenapa timpang:** jumlah chunk sebanding dengan **panjang & padatnya teks**, bukan
  angka tetap; D3/D5 paling tebal, D8/D10 paling tipis.
- Angka ini **setelah filter noise**; "halaman" = jumlah halaman yang menghasilkan ≥1 chunk.

---

## 5. Apakah kita memakai cosine similarity?

**Ya — di jalur dense (IndoBERT).**
- `retrieve()` (baris 258): `cos = matriks @ q / (norm(matriks)*norm(q) + 1e-9)` → rumus
  cosine similarity.
- Skor yang **ditampilkan** ke user juga cosine (`item["skor"] = float(cos[idx])`).
- Karena `embed()` memakai `normalize_embeddings=True` (vektor panjang 1), **dot product
  = cosine**, jadi pembagian norm hanya penegasan.

**Jangan bilang "hanya cosine".** Retrieval-nya hybrid:
| Komponen | Ukuran kemiripan |
|----------|------------------|
| Dense (IndoBERT) | cosine similarity ✅ |
| Leksikal (BM25) | skor BM25 (bukan cosine) |
| Penggabung | RRF — pakai peringkat, bukan skor |

> Cosine ini konsep yang sama dengan metode ke-4 (VSM/TF-IDF); bedanya vektornya dari
> embedding IndoBERT 768-d, bukan bobot TF-IDF.

---

## 6. Penjelasan kode embedding (`get_encoder` & `embed`)

**Lazy load:** `_encoder = None` — model IndoBERT (~500 MB) baru dimuat saat pertama
dibutuhkan, bukan saat import. Membuat `import rag_pipeline` tetap instan.

**`get_encoder()` — singleton:**
- `if _encoder is None:` → model dimuat **sekali**; panggilan berikutnya pakai ulang.
- `import torch` di dalam fungsi → menunda import berat.
- `torch.set_num_threads(int(_N_THREAD))` → batasi 4 thread agar CPU tak 100%.
- `SentenceTransformer(MODEL_EMBED, device="cpu")` → paksa CPU (HF Space gratis CPU-only).

**`embed()`:**
- `if isinstance(texts, str): texts = [texts]` → seragamkan input (encode maunya list);
  makanya di `retrieve()` ditulis `q = embed(query)[0]`.
- `normalize_embeddings=True` → vektor dinormalisasi jadi panjang 1 → **dot product =
  cosine** di retrieval. Ini kuncinya.
- `batch_size=16`, `show_progress_bar=False` → kompromi memori & log rapi.

---

## 7. Setelah teks menjadi vektor, lalu apa?

1. **Vektor korpus jadi satu matriks** `700 × 768` (`_matriks`), lalu **di-cache** ke
   `rag_index_v2.npz` (fase indexing, sekali saja).
2. **Query di-embed** dengan model yang sama (`q = embed(query)[0]`) → ruang 768-d yang sama.
3. **Cosine** query vs seluruh 700 passage sekaligus: `cos = matriks @ q / (...)`.
4. **BM25 paralel** → array 700 skor leksikal.
5. **RRF** menggabungkan peringkat cosine + BM25 → ambil **top-5**.
6. **5 passage** dirakit jadi "KONTEKS DOKUMEN" `[D#]` → dikirim ke GPT → jawaban ber-sitasi.

> Peran vektor berhenti di **pengukuran kemiripan**. Yang dikirim ke GPT adalah **teks
> asli** passage, bukan vektornya.

---

## 8. Apa fungsi cache index?

**Menyimpan hasil embedding korpus ke disk agar tidak dihitung ulang tiap run.**
- Membangun index (ekstraksi + embed 700 passage di CPU) makan **~2–4 menit**. Cache
  membuatnya cukup **sekali**.
- File: `rag/hf_cache/rag_index_v2.npz`, isinya `korpus` (metadata) + `matriks` (vektor).
- Logika `load_index()` bertingkat: (1) sudah di RAM → pakai; (2) ada file `.npz` → muat
  (~detik); (3) belum ada → bangun penuh lalu `np.savez`.

**Dua level cache (jangan tertukar):**
| Cache | Isi | Fungsi |
|-------|-----|--------|
| `rag_index_v2.npz` | embedding korpus | hindari re-embed |
| `hf_cache/` (HF_HOME) | model IndoBERT ~500 MB | hindari re-download |

> **Konsekuensi:** jika korpus/logika berubah, cache lama tetap dipakai → perubahan tak
> muncul. Paksa: `load_index(rebuild=True)`. Sufiks `_v2` menandai versi cache saat logika
> chunking berubah (jadi per-halaman), agar cache `v1` lama tidak terpakai keliru.

---

## 9. Skor relevansi diambil dari mana?

Ada beda antara skor **penentu peringkat** vs skor **yang ditampilkan**:
- **Ditampilkan** (mis. `skor 0.61`) = **cosine** (IndoBERT dense) — dipilih karena
  rentang 0–1 mudah ditafsirkan (`item["skor"] = float(cos[idx])`).
- **Penentu peringkat** top-5 = **RRF** (`urut = np.argsort(-rrf)[:top_k]`), bukan cosine.

Tiap item hasil `retrieve()` punya **tiga skor**:
| Field | Dari | Peran |
|-------|------|-------|
| `skor` | cosine (dense) | ditampilkan (0–1) |
| `skor_bm25` | BM25 | info kekuatan kecocokan kata |
| `rank` | urutan RRF | posisi 1–5 |

> **Konsekuensi:** passage rank #1 bisa punya cosine lebih kecil dari rank #2 — **bukan
> bug**; artinya #1 menang karena kuat di BM25. Ini bukti hybrid bekerja.

---

## 10. `TOP_K = 5` itu apa?

**Jumlah passage paling relevan yang diambil sebagai konteks GPT.**
- Dari 700 passage, RRF mengurutkan semua, lalu **5 teratas** dipakai jadi konteks +
  ditampilkan di panel UI (`urut = np.argsort(-rrf)[:top_k]`).
- **Trade-off:** terlalu banyak → token besar (mahal/lambat) + konteks tak relevan
  mengganggu; terlalu sedikit → informasi bisa tak terambil (buruk untuk query
  lintas-dokumen). **5** = kompromi umum RAG. Bisa diubah bila perlu.
- Istilahnya **top-k retrieval** (`k = 5`).

---

## 11. Di mana vector storage-nya?

**Tidak memakai vector database khusus (FAISS/Chroma/Pinecone/pgvector).** Vektor
disimpan sebagai:
1. **Di RAM:** matriks NumPy `_matriks` (`700 × 768`).
2. **Di disk:** file **`rag/hf_cache/rag_index_v2.npz`** (via `np.savez`).

**Pencarian brute-force / exact** (tanpa index ANN): `cos = matriks @ q / (...)`
membandingkan query ke seluruh 700 vektor sekaligus — cepat (mili-detik) karena korpus kecil.

| Peran | Sistem besar | Proyek ini |
|-------|-------------|-----------|
| Simpan vektor | FAISS/Pinecone/pgvector | matriks NumPy + `.npz` |
| Cari terdekat | ANN (HNSW/IVF) | brute-force `matriks @ q` (exact) |
| Kapan upgrade | ratusan ribu+ vektor | belum perlu (700 vektor) |

> Vector DB baru diperlukan pada skala ratusan ribu–jutaan vektor. Untuk 700 vektor,
> matriks NumPy sudah cukup (KISS), dan hasilnya **exact**, bukan aproksimasi.

---

## 12. Cara kerja RRF (Reciprocal Rank Fusion)

**Rumus:** `RRF(d) = Σ 1/(K + rank(d))` atas tiap daftar, `K = 60`.

**Langkah di kode:**
```python
rank_cos[np.argsort(-cos)] = np.arange(len(cos))   # cosine terbaik → rank 0
rank_lex[np.argsort(-lex)] = np.arange(len(lex))   # BM25 terbaik → rank 0
rrf = 1.0/(RRF_K + rank_cos) + 1.0/(RRF_K + rank_lex)
urut = np.argsort(-rrf)[:top_k]
```
(Di implementasi ini rank mulai dari **0**.)

**Contoh (K=60), kenapa "bagus di keduanya" menang:**
| Passage | rank cosine | rank BM25 | RRF | Hasil |
|---------|:----------:|:---------:|:----|:-----:|
| A | 0 | 49 | 1/60 + 1/109 = **0,0259** | |
| B | 2 | 1 | 1/62 + 1/61 = **0,0325** | 🏆 #1 |
| C | 1 | 30 | 1/61 + 1/90 = **0,0275** | #2 |

A juara di cosine tapi jeblok di BM25, kalah dari B yang bagus-bagus di keduanya. **RRF
memberi hadiah pada konsensus.**

**Peran `K=60`:** meredam dominasi rank teratas (`1/60` vs `1/61` selisih tipis). K kecil
(mis. 1) → `1/1` vs `1/2` terlalu tajam. 60 = nilai standar paper asli (Cormack dkk., 2009).

> Tiga kata kunci: **rank bukan skor**, **konsensus dua metode**, **tanpa tuning bobot**.

---

## Lampiran — konstanta penting (`rag_pipeline.py`)

| Konstanta | Nilai | Arti |
|-----------|-------|------|
| `MODEL_EMBED` | `firqaaa/indo-sentence-bert-base` | Encoder IndoBERT (768-d) |
| `MODEL_LLM` | `openai/gpt-4o-mini` | Generator jawaban (via OpenRouter) |
| `TOP_K` | 5 | Jumlah passage konteks |
| `KALIMAT_PER_CHUNK` / `MAX_KATA_CHUNK` | 4 / 160 | Ukuran chunk |
| `MIN_CHARS` | 40 | Panjang minimum chunk (filter) |
| `WORDY_MIN` | 0.60 | Ambang filter noise (rasio kata "wajar") |
| `RRF_K` | 60 | Konstanta Reciprocal Rank Fusion |
