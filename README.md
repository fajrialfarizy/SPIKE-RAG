# Lab Prestasi — RAG Chatbot MVP

Chatbot berbasis Streamlit dengan arsitektur Retrieval-Augmented Generation (RAG) untuk membantu
mahasiswa memakai pembelajaran, kendala, evaluasi, dan rekomendasi dari dokumen evaluasi kompetisi
terdahulu. Jawaban selalu menyertakan blok **Sumber** berisi laporan yang benar-benar dikutip.

## Struktur Repositori

```
app.py                        # Streamlit UI (login, chat, kuota sesi)
app_texts.py                  # Seluruh teks antarmuka Bahasa Indonesia
config.py                     # Semua parameter (tanpa magic numbers)
core/                         # Engine RAG + settings, auth, kb_sync, sqlite_compat
ingestion/                    # Ekstraksi PDF, chunker, embedder, repositori
scripts/                      # make_dummy_corpus, run_ingestion, smoke_groq, show_context
tests/                        # Unit test offline (+ test bertanda `slow`)
.streamlit/config.toml        # Konfigurasi Streamlit (tanpa usage stats)
.streamlit/secrets.toml.example
requirements.txt              # runtime aplikasi (yang dipasang Streamlit Cloud)
requirements-ingest.txt       # ingestion/Colab: pymupdf, reportlab, gspread, pillow
requirements-dev.txt          # test/dev: pytest (+ onnxruntime untuk eksperimen)
docs/FIRST_DEPLOY_CHECKLIST.md
docs/ONNX_PLAN.md             # rencana runtime ONNX (belum diimplementasikan)
```

## Persiapan

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

Tiga berkas kebutuhan dipisah menurut siapa yang memasangnya:

| Berkas | Dipasang oleh | Isi |
| --- | --- | --- |
| `requirements.txt` | Streamlit Cloud | Chroma, Groq, Streamlit, Google Drive client, embedder |
| `requirements-ingest.txt` | Colab/ingestion | `-r requirements.txt` + pymupdf, reportlab, gspread, pillow |
| `requirements-dev.txt` | pengembangan lokal | `-r requirements.txt -r requirements-ingest.txt` + pytest |

`tests/test_import_hygiene.py` gagal bila `app.py`/`core/` mengimpor paket ingestion
(pymupdf, gspread, reportlab) atau pytest.

Buat korpus dummy lalu bangun knowledge base lokal:

```bash
python scripts/make_dummy_corpus.py
python scripts/run_ingestion.py --source local
```

Perintah di atas menghasilkan `chroma_db/` dan `data/artifacts/manifest.json`.

## Menjalankan aplikasi secara lokal

Mode lokal memakai `chroma_db/` yang sudah ada, tanpa Google Drive.

```bash
# 1. Hash password akun bersama (diminta lewat getpass, tidak pernah lewat argumen CLI)
python scripts/make_password_hash.py

# 2. Isi kredensial: salin template lalu isi nilainya
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#    [auth] username, password_hash  -> hasil langkah 1
#    [groq] api_key                  -> kunci Groq Anda
#    [kb]   source = "local", local_dir = "chroma_db",
#           manifest_path = "data/artifacts/manifest.json"

# 3. Jalankan
streamlit run app.py
```

Alternatif tanpa `secrets.toml` (variabel lingkungan, hanya untuk pengembangan lokal):

```bash
export LAB_PRESTASI_USERNAME=labprestasi
export LAB_PRESTASI_PASSWORD_HASH='pbkdf2_sha256$600000$...'
export GROQ_API_KEY=gsk_...
export KB_SOURCE=local
export KB_LOCAL_DIR=chroma_db
export KB_MANIFEST_PATH=data/artifacts/manifest.json
streamlit run app.py
```

`.streamlit/secrets.toml` tidak boleh di-commit (lihat `.gitignore`).

Mode Drive (`KB_SOURCE=drive`) memakai service account read-only untuk mengunduh
`manifest.json` + `chroma_backup.zip`, memverifikasi checksum, mengekstrak ke direktori versi baru,
lalu berpindah secara atomik. Bila gagal, versi lama tetap dipakai.

## Mode perbandingan

Perbandingan dua kelompok memakai jawaban JSON terstruktur yang divalidasi backend.
Bagian **Perbedaan utama** (contrast) **dimatikan secara default**
(`COMPARE_CONTRAST_MODE = "off"`). Keputusan ini diambil dari run nyata: pasangan poin
antar kelompok tidak sebanding (mis. catatan waktu slide dipasangkan dengan "terlalu
fokus pada algoritma teknis"), karena modellah yang memilih pasangan dan backend hanya
dapat memvalidasi nomor poinnya. Ini keputusan kualitas, bukan perbaikan bug, dan akan
ditinjau kembali bersama penilaian manusia Phase 4.

Nilai yang tersedia, dipilih **hanya** lewat `config.py` (tidak pernah otomatis saat
runtime):

| Nilai | Skema | Prompt | Render |
| --- | --- | --- | --- |
| `off` (default) | tanpa properti `contrasts` | tanpa instruksi contrast | tanpa "Perbedaan utama" |
| `pick` | `{"dimension", "pick": {"g1": "1", ...}}` | aturan contrast (v2.5) | memakai teks poin tervalidasi |
| `sides` | `{"dimension", "sides": [{group, text, cite}]}` | aturan contrast (v2.5) | memakai sisi tervalidasi |

Biaya statis satu panggilan compare (system prompt + skema, tokenizer e5): `off` 1.606
vs `pick` 1.983 token untuk dua kelompok (hemat 377), dan 1.718 vs 2.131 untuk tiga
kelompok. Completion juga lebih pendek karena larik contrasts tidak lagi ditulis.

## Pengujian

```bash
pytest                 # suite offline, < 30 detik
pytest -m slow         # test yang memuat model e5 (butuh model terunduh)
```

## Perintah operasional lain

```bash
python -m core.cli "Apa kendala tim pada business plan?" --debug   # pipeline via CLI
python scripts/show_context.py --question "..." --compare-by achievement_group  # konteks saja, tanpa LLM
python scripts/smoke_groq.py --chroma-dir chroma_db --delay 20     # smoke Groq (butuh GROQ_API_KEY)
# Opsi: --only id_route,catalog  --show-raw  --show-context  --token-budget 30000  --answer-effort low
```

## Batas chunk dan reflow PDF

PyMuPDF mengembalikan satu baris per baris visual tanpa penanda paragraf, jadi ingest
mereflow dulu sebelum chunking: baris yang bersambung digabung dengan satu spasi,
paragraf hanya dipisah oleh baris kosong, penanda butir, atau judul bagian, dan kata
berhubung ("terburu-" + "buru") digabung tanpa spasi. Chunk berakhir di akhir kalimat
atau akhir butir daftar; hanya satu kalimat yang sendirian melebihi batas token yang
dipecah per jendela kata (dihitung di `oversized_sentence_splits`). Semua aturan dan
anggotanya ada di `config.py` bagian PDF reflow dan chunking.

## Catatan kuota

Penghitung kuota per sesi dan limiter token Groq hidup **di memori proses**:
restart atau sleep Streamlit Cloud mereset keduanya. Kendali yang sebenarnya adalah
kuota akun Groq (8.000 TPM / 200.000 TPD), bukan penghitung sesi ini.

Antrean token (Fix Pack 7) punya batas sendiri, `QUEUE_MAX_WAIT_SECONDS` (45 s):
bila sebuah panggilan harus menunggu jendela 60 detik lebih lama dari itu, pipeline
tidak menunggu dan langsung menjawab RATE_LIMITED beserta `retry_after_seconds`.
Waktu antre yang masih diizinkan tidak dibebankan ke `PIPELINE_TOTAL_DEADLINE_SECONDS`
(batas itu hanya menjaga waktu model dan pemrosesan). Estimasi token memakai hitungan
eksak untuk bagian statis (system prompt, addendum, skema JSON) dan rasio
`TOKEN_ESTIMATE_DYNAMIC_CHARS_PER_TOKEN` (4,0 karakter/token) plus margin 12% untuk
bagian dinamis; reservasi completion memakai perkiraan (600/700/120), bukan plafon.
`scripts/replay_limiter_sequence.py` memutar ulang satu urutan dua pertanyaan dengan
jam palsu untuk memeriksa kebijakan ini tanpa memanggil Groq.

Jawaban mode bebas (specific, rute report-id, cross_report tanpa kelompok) dipangkas
di backend bila melebihi `ANSWER_MAX_BULLETS` (6) atau `ANSWER_MAX_BULLETS_PER_REPORT`
(2 per laporan): tidak ada panggilan model kedua, dan sitasi, baris cakupan serta blok
Sumber dihitung ulang dari teks final.

## Catatan SQLite

`chromadb` membutuhkan SQLite ≥ 3.35. Di Linux yang SQLite-nya lebih lama, `pysqlite3-binary`
(Linux-only pada `requirements.txt`) dipakai lewat `core/sqlite_compat.py`, yang dipanggil sebelum
`chromadb` diimpor. Bila tidak tersedia, aplikasi menampilkan layar galat teknis, bukan diam-diam
memakai SQLite lama.
