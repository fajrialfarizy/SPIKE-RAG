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
scripts/                      # make_dummy_corpus, run_ingestion, smoke_groq, make_password_hash
tests/                        # Unit test offline (+ test bertanda `slow`)
.streamlit/config.toml        # Konfigurasi Streamlit (tanpa usage stats)
.streamlit/secrets.toml.example
docs/FIRST_DEPLOY_CHECKLIST.md
```

## Persiapan

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

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

## Pengujian

```bash
pytest                 # suite offline, < 30 detik
pytest -m slow         # test yang memuat model e5 (butuh model terunduh)
```

## Perintah operasional lain

```bash
python -m core.cli "Apa kendala tim pada business plan?" --debug   # pipeline via CLI
python scripts/smoke_groq.py --chroma-dir chroma_db --delay 20     # smoke Groq (butuh GROQ_API_KEY)
# Opsi: --only id_route,catalog  --show-raw  --token-budget 30000  --answer-effort low
```

## Catatan SQLite

`chromadb` membutuhkan SQLite ≥ 3.35. Di Linux yang SQLite-nya lebih lama, `pysqlite3-binary`
(Linux-only pada `requirements.txt`) dipakai lewat `core/sqlite_compat.py`, yang dipanggil sebelum
`chromadb` diimpor. Bila tidak tersedia, aplikasi menampilkan layar galat teknis, bukan diam-diam
memakai SQLite lama.
