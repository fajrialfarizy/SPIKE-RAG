# Checklist Deploy Pertama — Lab Prestasi RAG

Panduan langkah demi langkah untuk Supervisor. Semua langkah dilakukan lewat browser;
tidak perlu menulis kode. Angka di bawah diukur di mesin pengembangan (Windows,
model sudah terunduh), jadi angka di Streamlit Cloud akan berbeda.

| Metrik | Nilai (mesin pengembangan) | Status |
| --- | --- | --- |
| `import app` (Streamlit + chromadb + torch) | ± 4,5 detik | terukur |
| `build_runtime` (Chroma + model e5 + warm-up) | ± 35 detik | terukur |
| RSS setelah siap | ± 900 MB | terukur |
| Embedding satu pertanyaan (setelah warm-up) | ± 50 ms | terukur |
| Kecocokan pin Linux / Streamlit Cloud | — | **belum diverifikasi** |

> **Penting:** ± 900 MB lebih besar dari batas bawah Streamlit Cloud (690 MB). Pilih
> mesin dengan RAM ≥ 1 GB. Bila hanya tersedia 690 MB, aplikasi akan kehabisan memori;
> alternatifnya jalur ONNX untuk model yang sama, yang **belum diimplementasikan**
> (rencananya ada di `docs/ONNX_PLAN.md`).

> **Unduhan model (opsional):** `intfloat/multilingual-e5-small` diunduh sekali saat
> runtime pertama. Set `HF_TOKEN` di Secrets agar unduhan memakai kuota akun Hugging
> Face (menghindari peringatan unauthenticated dan laju unduhan yang dibatasi). Token
> ini hanya untuk unduhan model, bukan untuk Groq, dan tidak pernah dicatat di log.

## 0. Ringkasan urutan

1. Buat repo GitHub **private** (branch `main`).
2. Buat dua service account Google (ingestion dan app) — lihat bagian 3.
3. Jalankan ingestion di Colab, unggah artefak (zip dulu, manifest terakhir).
4. Buat app di Streamlit Cloud, tempel Secrets, deploy.
5. Verifikasi daftar di bagian 6.

## 1. Repo privat

- Repo GitHub **private**, branch utama `main`.
- `.gitignore` **wajib** mengecualikan: `.streamlit/secrets.toml`, `chroma_db/`,
  `data/artifacts/`, `data/kb_work/`, `data/dummy_pdfs/`, `logs/`, `pii_denylist.txt`.
- Jangan pernah commit `secrets.toml`, kunci Groq, JSON service account, token Hugging
  Face, atau korpus asli yang berisi data peserta. Repo private bukan alasan untuk
  menyimpan kunci di git: riwayat git tetap menyimpannya selamanya.

Yang **di-commit**: `app.py`, `app_texts.py`, `config.py`, `core/`, `ingestion/`,
`scripts/`, `tests/`, `docs/`, `.streamlit/config.toml`,
`.streamlit/secrets.toml.example`, `requirements.txt`, `requirements-ingest.txt`,
`requirements-dev.txt`, `pytest.ini`, `README.md`.

`chroma_db/` **tidak** di-commit: aplikasi membacanya dari artefak Drive. Tanpa
`KB_SOURCE=drive` di Secrets, aplikasi akan mencari `chroma_db/` lokal dan gagal.

Paket yang dipasang Streamlit Cloud hanya `requirements.txt`. Paket ingestion
(`pymupdf`, `reportlab`, `gspread`, `pillow`) ada di `requirements-ingest.txt` dan
hanya dipakai di Colab; `pytest` dan `onnxruntime` ada di `requirements-dev.txt`.
Sebuah tes (`tests/test_import_hygiene.py`) gagal bila modul aplikasi suatu saat
mengimpor paket ingestion.

## 2. Streamlit Cloud → Secrets

Buat app: **share.streamlit.io → New app → repo private → branch `main` →
main file `app.py`**. Lalu buka **Settings → Secrets** dan tempel (nilai asli, bukan
placeholder):

```toml
[auth]
username = "labprestasi"
password_hash = "pbkdf2_sha256$600000$...$..."

[groq]
api_key = "gsk_..."

[kb]
source = "drive"
drive_folder_id = "<id folder Drive>"
drive_manifest_name = "manifest.json"
drive_artifact_name = "chroma_backup.zip"
work_dir = "data/kb_work"
manifest_ttl_seconds = 300
# Opsional: id file persis. Bila diisi, aplikasi mengunduh file itu langsung dan
# tidak mencari berdasarkan nama. Kosongkan bila tidak yakin.
# manifest_file_id = "..."
# artifact_file_id = "..."

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

`[gcp_service_account]` di Secrets adalah service account **app** (read-only),
bukan service account ingestion.

Membuat `password_hash`: di komputer Anda jalankan
`python scripts/make_password_hash.py`, masukkan password lewat prompt (tidak tampil
di layar), lalu salin hasilnya. Password mentah tidak pernah disimpan.

## 3. Dua service account (model yang benar)

Kedua akun memakai folder Drive yang sama, tetapi haknya berbeda:

| | Akun **ingestion** | Akun **app** |
| --- | --- | --- |
| Dipakai di | Colab (`scripts/run_ingestion.py`) | Streamlit Secrets |
| Akses folder | Editor (harus menulis artefak) | Viewer (hanya membaca) |
| Scope OAuth | `spreadsheets`, `drive.file`, `drive.readonly` | `drive.readonly` saja |
| Alasan | Menulis `rag_status` di Sheets dan mengunggah/mengganti `chroma_backup.zip` + `manifest.json` | Aplikasi tidak pernah menulis; bila kuncinya bocor, penyerang tidak bisa mengubah basis pengetahuan |
| Kredensial | `GCP_SERVICE_ACCOUNT_JSON`, `SHEET_ID`, `DRIVE_FOLDER_ID` di Colab Secrets | `[gcp_service_account]` di Streamlit Secrets |

Langkah membuat akun **ingestion**:

1. Google Cloud Console → aktifkan **Google Drive API** dan **Google Sheets API**.
2. Buat service account, buat key JSON, unduh sekali.
3. Share folder Drive ke `client_email` akun ingestion sebagai **Editor**.
4. Share spreadsheet ke `client_email` yang sama sebagai **Editor** (agar
   `rag_status` bisa diperbarui).

Langkah membuat akun **app**:

1. Service account kedua di project yang sama (atau project lain).
2. **Tidak** perlu akses Sheets. Cukup Drive API aktif.
3. Share folder Drive yang sama ke `client_email` akun app sebagai **Viewer**.
4. Salin seluruh isi key JSON-nya ke `[gcp_service_account]`.

Jangan memberi akun app hak Editor: satu-satunya tulis ke Drive dilakukan ingestion.

## 4. Unggah artefak pertama

Di Colab (bukan di Streamlit Cloud):

```bash
pip install -r requirements-ingest.txt
python scripts/run_ingestion.py --source google
```

`run_ingestion.py` membuat `chroma_backup.zip` dan `manifest.json`, lalu mengunggah
**zip lebih dulu, manifest terakhir**. Urutan ini disengaja: bila unggahan manifest
gagal, manifest lama masih menunjuk checksum zip lama, sehingga aplikasi menolak
ketidakcocokan dan tetap menyajikan versi lama daripada menyajikan setengah artefak.

Bila mengunggah manual ke Drive:

1. Hapus/rencanakan penggantian; folder harus berisi **tepat satu** `chroma_backup.zip`
   dan **tepat satu** `manifest.json`.
2. Unggah `chroma_backup.zip` lebih dulu (timpa file lama, jangan buat salinan baru).
3. Unggah `manifest.json` terakhir (timpa file lama).
4. Bila ada dua file dengan nama sama, aplikasi **menolak** startup dengan pesan jelas;
   rapikan dulu folder Drive.

## 5. Deploy

Tekan **Deploy**. Streamlit Cloud memasang `requirements.txt` saja, lalu menjalankan
`app.py`. Login belum bisa dipakai sampai Secrets tersimpan dan app di-reboot.

## 6. Setelah deploy: yang harus diverifikasi

1. **Login salah** → pesan galat generik, tidak ada detail; 5 kali salah → akun
   terkunci 5 menit untuk semua orang (akun bersama).
2. **Login benar** → layar chat muncul, tiga tombol contoh tersedia.
3. **Tiga pertanyaan**:
   - satu spesifik (kendala/tema): jawaban masuk akal, **blok Sumber hanya berisi
     laporan yang dikutip**, baris cakupan (bila ada) menyebut angka yang sama dengan
     jumlah baris di Sumber;
   - satu perbandingan (juara vs belum juara): ada dua judul kelompok, setiap poin
     berakhir dengan sitasi, dan tidak ada kode `LAP...` yang bocor ke jawaban;
   - satu di luar cakupan: teks penolakan standar, bukan jawaban karangan.
4. **Kuota sesi**: kirim pertanyaan kedua langsung setelah yang pertama → pesan
   "tunggu"; setelah 10 pertanyaan dalam satu jam → pesan batas jam; setelah 15
   pertanyaan dalam sehari → pesan batas harian. Ini penghitung **per sesi browser**,
   jadi beberapa tab/pengguna bisa melewatinya; lihat catatan risiko.
5. **Pesan antrean**: bila layanan Groq sedang lambat (lebih dari 3 detik menunggu),
   status pemrosesan menampilkan "Sedang mengantre karena batas layanan, perkiraan
   tunggu ~N detik."
6. **Refresh knowledge base**: setelah ingestion ulang dan unggah artefak baru, tunggu
   ± 5 menit (TTL manifest) lalu tanyakan isi laporan baru. Bila gagal, aplikasi tetap
   menyajikan versi lama dan menulis `kb_refresh_failed` di log.
7. **Lockout**: pastikan pesan terkunci muncul setelah 5 percobaan salah.

## 7. Membaca log dan memori

Streamlit Cloud → **Manage app → Logs**.

Satu baris startup (cari `app_startup`):

```
app_startup embedding_runtime=sentence-transformers embedding_model=intfloat/multilingual-e5-small
kb_version=... kb_source=drive chunks=... kb_load_seconds=... embedder_load_seconds=...
embedder_warmup_ms=... total_load_seconds=... rss_mb=...
```

- `rss_mb` adalah memori proses saat siap. Bandingkan dengan kuota mesin.
- `kb_load_seconds` dan `embedder_load_seconds` memisahkan lambatnya Drive/Chroma dari
  lambatnya model.
- Baris lain: `first_query latency_ms=... rss_mb=...` (pertanyaan pertama),
  `app_turn status=... latency_ms=...` (setiap jawaban), `kb_switched version=...`
  (artefak baru dipakai), `kb_refresh_failed error_type=...` (artefak baru ditolak).
- Log **tidak pernah** memuat teks pertanyaan, jawaban, dokumen, atau kunci.

## 8. Bila terjadi masalah

| Gejala | Kemungkinan penyebab | Tindakan |
| --- | --- | --- |
| Layar "konfigurasi belum lengkap" | Secrets kurang satu seksi | Periksa `[auth]`, `[groq]`, `[kb]` |
| Layar galat teknis saat login berhasil | Manifest/zip tidak valid, nama file ganda, atau SQLite < 3.35 | Cek log: `kb_*`, `DriveArtifactError`, `SQLiteVersionError` |
| `app_startup` tidak muncul | Proses mati saat memuat (kemungkinan memori) | Cek kuota RAM; pertimbangkan mesin lebih besar |
| Jawaban "terlalu banyak permintaan" | Batas token per menit Groq | Tunggu beberapa menit |
| Jawaban "kuota harian layanan sedang habis" | Batas token harian Groq (200.000/hari) | Tunggu sampai jendela harian direset besok; jangan menaikkan limit di kode |
| Semua pengguna terkunci | 5 login salah | Tunggu 5 menit |
| Jawaban perbandingan bilang "terjadi kendala teknis" | Model mengembalikan JSON tidak valid/terpotong | Catat waktu, lihat `invalid_structured_output` di log; tidak ada retry otomatis karena kuota |
