# Checklist Deploy Pertama — Lab Prestasi RAG

Panduan langkah demi langkah untuk Supervisor. Semua langkah dilakukan lewat browser;
tidak perlu menulis kode.

**Deploy pertama hanya butuh SATU akun layanan (service account): akun *app* yang
read-only.** Artefak (zip + manifest) diunggah **manual** lewat Drive web. Akun
*ingestion* yang bisa menulis baru dipakai setelah aplikasi terbukti jalan (bagian 10).

| Metrik | Nilai (mesin pengembangan) | Status |
| --- | --- | --- |
| `pip install -r requirements.txt` (venv baru, cache pip hangat) | lihat laporan Fix Pack 12 | terukur |
| `import app` (Streamlit + chromadb + torch) | ± 2-5 detik | terukur |
| `build_runtime` (Chroma + model e5 + warm-up) | ± 70-100 detik | terukur |
| RSS setelah siap (`rss_mb`) | ± 920-970 MB | terukur |
| Kecocokan pin Linux / Streamlit Cloud | — | **belum diverifikasi** |

> **Penting:** ± 920 MB lebih besar dari batas bawah Streamlit Cloud (690 MB). Pilih
> mesin dengan RAM ≥ 1 GB. Bila hanya tersedia 690 MB, aplikasi akan kehabisan memori;
> alternatifnya jalur ONNX untuk model yang sama, yang **belum diimplementasikan**
> (rencananya ada di `docs/ONNX_PLAN.md`).

## 0. Ringkasan urutan

1. Repo GitHub **private**, branch **`phase3-app`**.
2. Verifikasi dua berkas artefak di komputer ini (bagian 2).
3. Buat **satu** service account app (read-only), catat e-mail-nya (bagian 3).
4. Share folder Drive ke e-mail itu sebagai **Viewer**, lalu unggah dua berkas manual
   (zip dulu, manifest terakhir) — bagian 4.
5. Buat app di Streamlit Cloud dari branch `phase3-app`, Python **3.12**, tempel
   Secrets (bagian 5-6), deploy.
6. Verifikasi daftar di bagian 8, lalu kirim hasilnya ke saya (bagian 9).

## 1. Repo dan branch

- Repo GitHub **private**.
- Branch yang dipakai: **`phase3-app`** (bukan `main`). Pilih branch ini saat membuat
  app di Streamlit Cloud.
- `.gitignore` **wajib** mengecualikan: `.streamlit/secrets.toml`, `chroma_db/`,
  `data/artifacts/`, `data/kb_work/`, `data/dummy_pdfs/`, `logs/`, `pii_denylist.txt`.
- Jangan pernah commit `secrets.toml`, kunci Groq, JSON service account, token Hugging
  Face, atau korpus asli yang berisi data peserta. Repo private bukan alasan menyimpan
  kunci di git: riwayat git tetap menyimpannya selamanya.

`chroma_db/` **tidak** di-commit: aplikasi membacanya dari artefak Drive. Tanpa
`[kb] source = "drive"` di Secrets, aplikasi akan mencari `chroma_db/` lokal dan gagal.

## 2. Siapkan dan periksa artefak di komputer ini

Dua berkas yang akan diunggah, di folder proyek:

```
data/artifacts/chroma_backup.zip     <- artefak basis pengetahuan
data/artifacts/manifest.json         <- ringkasan + checksum zip
```

Periksa keduanya **sebelum** diunggah:

```bash
python scripts/print_artifact_info.py
```

Skrip itu mencetak nama zip, ukurannya, SHA-256, serta ringkasan manifest (versi,
model embedding, dimensi, jumlah laporan, jumlah chunk) dan memastikan checksum zip
sama dengan yang tercatat di manifest. Skrip ini tidak memanggil Drive, tidak memakai
model, dan tidak pernah mencetak kunci rahasia. Bila ada `problem:`, jangan unggah
dulu; laporkan ke saya.

## 3. Service account app (satu-satunya akun untuk deploy pertama)

1. Buka Google Cloud Console → aktifkan **Google Drive API**.
2. **IAM & Admin → Service Accounts → Create service account**.
3. Buat key **JSON**, unduh sekali (file ini rahasia; jangan di-commit, jangan
   dikirim lewat chat).
4. Catat kolom **`client_email`** di file JSON itu, bentuknya seperti
   `lab-prestasi-reader@nama-project.iam.gserviceaccount.com`.

Akun ini **hanya membaca**. Jangan memberinya hak Editor.

### Membaca Folder ID dari URL Drive

Buka folder Drive di browser. URL-nya berbentuk:

```
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz012345
                                        \_______________________________/
                                                ini Folder ID
```

Salin bagian setelah `/folders/` dan sebelum `?` (bila ada). Itulah
`[kb] drive_folder_id`.

### Share folder ke akun app

Di folder Drive: **Share** → tempel `client_email` akun app → pilih **Viewer** →
kirim. Tanpa langkah ini aplikasi akan gagal mengunduh artefak.

## 4. Unggah artefak secara manual (zip dulu, manifest terakhir)

Aturan yang tidak boleh dilanggar:

- Folder harus berisi **tepat satu** `chroma_backup.zip` dan **tepat satu**
  `manifest.json`. Aplikasi **menolak start** bila ada duplikat nama.
- Unggah **`chroma_backup.zip` lebih dulu**, lalu **`manifest.json` terakhir**.
  Urutan ini disengaja: bila manifest lama masih menunjuk zip lama, aplikasi menolak
  ketidakcocokan checksum dan tetap menyajikan versi lama daripada menyajikan setengah
  artefak.
- Saat mengganti berkas, **timpa** file lama (Drive: *Manage versions* / *Replace*),
  jangan membuat salinan dengan nama sama.

## 5. Buat app di Streamlit Cloud

1. **share.streamlit.io → Create app → Deploy a public/private app → GitHub**.
2. Repo: repo privat di atas. **Branch: `phase3-app`**. Main file: `app.py`.
3. **Advanced settings → Python version: 3.12.** (Pin di `requirements.txt` sudah
   diverifikasi di Python 3.12; Python 3.14 belum diuji.)
4. Lanjut ke Secrets (bagian 6) sebelum menekan Deploy, atau Deploy lalu simpan Secrets
   dan **Reboot app**.

Streamlit Cloud memasang `requirements.txt` saja. Paket ingestion (`pymupdf`,
`reportlab`, `gspread`, `pillow`) ada di `requirements-ingest.txt` dan hanya dipakai di
Colab; `pytest` dan `onnxruntime` ada di `requirements-dev.txt`.

## 6. Streamlit Secrets — semua kunci

**Manage app → Settings → Secrets.** Ganti setiap placeholder. Struktur ini sama
dengan `.streamlit/secrets.toml.example`.

```toml
[auth]
username = "labprestasi"
# Buat di komputer ini: python scripts/make_password_hash.py
# Format: pbkdf2_sha256$<iterations>$<salt>$<digest>
password_hash = "pbkdf2_sha256$600000$REPLACE_WITH_SALT$REPLACE_WITH_DIGEST"

[groq]
api_key = "gsk_REPLACE_ME"

[kb]
source = "drive"
local_dir = "chroma_db"
manifest_path = "data/artifacts/manifest.json"
drive_folder_id = "REPLACE_WITH_FOLDER_ID_DARI_URL"
drive_manifest_name = "manifest.json"
drive_artifact_name = "chroma_backup.zip"
work_dir = "data/kb_work"
manifest_ttl_seconds = 300
# Opsional: id file persis. Isi hanya bila mencari berdasarkan nama bermasalah.
# manifest_file_id = "REPLACE_ME"
# artifact_file_id = "REPLACE_ME"

[gcp_service_account]
type = "service_account"
project_id = "REPLACE_WITH_PROJECT_ID"
private_key_id = "REPLACE_WITH_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nREPLACE_WITH_KEY\n-----END PRIVATE KEY-----\n"
client_email = "lab-prestasi-reader@REPLACE_WITH_PROJECT_ID.iam.gserviceaccount.com"
client_id = "REPLACE_WITH_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/REPLACE_WITH_SA_EMAIL"
```

- `[gcp_service_account]` = akun **app** (read-only). Isi dari file JSON langkah 3;
  `private_key` tetap memakai `\n` sebagai pemisah baris.
- `private_key` dan `api_key` tidak pernah dicetak aplikasi; jangan menempelkannya di
  chat, tiket, atau screenshot.
- **HF_TOKEN (opsional, tidak dipakai aplikasi).** Streamlit Secrets bukan variabel
  lingkungan, jadi `HF_TOKEN` di Secrets tidak berpengaruh. Model
  `intfloat/multilingual-e5-small` bersifat publik dan diunduh tanpa token (hanya
  muncul peringatan rate-limit). Bila host Anda mendukung variabel lingkungan, set
  `HF_TOKEN` di sana. Token ini bukan kunci Groq dan tidak pernah dicatat di log.

Membuat `password_hash`: di komputer ini jalankan
`python scripts/make_password_hash.py`, masukkan password lewat prompt (tidak tampil di
layar), salin hasilnya. Password mentah tidak pernah disimpan.

## 7. Log dan memori

**Manage app → Logs.** Cari satu baris startup:

```
app_startup embedding_runtime=sentence-transformers embedding_model=intfloat/multilingual-e5-small
kb_version=... kb_source=drive chunks=... kb_load_seconds=... embedder_load_seconds=...
embedder_warmup_ms=... total_load_seconds=... rss_mb=...
```

- `rss_mb` = memori proses saat siap; bandingkan dengan kuota mesin (bagian 0).
- `kb_load_seconds` (Drive + Chroma) dan `embedder_load_seconds` (model) memisahkan
  penyebab lambat.
- `Manage app → Metrics` (bila tersedia) menampilkan pemakaian memori/CPU berjalan.
- Baris lain: `first_query ... rss_mb=...`, `app_turn status=... latency_ms=...`,
  `kb_switched version=...`, `kb_refresh_failed error_type=...`.
- Log **tidak pernah** memuat teks pertanyaan, jawaban, dokumen, atau kunci.

## 8. Setelah deploy: yang harus diverifikasi

1. **Login salah** → pesan galat generik tanpa detail; **5 kali salah** → akun
   terkunci 5 menit untuk semua orang (akun bersama). Pastikan pesan terkunci muncul.
2. **Login benar** → layar chat muncul.
3. **Hanya 2-3 pertanyaan** (hemat kuota Groq 200.000 token/hari):
   - satu spesifik (kendala/tema): jawaban masuk akal, **blok Sumber hanya berisi
     laporan yang dikutip**;
   - satu perbandingan (juara vs belum juara): dua judul kelompok, setiap poin
     berakhir dengan sitasi, tidak ada kode `LAP...` yang bocor, dan **tidak ada**
     bagian "Perbedaan utama" (bagian itu dimatikan di Fix Pack 12);
   - satu di luar cakupan: teks penolakan standar, bukan jawaban karangan.
4. **Pesan antrean**: bila layanan Groq lambat (> 3 detik menunggu), status
   menampilkan "Sedang mengantre karena batas layanan, perkiraan tunggu ~N detik."
5. **Refresh knowledge base**: unggah ulang zip lalu manifest (urutan sama), tunggu
   ± 5 menit (TTL manifest), lalu tanyakan isi laporan baru. Bila gagal, aplikasi tetap
   menyajikan versi lama dan menulis `kb_refresh_failed` di log.

## 9. Yang dikirim balik ke saya

1. Baris **`app_startup`** lengkap dari Logs (terutama `rss_mb`,
   `total_load_seconds`, `chunks`).
2. **Waktu instalasi** yang ditampilkan Streamlit Cloud saat build selesai, bila ada.
3. **Teks galat apa pun** yang muncul (Logs atau layar), apa adanya — jangan
   meringkas pesan galat.

## 10. Langkah berikutnya (setelah deploy pertama terbukti jalan)

Akun **ingestion** (boleh menulis) baru dibuat kemudian, untuk menjalankan
`scripts/run_ingestion.py` dari Colab agar artefak terunggah otomatis dan `rag_status`
di Sheets ikut diperbarui. Saat itu:

1. Service account kedua; aktifkan Drive API **dan** Sheets API.
2. Share folder Drive sebagai **Editor** (perlu menulis artefak) dan spreadsheet
   sebagai **Editor** (perlu `rag_status`).
3. Simpan kredensialnya di **Colab Secrets**: `GCP_SERVICE_ACCOUNT_JSON`, `SHEET_ID`,
   `DRIVE_FOLDER_ID`.

Jangan memberi akun app hak Editor: satu-satunya penulis artefak adalah ingestion.

## 11. Bila terjadi masalah

| Gejala | Kemungkinan penyebab | Tindakan |
| --- | --- | --- |
| Layar "konfigurasi belum lengkap" | Secrets kurang satu seksi | Periksa `[auth]`, `[groq]`, `[kb]` |
| `app_startup` tidak muncul | Proses mati saat memuat (kemungkinan memori) | Cek `rss_mb`/kuota; pilih mesin lebih besar |
| Galat `DriveArtifactError` saat login | Folder belum di-share ke akun app, Folder ID salah, atau ada duplikat nama | Bagian 3-4 |
| `kb_refresh_failed` setelah unggah ulang | Zip dan manifest tidak sepasang (checksum beda) | Unggah zip dulu, manifest terakhir |
| Jawaban "terlalu banyak permintaan" | Batas token per menit Groq | Tunggu beberapa menit |
| Jawaban "kuota harian layanan sedang habis" | Batas 200.000 token/hari | Tunggu reset harian; jangan menaikkan limit di kode |
| Semua pengguna terkunci | 5 login salah | Tunggu 5 menit |
| Perbandingan bilang "terjadi kendala teknis" | Model mengembalikan JSON tidak valid/terpotong | Catat waktu, lihat `invalid_structured_output` di log |
