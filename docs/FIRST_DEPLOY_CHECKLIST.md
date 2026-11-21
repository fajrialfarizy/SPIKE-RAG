# Checklist Deploy Pertama — Lab Prestasi RAG

Panduan operasional untuk Supervisor. Tidak perlu menulis kode; semua langkah di bawah
dilakukan lewat browser dan Streamlit Cloud.

Angka cold start yang diukur di mesin pengembangan (Windows, model sudah terunduh):

| Metrik | Nilai |
| --- | --- |
| `import app` (Streamlit + torch + chromadb) | ± 4,5 detik |
| `build_runtime` (Chroma + model e5 + warm-up) | ± 35 detik |
| RSS setelah siap | ± 900 MB |
| Embedding satu pertanyaan (setelah warm-up) | ± 50 ms |

> **Penting:** 900 MB lebih besar dari batas bawah Streamlit Cloud (690 MB). Pilih mesin
> dengan RAM ≥ 1 GB. Jika hanya tersedia 690 MB, aplikasi akan kehabisan memori; alternatif
> yang perlu persetujuan Anda adalah jalur ONNX untuk model yang sama (belum diimplementasikan).

## 1. Repo privat

- Repo GitHub **private**, branch utama `main`.
- Pastikan `.gitignore` sudah mengecualikan: `.streamlit/secrets.toml`, `chroma_db/`,
  `data/artifacts/`, `data/kb_work/`, `data/dummy_pdfs/`, `logs/`, `pii_denylist.txt`.
- Jangan pernah commit `secrets.toml`, kunci Groq, service-account JSON, atau korpus asli
  yang berisi data peserta.

## 2. Streamlit Cloud → Secrets

Buka **Settings → Secrets** dan tempel (nilai asli, bukan placeholder):

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

Membuat `password_hash`: jalankan `python scripts/make_password_hash.py` di komputer Anda,
masukkan password lewat prompt (tidak tampil di layar), lalu salin hasilnya. Password mentah
tidak pernah disimpan di mana pun.

## 3. Folder Drive dan service account

Struktur folder (satu folder, tanpa subfolder):

```
Lab Prestasi KB/
├── manifest.json
└── chroma_backup.zip
```

1. Buat service account di Google Cloud Console, aktifkan **Google Drive API**.
2. Buat key JSON, ambil `client_email`-nya.
3. Di Google Drive, klik folder → **Share** → masukkan `client_email` sebagai **Viewer**.
4. Service account hanya butuh scope `drive.readonly`; aplikasi tidak pernah menulis ke Drive.

## 4. Unggah artefak pertama

Di komputer yang memiliki korpus hasil ingestion:

```bash
python scripts/run_ingestion.py --source drive   # atau --source local lalu unggah manual
```

Unggah `data/artifacts/chroma_backup.zip` dan `data/artifacts/manifest.json` ke folder Drive
di atas (timpa keduanya bersamaan; manifest lama dengan zip baru akan ditolak karena checksum).

## 5. Setelah deploy: yang harus diverifikasi

1. **Login salah** → pesan galat generik, tidak ada detail.
2. **Login benar** → layar chat muncul, tiga tombol contoh tersedia.
3. **Tiga pertanyaan** (satu spesifik, satu perbandingan, satu di luar cakupan):
   - jawaban masuk akal dan **blok Sumber hanya berisi laporan yang dikutip**;
   - status di luar cakupan memakai teks penolakan, bukan jawaban karangan.
4. **Kuota sesi**: kirim pertanyaan kedua langsung setelah yang pertama → pesan "tunggu";
   setelah 20 pertanyaan dalam satu jam → pesan batas jam.
5. **Refresh knowledge base**: setelah ingestion ulang dan unggah artefak baru, tunggu
   ± 5 menit (TTL manifest) lalu ajukan pertanyaan tentang isi laporan baru. Bila gagal,
   aplikasi tetap menyajikan versi lama.
6. **Lockout**: 5 kali login salah → akun terkunci 5 menit. Ingat: ini akun bersama, jadi
   penguncian berlaku untuk semua pengguna (bisa disalahgunakan untuk menolak layanan).
7. **Log**: Streamlit Cloud → **Manage app → Logs**. Cari baris:
   `app_runtime_ready ... load_seconds=... warmup_ms=... rss_mb=...`,
   `first_query latency_ms=... rss_mb=...`, dan `app_turn status=... latency_ms=...`.
   Log hanya berisi metadata — tidak pernah teks pertanyaan, jawaban, atau kunci.
8. **Memori**: bandingkan `rss_mb` di log dengan kuota mesin. Jika mendekati batas, turunkan
   konkurensi atau ajukan jalur ONNX.

## 6. Bila terjadi masalah

| Gejala | Kemungkinan penyebab | Tindakan |
| --- | --- | --- |
| Layar "konfigurasi belum lengkap" | Secrets kurang satu seksi | Periksa `[auth]`, `[groq]`, `[kb]` |
| Layar galat teknis saat login berhasil | Manifest/zip tidak valid atau SQLite < 3.35 | Cek log: `kb_*`/`SQLiteVersionError` |
| Jawaban "terlalu banyak permintaan" | Kuota Groq habis / rate limit | Tunggu beberapa menit, cek kuota Groq |
| Semua pengguna terkunci | 5 login salah | Tunggu 5 menit |
