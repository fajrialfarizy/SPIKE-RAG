# Rencana Runtime Embedding ONNX (belum diimplementasikan)

Dokumen ini **hanya rencana**. Tidak ada kode ONNX di repositori, tidak ada
dependensi ONNX yang dipasang oleh aplikasi, dan `onnxruntime` sengaja hanya ada di
`requirements-dev.txt`.

Tujuan: bila deploy pertama ke Streamlit Cloud gagal karena memori, model embedding
yang **sama** (`intfloat/multilingual-e5-small`, 384 dimensi, cosine) dijalankan
dengan ONNX Runtime alih-alih PyTorch. Mengganti model bukan pilihan: manifest,
koleksi Chroma, dan vektor yang tersimpan terikat pada model tersebut.

Label status yang dipakai di dokumen ini:

- **Terverifikasi**: sudah dicek di mesin pengembangan Windows pada 2026-11-21.
- **Belum diverifikasi**: belum dijalankan di sini; harus diuji sebelum dipakai.

## 1. Kenapa ONNX, dan apa yang TIDAK berubah

- Yang berubah hanya cara menghitung vektor. Model, prefix `query: ` / `passage: `,
  normalisasi L2, dimensi 384, dan metrik cosine tetap.
- Yang **tidak** boleh berubah: teks chunk, metadata chunk, `report_id`, urutan
  section, dan nama model pada manifest. Bila salah satu berubah, koleksi lama tidak
  boleh dipakai.
- Klaim penghematan memori **belum diverifikasi**. Perkiraan umum (PyTorch vs
  onnxruntime untuk model kecil) tidak cukup; angkanya harus diukur dengan langkah 6.

## 2. Sumber file ONNX

1. Periksa apakah repo Hugging Face `intfloat/multilingual-e5-small` sudah memuat
   file ONNX (`onnx/model.onnx`, `onnx/model_fp16.onnx`, `model.onnx`).
   **Status: belum diverifikasi** — jangan mengasumsikan ada.
2. Bila tidak ada, ekspor sendiri dengan Optimum:

   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx \
     --model intfloat/multilingual-e5-small \
     --task feature-extraction \
     ./onnx-e5-small
   ```

   Perintah, nama file keluaran, dan dukungan `--task feature-extraction` untuk
   model ini **belum diverifikasi**. Periksa isi folder hasil ekspor; yang dipakai
   hanya `model.onnx` (fp32) beserta `tokenizer.json`, `tokenizer_config.json`,
   `special_tokens_map.json`, dan `config.json`.
3. **Hanya fp32.** Varian fp16/int8 ditolak: perubahan presisi tidak boleh masuk
   tanpa gerbang paritas baru.

## 3. Bentuk integrasi yang direncanakan

- `ingestion/embedder.py` mendapat implementasi kedua, misalnya
  `OnnxE5Embedder`, dengan antarmuka yang sama (`embed_query`, `embed_passages`,
  `embed_documents`) dan pengecekan dimensi yang sama.
- Pemilihan runtime lewat `config.EMBEDDING_RUNTIME`:
  - `"sentence-transformers"` (sekarang, default),
  - `"onnx"` (baru).
- Tidak ada fallback otomatis. Bila `embedding_runtime="onnx"` dan file/model gagal
  dimuat, proses **gagal keras** dengan pesan jelas; tidak boleh diam-diam kembali ke
  PyTorch atau ke vektor pseudo.
- `app_startup` sudah mencatat `embedding_runtime`, sehingga log deploy langsung
  menunjukkan runtime yang benar-benar dipakai.

## 4. Gerbang paritas (wajib sebelum artefak baru dipakai)

Dijalankan offline dengan koleksi lama dan file ONNX yang sama:

1. **Cosine ≥ 0,999 untuk SEMUA chunk korpus.** Bandingkan vektor PyTorch dan ONNX
   per chunk; hitung cosine antar keduanya. Satu chunk di bawah 0,999 = gagal.
2. **Cosine ≥ 0,999 untuk minimal 50 query** yang mewakili pertanyaan nyata
   (spesifik, perbandingan, rute ID, topik yang tidak ada).
3. **Top-k identik, atau hanya beda pada near-tie.** Jalankan retrieval penuh untuk
   50 query itu dan bandingkan daftar `chunk_id` pada k yang dikonfigurasi. Perbedaan
   hanya boleh terjadi bila selisih skor dua chunk yang bertukar posisi ≤ 0,001, dan
   jumlahnya dicatat.
4. **Tidak ada perubahan pada sitasi atau status.** Untuk 50 query, status
   (`ANSWERED`/`NOT_FOUND`/`NO_SOURCE`) harus sama dengan run PyTorch pada pertanyaan
   yang sama.
5. Simpan hasil gerbang sebagai lampiran laporan (angka cosine minimum, jumlah near-tie,
   daftar query).

Bila salah satu gerbang gagal, runtime ONNX **tidak dipakai**; jangan menurunkan
ambang agar lolos.

## 5. Apa yang harus di-ingest ulang

- Bila vektor ONNX lolos gerbang dan identik secara praktis, koleksi lama secara
  teknis masih kompatibel. Meski begitu, rencananya tetap **ingest ulang penuh** dengan
  runtime ONNX sebelum artefak diterbitkan, agar artefak yang beredar dibangun oleh
  runtime yang benar-benar melayaninya. Ini keputusan konservatif, bukan keharusan
  matematis.
- Artefak baru harus membawa `embedding_runtime` di manifest (lihat bagian 6). Artefak
  lama tanpa field itu dianggap `sentence-transformers`.
- Bila ingest ulang dijalankan, semua `report_id`, section, dan chunk harus sama
  jumlahnya; perbedaan jumlah chunk adalah kegagalan.

## 6. Field konfigurasi dan manifest

`config.py`:

```python
EMBEDDING_RUNTIME = "sentence-transformers"   # atau "onnx"
EMBEDDING_ONNX_MODEL_PATH = ""                # folder berisi model.onnx (hanya untuk "onnx")
```

`manifest.json` (skema tetap `1.0`, field baru opsional):

```json
{
  "embedding_model": "intfloat/multilingual-e5-small",
  "embedding_dimension": 384,
  "embedding_runtime": "onnx"
}
```

Aturan **refuse-on-mismatch**: `KnowledgeBase` dan `KBSynchronizer` membandingkan
`manifest.embedding_runtime` (default `"sentence-transformers"` bila tidak ada) dengan
`config.EMBEDDING_RUNTIME`. Bila berbeda, keduanya menolak memuat dan aplikasi
menampilkan layar galat teknis; tidak ada re-embedding otomatis, tidak ada koleksi
cadangan. Pesan galat menyebut kedua nilai.

## 7. Cara mengukur RSS

1. Di mesin pengembangan: jalankan `python -c "import app"` lalu ukur dengan
   `psutil.Process().memory_info().rss` sebelum dan sesudah memuat embedder (angka
   terukur sekarang: ± 900 MB setelah siap, Windows).
2. Di Streamlit Cloud: baca `rss_mb` pada baris `app_startup` dan pada
   `first_query ... rss_mb=...`. Bandingkan PyTorch vs ONNX dengan build yang sama.
3. Catat juga waktu muat (`embedder_load_seconds`) karena ONNX biasanya lebih cepat
   dimuat; klaim ini **belum diverifikasi** untuk mesin ini.
4. Ukur minimal tiga kali restart untuk melihat variasi.

## 8. Risiko yang diketahui

- ONNX Runtime di Linux bisa membutuhkan wheel berbeda; kecocokan dengan Streamlit
  Cloud **belum diverifikasi**.
- Tokenizer tetap dari `transformers`, jadi penghematan memori berasal dari model,
  bukan dari tokenizer.
- Bila file ONNX berasal dari ekspor sendiri, versi `optimum` dan `onnxruntime` harus
  dipin di `requirements-dev.txt` agar hasil bisa direproduksi.
- Semua langkah di dokumen ini harus diuji di workerd/mesin target sebelum deploy;
  dokumen ini bukan jaminan hasil.
