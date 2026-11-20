# Lab Prestasi — RAG Chatbot MVP

Chatbot berbasis website (Streamlit) dengan arsitektur Retrieval-Augmented Generation (RAG) untuk membantu mahasiswa mengakses pembelajaran, kendala, evaluasi, dan rekomendasi dari dokumen evaluasi kompetisi terdahulu.

## Struktur Repositori

```
lab-prestasi-rag/
├── app.py                     # Streamlit UI & Chat
├── config.py                  # Parameter & konfigurasi terpusat (bebas magic numbers)
├── enums.py                   # CompetitionType, AchievementLevel, SectionType, RagStatus
├── requirements.txt           # Pustaka ter-pin
├── pytest.ini                 # Konfigurasi pengujian
├── core/                      # Engine RAG (Analyzer, Retriever, Context Builder, Citations, Guardrails)
├── ingestion/                 # Pipeline ekstraksi PDF, chunker, embedder, repos
│   ├── repos/                 # Abstraksi repositori (SheetsRepo & DriveRepo: Google & Lokal)
├── scripts/                   # Pembuat korpus dummy, password hash, tuning, eval
│   └── make_dummy_corpus.py   # Generator korpus dummy (15 dokumen uji)
├── data/                      # Direktori korpus dummy lokal (.pdf)
├── tests/                     # Unit test & integrasi
│   ├── fixtures/              # Mock CSV competition_reports & report_documents
│   ├── test_config.py
│   ├── test_enums.py
│   └── test_repos.py
├── .streamlit/
│   └── secrets.toml.example   # Template kredensial Streamlit Secrets
└── README.md
```

## Menjalankan Pengujian Fase 0

1. Pastikan dependensi terpasang:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan pembuatan korpus data dummy:
   ```bash
   python scripts/make_dummy_corpus.py
   ```

3. Jalankan unit test:
   ```bash
   pytest tests/ -v
   ```
