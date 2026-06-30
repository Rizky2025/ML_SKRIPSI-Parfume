# 🤖 ML — Sistem Rekomendasi Parfum (CBF + CF)

Mesin rekomendasi (FastAPI) untuk sistem rekomendasi parfum. Dokumentasi besar proyek
ada di `../DOKUMENTASI/`.

## Struktur Folder
```
ML_SKRIPSI-Parfume/
├── main.py                 # Server API (entry point) — port 8000
├── requirements.txt        # Daftar library
├── .env                    # Kredensial Supabase (TIDAK di-commit)
│
├── dataset/
│   ├── parfum/             # Data inti parfum (perfumes_rows.json/.csv)
│   ├── interaksi/          # Data rating/profil user untuk CF
│   └── sql/                # Script seed data ke Supabase
│
├── models/                 # Model terlatih (.pkl) — hasil training
│   ├── model_tfidf.pkl     #   otak CBF
│   └── model_cf_item.pkl   #   otak CF
│
├── scripts/                # Script training & generator data
│   ├── train_model.py      #   ⭐ latih CBF + CF (hasilkan kedua .pkl)
│   ├── generate_dummy_cf.py
│   ├── generate_dataset.py
│   ├── generate_sql.py
│   └── train_supabase.py
│
└── _arsip/                 # File lama tak terpakai (boleh diabaikan)
```

## Cara Pakai
```bash
# 1. Install library (sekali saja)
pip install -r requirements.txt

# 2. (PENTING) Sinkronkan data dari Supabase + latih model.
#    Jalankan urutan ini setiap kali isi tabel perfumes di Supabase berubah:
python scripts/sync_from_supabase.py   # tarik data DB -> dataset/parfum/*.json+csv
python scripts/generate_dataset.py     # buat data interaksi CF (id sesuai DB)
python scripts/train_model.py          # latih CBF + CF -> models/*.pkl

# 3. Jalankan API
python main.py        # http://127.0.0.1:8000

# (opsional) ukur akurasi model
python scripts/evaluate_model.py
```

> ⚠️ **Wajib sinkron:** model HARUS dilatih dari data Supabase (bukan file JSON lama),
> agar `id` parfum di model CF cocok dengan yang diambil `main.py` saat runtime. Jika tidak,
> rekomendasi CF tidak akan muncul.

## Catatan
- Semua script memakai path absolut berbasis lokasi file, jadi **bisa dijalankan dari
  folder mana pun** (mis. `python scripts/train_model.py` dari root, atau dari dalam `scripts/`).
- Sumber data CF saat ini: `dataset/interaksi/dataset_rating_variatif.csv` (rating
  berkorelasi preferensi). Lihat `../DOKUMENTASI/01_LOG_BATCH.md` untuk riwayat keputusan.
