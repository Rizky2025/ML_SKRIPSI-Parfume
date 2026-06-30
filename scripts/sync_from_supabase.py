"""
======================================================================
SINKRONISASI DATA DARI SUPABASE (Batch 4)
======================================================================
Menarik SELURUH isi tabel public.perfumes dari Supabase (sumber yang
SAMA PERSIS dengan yang diambil main.py saat runtime), lalu menyimpannya
ke dataset/parfum/perfumes_rows.json + .csv.

Tujuan: memastikan model dilatih dari data yang identik dengan database
produksi, sehingga id parfum di model CF cocok dengan id saat runtime.

Jalankan script ini SETIAP KALI isi tabel perfumes di Supabase berubah,
diikuti: generate_dataset.py -> train_model.py.

Cara jalan:  python scripts/sync_from_supabase.py
======================================================================
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_JSON = BASE_DIR / "dataset" / "parfum" / "perfumes_rows.json"
OUT_CSV = BASE_DIR / "dataset" / "parfum" / "perfumes_rows.csv"

load_dotenv(BASE_DIR / ".env", override=True)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# schema "public" = sama dengan yang dipakai main.py
supabase: Client = create_client(
    SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public")
)

print("⏳ Menarik data dari tabel public.perfumes ...")
all_data = []
limit, offset = 1000, 0
while True:
    res = supabase.table("perfumes").select("*").range(offset, offset + limit - 1).execute()
    if not res.data:
        break
    all_data.extend(res.data)
    offset += limit

if not all_data:
    print("❌ Tabel perfumes kosong! Pastikan data sudah di-import ke Supabase.")
    sys.exit(1)

# Simpan JSON (sumber kebenaran untuk training)
OUT_JSON.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")

# Simpan CSV pendamping (dipakai generate_dataset.py)
pd.DataFrame(all_data).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ Berhasil sinkron {len(all_data)} parfum dari Supabase.")
print(f"   -> {OUT_JSON}")
print(f"   -> {OUT_CSV}")
print("\nLangkah berikut: python scripts/generate_dataset.py  lalu  python scripts/train_model.py")
