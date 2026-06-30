import os
import pickle
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env", override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"🔑 Kunci aktif: {SUPABASE_KEY[:15]}...") 
print("⏳ 1. Menghubungkan ke Supabase...")
opts = ClientOptions(schema="parfum")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=opts)

print("⏳ 2. Mengambil seluruh data dari tabel perfumes...")
all_data = []
limit = 1000
offset = 0

while True:
    res = supabase.table("perfumes").select("*").range(offset, offset + limit - 1).execute()
    data = res.data
    if not data:
        break
    all_data.extend(data)
    offset += limit

print(f"✅ Berhasil mengambil {len(all_data)} parfum dari database.")

df = pd.DataFrame(all_data)

# Jika database kosong, hentikan program
if df.empty:
    print("❌ Tabel perfumes masih kosong! Silakan import data ke Supabase terlebih dahulu.")
    exit()

print("⏳ 3. Memproses teks (Feature Engineering)...")
# Menyesuaikan dengan nama kolom BARU dari skema database Anda
df['features'] = (
    df['description'].fillna('') + " " + 
    df['notes'].fillna('') + " " + 
    df['top_notes'].fillna('') + " " + 
    df['middle_notes'].fillna('') + " " + 
    df['base_notes'].fillna('') + " " + 
    df['main_accords'].fillna('') + " " + 
    df['waktu_penggunaan'].fillna('') + " " + 
    df['occasion'].fillna('') + " " + 
    df['gender_target'].fillna('') + " " +
    df['wangi_vibe'].fillna('')
).str.lower()

print("⏳ 4. Melatih Model TF-IDF (Membangun Jam Terbang)...")
tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.8)
tfidf.fit(df['features'])

print("⏳ 5. Menyimpan 'Otak' AI ke file .pkl...")
with open(MODELS_DIR / 'model_tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("✅ SELESAI! File 'model_tfidf.pkl' berhasil dibuat.")