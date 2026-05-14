import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Memuat variabel dari .env
load_dotenv(override=True)

app = FastAPI(title="Perfume Hybrid Recommender API")

# --- KONFIGURASI SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Mengarahkan koneksi ke skema kustom "parfum" (Perbaikan nama skema)
opts = ClientOptions(schema="parfum")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=opts)

# --- LOAD MODEL AI ---
try:
    with open('model_tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    print("❌ Peringatan: file model_tfidf.pkl tidak ditemukan. Jalankan training terlebih dahulu.")

# Daftar kata abai untuk pembersihan log debug
STOP_WORDS_ID = {
    "dan", "yang", "di", "ke", "dari", "untuk", "pada", "dengan", "ini", "itu", 
    "atau", "juga", "sebuah", "ingin", "mau", "buat", "ada", "adalah", "lagi", 
    "parfum", "wangi", "aroma", "cari", "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "sih", "dong", "deh"
}

# --- FUNGSI PEMBANTU ---
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class RecommendRequest(BaseModel):
    user_input: Optional[str] = ""
    gender: Optional[str] = ""
    waktu_pemakaian: Optional[str] = ""
    kegiatan: Optional[str] = ""
    wangi: Optional[str] = ""

# --- ENDPOINT UTAMA ---
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    # 1. Ambil data terbaru dari Supabase (Look-up)
    response = supabase.table("perfumes").select("*").execute()
    db_data = response.data
    
    if not db_data:
        raise HTTPException(status_code=404, detail="Database parfum kosong.")
    
    df_db = pd.DataFrame(db_data)
    # --- TAMBAHAN KODE PEMBERSIH NaN UNTUK JSON ---
    # 1. Pastikan kolom angka tidak mengandung NaN (isi dengan 0)
    df_db['harga'] = pd.to_numeric(df_db['harga'], errors='coerce').fillna(0)
    df_db['rating_value'] = pd.to_numeric(df_db['rating_value'], errors='coerce').fillna(0)
    df_db['rating_count'] = pd.to_numeric(df_db['rating_count'], errors='coerce').fillna(0)
    
    # 2. Ubah sisa nilai NaN (seperti string yang kosong) menjadi None (akan jadi null di JSON)
    df_db = df_db.replace([np.inf, -np.inf], 0)
    df_db = df_db.replace({np.nan: None})
    # ----------------------------------------------

    # 2. Gabungkan Input User
    combined_query = " ".join([req.user_input, req.gender, req.waktu_pemakaian, req.kegiatan, req.wangi])
    clean_query = clean_text(combined_query)
    
    if not clean_query:
        return {"status": "error", "message": "Input pencarian kosong."}

    # 3. Proses Content-Based Filtering (CBF)
    # PERBAIKAN: Menyesuaikan dengan nama kolom tabel database yang baru
    df_db['features'] = (
        df_db['description'].fillna('') + " " + 
        df_db['notes'].fillna('') + " " + 
        df_db['top_notes'].fillna('') + " " + 
        df_db['middle_notes'].fillna('') + " " + 
        df_db['base_notes'].fillna('') + " " + 
        df_db['main_accords'].fillna('') + " " + 
        df_db['waktu_penggunaan'].fillna('') + " " + 
        df_db['occasion'].fillna('') + " " + 
        df_db['gender_target'].fillna('') + " " +
        df_db['wangi_vibe'].fillna('')
    ).str.lower()

    # Menggunakan model TF-IDF untuk mengubah data DB menjadi angka
    mat_db = tfidf.transform(df_db['features'])
    query_vec = tfidf.transform([clean_query])
    
    # Hitung kemiripan
    raw_scores = cosine_similarity(query_vec, mat_db).flatten()
    
    # Scaling skor agar terlihat lebih tinggi (Max 98%)
    max_raw = np.max(raw_scores)
    df_db['match_score_raw'] = raw_scores
    if max_raw > 0:
        df_db['display_score'] = (raw_scores / max_raw) * 0.98
    else:
        df_db['display_score'] = 0

# --- BAGIAN REKOMENDASI (main.py) ---
    
    # 1. Ambil SEMUA hasil CBF yang skornya di atas 30% (0.3)
    cbf_results = df_db[df_db['display_score'] >= 0.30].sort_values(by='display_score', ascending=False)
    
    final_output = []
    query_keywords = {w for w in clean_query.split() if w not in STOP_WORDS_ID and len(w) > 2}

    for _, row in cbf_results.iterrows():
        matched = [w for w in query_keywords if w in str(row['features'])]
        
        # Bahasa AI yang lebih natural & elegan
        if matched:
            alasan_ai = f"Aroma ini sangat selaras dengan kepribadian Anda, memancarkan nuansa {', '.join(matched)} yang Anda cari."
        else:
            alasan_ai = "Karakteristik racikan parfum ini memiliki kedekatan pola yang kuat dengan preferensi unik Anda."

        final_output.append({
            "nama": row.get('nama_parfum', row.get('nama', 'Unknown')),
            "brand": row.get('brand', 'Bespoke'),
            "match_score": f"{round(row['display_score'] * 100, 1)}%",
            "match_score_num": round(row['display_score'] * 100, 1), # Angka murni untuk sorting di FE
            "notes": row.get('notes', ''),
            "harga": float(row.get('harga', 0)), # Pastikan float untuk sorting
            "deskripsi": row.get('description', row.get('deskripsi', '')),
            "link_pembelian": row.get('url_produk', row.get('link_pembelian', '#')),
            "url_foto": row.get('url_foto', ''),
            "debug_alasan": alasan_ai,
            "tipe": "CBF"
        })

    # 2. Ambil 15 besar hasil CF (Collaborative/Populer) sebagai alternatif
    already_added = [p['nama'] for p in final_output]
    df_db['cf_score'] = df_db['rating_value'] * np.log1p(df_db['rating_count'])
    cf_candidates = df_db[~df_db['nama_parfum' if 'nama_parfum' in df_db else 'nama'].isin(already_added)].sort_values(by='cf_score', ascending=False).head(15)
    
    for _, row in cf_candidates.iterrows():
        final_output.append({
            "nama": row.get('nama_parfum', row.get('nama', 'Unknown')),
            "brand": row.get('brand', 'Bespoke'),
            "match_score": "Populer",
            "match_score_num": 0,
            "notes": row.get('notes', ''),
            "harga": float(row.get('harga', 0)),
            "deskripsi": row.get('description', row.get('deskripsi', '')),
            "link_pembelian": row.get('url_produk', row.get('link_pembelian', '#')),
            "url_foto": row.get('url_foto', ''),
            "debug_alasan": "Parfum ini merupakan mahakarya yang paling banyak dicari dan direkomendasikan oleh komunitas penikmat wewangian saat ini.",
            "tipe": "CF"
        })

    return {"status": "success", "data": final_output}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)