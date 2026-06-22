import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

load_dotenv(override=True)
app = FastAPI(title="Perfume Hybrid Recommender API")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

# --- LOAD MODELS ---
try:
    with open('model_tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    try:
        with open('model_cf_item.pkl', 'rb') as f:
            cf_item_matrix = pickle.load(f)
    except:
        cf_item_matrix = pd.DataFrame() # Fallback jika gagal
except FileNotFoundError:
    print("❌ Model AI belum di-train.")

class RecommendRequest(BaseModel):
    user_input: Optional[str] = ""
    gender: Optional[str] = ""
    waktu_pemakaian: Optional[str] = ""
    kegiatan: Optional[str] = ""
    wangi: Optional[str] = ""

def clean_text(text: str) -> str:
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', ' ', text).strip()

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    res = supabase.table("perfumes").select("*").execute()
    df_db = pd.DataFrame(res.data)
    if df_db.empty: return {"status": "error", "message": "Database kosong"}

    # ========================================================
    # 1. ALGORITMA CBF (CONTENT-BASED FILTERING)
    # ========================================================
    # Merakit query AI dari dropdown user
    combined_query = f"{req.user_input} {req.gender} {req.waktu_pemakaian} {req.kegiatan} {req.wangi}"
    clean_query = clean_text(combined_query)

    # Menyatukan data parfum untuk dinilai AI
    df_db['features'] = (
        df_db['description'].fillna('') + " " + 
        df_db['notes'].fillna('') + " " + 
        df_db['waktu_penggunaan'].fillna('') + " " + 
        df_db['occasion'].fillna('') + " " + 
        df_db['gender_target'].fillna('') + " " +
        df_db['main_accords'].fillna('')
    ).str.lower()

    # PERBEDAAN DENGAN FILTER BIASA: Kita hitung kemiripan Teks (Cosine Similarity)
    mat_db = tfidf.transform(df_db['features'])
    query_vec = tfidf.transform([clean_query])
    raw_scores = cosine_similarity(query_vec, mat_db).flatten()
    
    df_db['display_score'] = raw_scores * 100
    
    # CBF mengambil data yang kecocokannya di atas 5% berdasarkan AI TF-IDF
    cbf_results = df_db[df_db['display_score'] >= 5.0].sort_values(by='display_score', ascending=False)
    
    final_output = []
    top_cbf_id = None # Disimpan untuk trigger CF

    for idx, row in cbf_results.head(10).iterrows():
        if top_cbf_id is None: top_cbf_id = row['id']
        
        final_output.append({
            "id": row['id'],
            "nama": row.get('nama_parfum', 'Unknown'),
            "brand": row.get('brand', '-'),
            "match_score": f"{round(row['display_score'], 1)}%",
            "match_score_num": round(row['display_score'], 1),
            "deskripsi": row.get('description', ''),
            "debug_alasan": f"Metode CBF (AI Text Matching): Parfum ini memiliki kedekatan pola deskripsi sebesar {round(row['display_score'], 1)}% dengan apa yang Anda cari di dropdown.",
            "tipe": "CBF"
        })

    # ========================================================
    # 2. ALGORITMA CF (COLLABORATIVE FILTERING - DUMMY BASED)
    # ========================================================
    # Logika: Jika user suka parfum nomor 1 di CBF, maka dummy users juga merekomendasikan...
    cf_output = []
    already_added = [p['id'] for p in final_output]

    if top_cbf_id and not cf_item_matrix.empty and top_cbf_id in cf_item_matrix.index:
        # Mengambil parfum yang mirip berdasarkan rating dari dummy users
        similar_items = cf_item_matrix[top_cbf_id].sort_values(ascending=False)[1:15]
        
        for sim_id, sim_score in similar_items.items():
            if sim_id not in already_added and sim_score > 0:
                # Cari data detail parfum di database
                match = df_db[df_db['id'] == sim_id]
                if not match.empty:
                    row = match.iloc[0]
                    cf_output.append({
                        "id": row['id'],
                        "nama": row.get('nama_parfum', 'Unknown'),
                        "brand": row.get('brand', '-'),
                        "match_score": "Rekomendasi User",
                        "match_score_num": 0,
                        "deskripsi": row.get('description', ''),
                        "debug_alasan": f"Metode CF: Berdasarkan data interaksi dummy, pengguna yang menyukai karakter pencarian Anda juga memberikan rating tinggi pada parfum ini.",
                        "tipe": "CF"
                    })
                    already_added.append(row['id'])

    # Gabung array
    return {"status": "success", "data": final_output + cf_output[:10]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)