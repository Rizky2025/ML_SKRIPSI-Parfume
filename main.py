import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# --- PATH DASAR (robust, tidak bergantung dari mana script dijalankan) ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

load_dotenv(BASE_DIR / ".env", override=True)
app = FastAPI(title="Perfume Hybrid Recommender API")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

# --- LOAD MODEL (CBF / TF-IDF saja; CF dimatikan agar output murni hasil pencarian user) ---
try:
    with open(MODELS_DIR / 'model_tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    print("❌ Model AI belum di-train. Jalankan: python scripts/train_model.py")


class RecommendRequest(BaseModel):
    user_input: Optional[str] = ""
    gender: Optional[str] = ""
    waktu_pemakaian: Optional[str] = ""
    kegiatan: Optional[str] = ""
    wangi: Optional[str] = ""


def clean_text(text: str) -> str:
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', ' ', text).strip()


# Kata gender generik untuk guard nama vs gender (lihat blok GENDER di /recommend).
# \b agar "men" TIDAK ikut match di dalam "women".
LAWAN_GENDER = {
    'pria': r'\b(?:women|woman|wanita|lady|ladies|girl|her|miss|feminine|cewe|cewek)\b',
    'wanita': r'\b(?:men|man|pria|homme|male|gentleman|cowo|cowok|masculine|boy)\b',
}


# ======================================================================
# BUKTI CBF — menjelaskan DARI MANA persentase kecocokan berasal
# ======================================================================
def bukti_cbf(query_vec, doc_vec, feature_names, field_texts, parfum_fields, top_n=5):
    """
    Mengurai skor cosine TF-IDF menjadi kata-kata penyumbang.
    Karena vektor TF-IDF sudah dinormalisasi (L2), skor cosine = jumlah
    perkalian bobot per-kata. Jadi kontribusi tiap kata = qv[kata] * dv[kata].
    """
    qv = query_vec.toarray().flatten()
    dv = doc_vec.toarray().flatten()
    contrib = qv * dv
    total = contrib.sum()

    kata_cocok = []
    for i in np.argsort(contrib)[::-1]:
        if contrib[i] <= 0 or len(kata_cocok) >= top_n:
            break
        term = feature_names[i]

        # Dari kategori input mana kata ini berasal?
        kategori = "Pencarian bebas Anda"
        for nama_kat, teks in field_texts:
            if re.search(r'\b' + re.escape(term) + r'\b', teks):
                kategori = nama_kat
                break

        # Di bagian mana parfum kata ini muncul?
        ditemukan_di = [
            label for label, isi in parfum_fields
            if re.search(r'\b' + re.escape(term) + r'\b', str(isi).lower())
        ]

        kata_cocok.append({
            "kata": term,
            "dari_kategori": kategori,
            "ditemukan_di": ditemukan_di or ["deskripsi parfum"],
            "kontribusi_persen": round(100 * contrib[i] / total, 1) if total > 0 else 0.0,
        })
    return kata_cocok


def alasan_cbf(kata_cocok, skor):
    """Susun kalimat penjelasan yang mudah dibaca user."""
    alasan = []
    for k in kata_cocok:
        lokasi = " & ".join(k["ditemukan_di"])
        alasan.append(
            f"Karena Anda mencari \"{k['kata']}\" ({k['dari_kategori']}), "
            f"dan kata ini ada pada {lokasi} parfum ini "
            f"- menyumbang {k['kontribusi_persen']}% dari skor kecocokan."
        )
    if not alasan:
        alasan.append("Kecocokan berasal dari kemiripan pola teks secara umum.")
    return alasan


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    res = supabase.table("perfumes").select("*").order("id").execute()
    df_db = pd.DataFrame(res.data)
    if df_db.empty:
        return {"status": "error", "message": "Database kosong"}

    # ========================================================
    # 1. CBF (CONTENT-BASED FILTERING) — kemiripan teks TF-IDF
    # ========================================================
    combined_query = f"{req.user_input} {req.gender} {req.waktu_pemakaian} {req.kegiatan} {req.wangi}"
    clean_query = clean_text(combined_query)

    # Fitur HARUS sama persis dengan scripts/train_model.py (pakai notes pyramid)
    for _c in ['top_notes', 'middle_notes', 'base_notes', 'main_accords']:
        if _c not in df_db.columns:
            df_db[_c] = ''
    df_db['features'] = (
        df_db['description'].fillna('') + " " +
        df_db['top_notes'].fillna('') + " " +
        df_db['middle_notes'].fillna('') + " " +
        df_db['base_notes'].fillna('') + " " +
        df_db['main_accords'].fillna('') + " " +
        df_db['waktu_penggunaan'].fillna('') + " " +
        df_db['occasion'].fillna('') + " " +
        df_db['gender_target'].fillna('')
    ).str.lower()

    mat_db = tfidf.transform(df_db['features'])
    query_vec = tfidf.transform([clean_query])
    raw_scores = cosine_similarity(query_vec, mat_db).flatten()
    df_db = df_db.reset_index(drop=True)
    df_db['display_score'] = raw_scores * 100

    feature_names = tfidf.get_feature_names_out()
    # Teks tiap kategori input (untuk menelusuri asal kata) — dropdown dulu, baru teks bebas
    field_texts = [
        ("Wangi", clean_text(req.wangi)),
        ("Waktu Pemakaian", clean_text(req.waktu_pemakaian)),
        ("Kegiatan", clean_text(req.kegiatan)),
        ("Gender", clean_text(req.gender)),
        ("Pencarian bebas Anda", clean_text(req.user_input)),
    ]

    # --- GENDER sebagai FILTER KERAS + prioritas urutan (beda dari TF-IDF yg soft) ---
    # Pilih "Pria" => tampilkan Pria + Unisex (Unisex pantas dipakai pria), buang Wanita,
    # dan parfum ber-gender PERSIS diprioritaskan di atas Unisex.
    df_db['gender_exact'] = 0
    g = clean_text(req.gender)
    df_cbf = df_db
    if g in ('pria', 'wanita', 'unisex'):
        gt = df_db['gender_target'].fillna('').str.lower()
        nm = df_db['nama_parfum'].fillna('').str.lower()
        if g == 'unisex':
            mask_g = gt.str.contains('unisex', na=False)
            df_db.loc[mask_g, 'gender_exact'] = 1
        else:
            df_db.loc[gt.str.contains(g, na=False), 'gender_exact'] = 1
            mask_g = gt.str.contains(g, na=False) | gt.str.contains('unisex', na=False)
            # ponytail: guard nama vs gender — label gender_target di DB banyak salah
            # (data marketplace), jadi buang parfum yg NAMANYA jelas lawan gender.
            # Ceiling: hanya menangkap kata gender generik (women/lady/pria/homme...),
            # bukan nama orang (mis. "Nagita Slavina"). Upgrade: bersihkan kolom DB.
            mask_g &= ~nm.str.contains(LAWAN_GENDER[g], regex=True, na=False)
        df_cbf = df_db[mask_g]

    # 'id' sbg tiebreaker tetap → saat skor seri, urutan selalu sama (hasil reproducible)
    sort_keys = ['gender_exact', 'display_score', 'id']
    sort_asc = [False, False, True]
    cbf_results = df_cbf[df_cbf['display_score'] >= 5.0].sort_values(by=sort_keys, ascending=sort_asc)

    # [FALLBACK] selalu beri minimal 5 hasil terdekat agar UX konsisten
    is_fallback = cbf_results.empty
    if is_fallback:
        cbf_results = df_cbf.sort_values(by=sort_keys, ascending=sort_asc).head(5)

    final_output = []
    top_cbf_id = None

    for pos, row in cbf_results.head(10).iterrows():
        if top_cbf_id is None:
            top_cbf_id = row['id']

        skor = round(row['display_score'], 1)
        parfum_fields = [
            ("nama produk", row.get('nama_parfum', '')),
            ("deskripsi", row.get('description', '')),
            ("notes atas", row.get('top_notes', '')),
            ("notes tengah", row.get('middle_notes', '')),
            ("notes dasar", row.get('base_notes', '')),
            ("aroma (accords)", row.get('main_accords', '')),
            ("occasion", row.get('occasion', '')),
            ("waktu pemakaian", row.get('waktu_penggunaan', '')),
        ]
        kata_cocok = bukti_cbf(query_vec, mat_db[pos], feature_names, field_texts, parfum_fields)
        daftar_alasan = alasan_cbf(kata_cocok, skor)

        final_output.append({
            "id": row['id'],
            "nama": row.get('nama_parfum', 'Unknown'),
            "brand": row.get('brand', '-'),
            "deskripsi": row.get('description', ''),
            # --- field produk lengkap agar frontend bisa tampilkan foto/harga/link/notes ---
            "harga": row.get('harga', 0),
            "url_foto": row.get('url_foto', ''),
            "url_produk": row.get('url_produk', ''),
            "notes": row.get('notes', ''),
            "top_notes": row.get('top_notes', ''),
            "middle_notes": row.get('middle_notes', ''),
            "base_notes": row.get('base_notes', ''),
            "main_accords": row.get('main_accords', ''),
            "occasion": row.get('occasion', ''),
            "waktu_penggunaan": row.get('waktu_penggunaan', ''),
            "gender_target": row.get('gender_target', ''),
            "tipe": "CBF",
            "match_score": f"{skor}%",
            "match_score_num": skor,
            "ringkasan": (
                f"Cocok {skor}% karena {len(kata_cocok)} kata kunci pencarian Anda "
                f"ditemukan di teks parfum ini."
                if kata_cocok else f"Kecocokan umum sebesar {skor}%."
            ),
            "alasan": daftar_alasan,
            "bukti": {"kata_cocok": kata_cocok},
            # tetap sediakan untuk kompatibilitas lama
            "debug_alasan": " ".join(daftar_alasan),
        })

    # ========================================================
    # 2. PENCARIAN BIASA (baseline query / substring) — sebagai PEMBANDING di web.
    # Bedanya dgn CBF: ini hanya cek "apakah kata ADA" (presence), TANPA bobot TF-IDF
    # & tanpa urutan relevansi. Cenderung banyak/umum. Untuk menunjukkan keunggulan AI.
    # ========================================================
    kata_kunci = [w for w in set(clean_query.split()) if len(w) > 1 and w in tfidf.vocabulary_]
    biasa_output = []
    total_biasa = 0
    if kata_kunci:
        df_db['n_match'] = df_db['features'].apply(lambda f: sum(1 for k in kata_kunci if k in f))
        biasa = df_db[df_db['n_match'] > 0].sort_values(by=['n_match', 'id'], ascending=[False, True])
        total_biasa = int(len(biasa))
        for _, row in biasa.head(24).iterrows():
            feat = str(row['features'])
            ditemukan = [k for k in kata_kunci if k in feat]
            biasa_output.append({
                "id": row['id'],
                "nama": row.get('nama_parfum', 'Unknown'),
                "brand": row.get('brand', '-'),
                "deskripsi": row.get('description', ''),
                "harga": row.get('harga', 0),
                "url_foto": row.get('url_foto', ''),
                "url_produk": row.get('url_produk', ''),
                "top_notes": row.get('top_notes', ''),
                "middle_notes": row.get('middle_notes', ''),
                "base_notes": row.get('base_notes', ''),
                "main_accords": row.get('main_accords', ''),
                "occasion": row.get('occasion', ''),
                "waktu_penggunaan": row.get('waktu_penggunaan', ''),
                "gender_target": row.get('gender_target', ''),
                "tipe": "QUERY",
                "kata_ditemukan": ditemukan,
                "jumlah_kata": len(ditemukan),
            })

    return {
        "status": "success",
        "is_fallback": is_fallback,
        "message": (
            "Tidak ada parfum yang benar-benar cocok dengan pencarian Anda. "
            "Berikut 5 parfum yang paling mendekati."
            if is_fallback else ""
        ),
        "kata_kunci": kata_kunci,
        "total_pencarian_biasa": total_biasa,
        "pencarian_biasa": biasa_output,
        "data": final_output,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
