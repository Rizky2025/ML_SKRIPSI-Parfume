from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# 1. LOAD MODEL ML (Dual Matrix)
# ==========================================
print("⏳ Memuat model Machine Learning...")
try:
    with open('tfidf_ingredient.pkl', 'rb') as f: tfidf_ingr = pickle.load(f)
    with open('tfidf_desc.pkl', 'rb') as f: tfidf_desc = pickle.load(f)
    with open('mat_ingredient.pkl', 'rb') as f: mat_ingr = pickle.load(f)
    with open('mat_desc.pkl', 'rb') as f: mat_desc = pickle.load(f)
    with open('df_perfume.pkl', 'rb') as f: df = pickle.load(f)
    with open('model_config.json', 'r') as f: config = json.load(f)

    ALPHA = config.get('alpha', 0.7)
    print(f"✅ Model ML berhasil dimuat! (Alpha: {ALPHA}, Total: {len(df)} parfum)")
except FileNotFoundError as e:
    print(f"❌ ERROR: File model tidak ditemukan — {e}")
    print("   Jalankan main.py dulu untuk generate file .pkl")
    exit()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==========================================
# 2. FUNGSI REKOMENDASI UTAMA
# ==========================================
def recommend_from_input(user_input, top_n=10):
    user_input_clean = clean_text(user_input)

    vec_ingr = tfidf_ingr.transform([user_input_clean])
    vec_desc = tfidf_desc.transform([user_input_clean])

    sim_ingr  = cosine_similarity(vec_ingr, mat_ingr).flatten()
    sim_desc  = cosine_similarity(vec_desc, mat_desc).flatten()
    sim_final = (ALPHA * sim_ingr) + ((1 - ALPHA) * sim_desc)

# Urutkan dari skor tertinggi
    scores = sorted(enumerate(sim_final), key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if s > 0]

    result_data = []
    seen_names = set()
    for index_parfum, skor in scores:
        nama = str(df['Name'].iloc[int(index_parfum)])
        if nama not in seen_names:
            # Mengambil deskripsi, tangani jika nilainya kosong/NaN
            deskripsi = str(df['Description'].iloc[int(index_parfum)])
            if deskripsi.lower() == 'nan':
                deskripsi = "Deskripsi belum tersedia."

            result_data.append({
                "name": nama,
                "score": float(skor),
                "description": deskripsi # Kirim deskripsi ke Node.js
            })
            seen_names.add(nama)
        if len(result_data) == top_n:
            break

    return result_data

# ==========================================
# 3. API ENDPOINT
# ==========================================
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get("input")

    # 1. AMBIL NILAI DARI NODE.JS DULU (Baris ini yang sepertinya hilang)
    top_n_request = data.get("top_n", 50)
    top_n = int(top_n_request)
    
    # 2. BARU DIBATASI ANTARA 5 SAMPAI 100
    top_n = max(5, min(top_n, 100))

    if not user_input:
        return jsonify({"error": "Input tidak boleh kosong"}), 400

    results = recommend_from_input(user_input, top_n=top_n)

    print(f"\n[INPUT]  : {user_input}")
    print(f"[TOP_N]  : {top_n}")
    print(f"[OUTPUT] : {results}")

    return jsonify({
        "input": user_input,
        "top_n_requested": top_n,
        "recommendations": results
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "total_perfumes": len(df), "alpha": ALPHA})

if __name__ == '__main__':
    app.run(debug=True)