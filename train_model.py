import pandas as pd
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=== 1. TRAINING CONTENT-BASED FILTERING (CBF) ===")
# MENGGUNAKAN JSON YANG SUDAH BERSIH DAN SEMPURNA
try:
    with open('perfumes_rows.json', 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    df_parfum = pd.DataFrame(data_json)
    print(f"✅ Berhasil memuat {len(df_parfum)} data parfum dari JSON.")
except FileNotFoundError:
    print("❌ ERROR: File perfumes_rows.json tidak ditemukan! Pastikan namanya benar.")
    exit()

# Ganti nilai yang kosong (NaN/Null) menjadi string kosong agar tidak error
df_parfum = df_parfum.fillna('')

# Gabungkan fitur untuk CBF
df_parfum['cbf_features'] = (
    df_parfum['description'].astype(str) + " " + 
    df_parfum['notes'].astype(str) + " " + 
    df_parfum['waktu_penggunaan'].astype(str) + " " + 
    df_parfum['occasion'].astype(str) + " " + 
    df_parfum['gender_target'].astype(str) + " " +
    df_parfum['main_accords'].astype(str)
).str.lower()

tfidf = TfidfVectorizer(stop_words=None, min_df=1)
mat_features = tfidf.fit_transform(df_parfum['cbf_features'])

with open('model_tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✅ Model CBF (TF-IDF) berhasil disimpan!")

print("\n=== 2. TRAINING COLLABORATIVE FILTERING (CF) ===")
try:
    # Menggunakan dummy interaksi user
    df_interaksi = pd.read_csv('user_interactions_v2.csv')
    
    # Buat Pivot Table: Baris=Parfum_id, Kolom=User_id, Value=Rating
    cf_matrix = df_interaksi.pivot_table(index='parfum_id', columns='user_id', values='rating').fillna(0)
    
    # Hitung kemiripan antar ITEM (Parfum) berdasarkan pola rating
    item_similarity = cosine_similarity(cf_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=cf_matrix.index, columns=cf_matrix.index)
    
    with open('model_cf_item.pkl', 'wb') as f:
        pickle.dump(item_similarity_df, f)
    print("✅ Model CF (Item-Based) dari data dummy berhasil disimpan!")

except FileNotFoundError:
    print("❌ Peringatan: File user_interaction_v2.csv tidak ditemukan. CF Matrix tidak dibuat.")

print("\n=== TAHAP TRAINING SELESAI ===")