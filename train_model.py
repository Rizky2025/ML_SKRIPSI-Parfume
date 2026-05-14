import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=== TAHAP 1: MEMBACA & MENYERAGAMKAN 4 DATASET ===")

# Wadah untuk menampung semua dataframe yang sudah diseragamkan
list_df = []

# 1. DATASET 1 (Tanpa Header)
try:
    # Tambahkan encoding='latin-1'
    df1_raw = pd.read_csv('dataset1.csv', header=None, encoding='latin-1', on_bad_lines='skip')
    df1 = pd.DataFrame({
        'nama': df1_raw[0],
        'brand': df1_raw[1],
        'deskripsi': df1_raw[2],
        'harga': 0, 
        'gender': df1_raw[8],
        'waktu_pemakaian': df1_raw[6],
        'acara': df1_raw[7],
        'notes': df1_raw[3],
        'link_pembelian': df1_raw[9],
        'rating_score': 0.0 
    })
    list_df.append(df1)
except Exception as e:
    print(f"Dataset 1 dilewati: {e}")

# 2. DATASET 2 (Ada Header)
try:
    # Tambahkan encoding='latin-1'
    df2_raw = pd.read_csv('dataset2.csv', encoding='latin-1', on_bad_lines='skip')
    df2 = pd.DataFrame({
        'nama': df2_raw['Name'],
        'brand': df2_raw['Brand'],
        'deskripsi': df2_raw['Description'],
        'harga': 0,
        'gender': 'N/A',
        'waktu_pemakaian': 'N/A',
        'acara': 'N/A',
        'notes': df2_raw['Notes'],
        'link_pembelian': 'N/A',
        'rating_score': 0.0
    })
    list_df.append(df2)
except Exception as e:
    print(f"Dataset 2 dilewati: {e}")

# 3. DATASET 3 (Marketplace)
try:
    # Tambahkan sep=None dan engine='python' agar otomatis mendeteksi koma (,) atau titik koma (;)
    df3_raw = pd.read_csv('dataset3.csv', encoding='latin-1', on_bad_lines='skip', sep=None, engine='python')
    df3_raw.columns = df3_raw.columns.str.strip() # Membersihkan spasi gaib di awal/akhir nama kolom
    
    notes_3 = df3_raw['top_notes'].fillna('') + " " + df3_raw['middle_notes'].fillna('') + " " + df3_raw['base_notes'].fillna('')
    
    rating_val_3 = pd.to_numeric(df3_raw['rating_value'], errors='coerce').fillna(0)
    rating_cnt_3 = pd.to_numeric(df3_raw['rating_count'], errors='coerce').fillna(0)
    cf_score_3 = rating_val_3 * np.log1p(rating_cnt_3)

    df3 = pd.DataFrame({
        'nama': df3_raw['nama_parfum'],
        'brand': df3_raw['brand'],
        'deskripsi': df3_raw['description'],
        'harga': pd.to_numeric(df3_raw['harga'], errors='coerce').fillna(0),
        'gender': df3_raw['gender_target'],
        'waktu_pemakaian': df3_raw['waktu_penggunaan'],
        'acara': df3_raw['occasion'],
        'notes': notes_3,
        'link_pembelian': df3_raw['url_produk'],
        'rating_score': cf_score_3
    })
    list_df.append(df3)
except Exception as e:
    print(f"Dataset 3 dilewati: {e}")

# 4. DATASET 4 (Lengkap)
try:
    df4_raw = pd.read_csv('dataset4.csv', encoding='latin-1', on_bad_lines='skip', sep=None, engine='python')
    df4_raw.columns = df4_raw.columns.str.strip() 
    
    notes_4 = df4_raw['top_notes'].fillna('') + " " + df4_raw['middle_notes'].fillna('') + " " + df4_raw['base_notes'].fillna('')
    
    rating_val_4 = pd.to_numeric(df4_raw['rating_value'], errors='coerce').fillna(0)
    rating_cnt_4 = pd.to_numeric(df4_raw['rating_count'], errors='coerce').fillna(0)
    cf_score_4 = rating_val_4 * np.log1p(rating_cnt_4)

    df4 = pd.DataFrame({
        'nama': df4_raw['nama_parfum'],
        'brand': df4_raw['brand'],
        'deskripsi': df4_raw['description'],
        'harga': pd.to_numeric(df4_raw['harga'], errors='coerce').fillna(0),
        'gender': df4_raw['gender_target'],
        'waktu_pemakaian': df4_raw['waktu_penggunaan'],
        'acara': df4_raw['occasion'],
        'notes': notes_4,
        'link_pembelian': df4_raw['url_produk'],
        'rating_score': cf_score_4
    })
    list_df.append(df4)
except Exception as e:
    print(f"Dataset 4 dilewati: {e}")

print("=== TAHAP 2: CLEANING & PENGGABUNGAN DATA ===")
# Gabungkan semua data yang berhasil diproses
df_final = pd.concat(list_df, ignore_index=True)

# Hapus duplikat berdasarkan nama parfum dan isi nilai kosong
df_final = df_final.drop_duplicates(subset=['nama']).dropna(subset=['nama']).reset_index(drop=True)
df_final = df_final.fillna('N/A')

# Fungsi pembersih teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# MENGGABUNGKAN FITUR UNTUK CBF (Notes + Deskripsi + Waktu + Acara + Gender)
# Pendekatan ini membuat model mampu merekomendasikan berdasarkan "vibes" keseluruhan
df_final['cbf_features'] = (
    df_final['notes'].astype(str) + " " + 
    df_final['deskripsi'].astype(str) + " " + 
    df_final['waktu_pemakaian'].astype(str) + " " + 
    df_final['acara'].astype(str) + " " + 
    df_final['gender'].astype(str)
)
df_final['cbf_features_clean'] = df_final['cbf_features'].apply(clean_text)

print(f"Total parfum siap ditraining: {len(df_final)} data")

print("=== TAHAP 3: TRAINING MODEL (CONTENT-BASED) ===")
# Mengubah teks fitur gabungan menjadi vektor angka (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.8)
mat_features = tfidf.fit_transform(df_final['cbf_features_clean'])

# Menghitung kemiripan (Cosine Similarity)
cbf_similarity = cosine_similarity(mat_features, mat_features)

print("=== TAHAP 4: EXPORT KE MAKSIMAL 3 FILE .PKL ===")
# Hanya 3 file .pkl untuk seluruh sistem
with open('data_parfum_bersih.pkl', 'wb') as f:
    pickle.dump(df_final[['nama', 'brand', 'deskripsi', 'harga', 'gender', 'waktu_pemakaian', 'acara', 'notes', 'link_pembelian', 'rating_score']], f)

with open('matriks_cbf.pkl', 'wb') as f:
    pickle.dump(cbf_similarity, f)

with open('model_tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("✅ SELESAI! Model dan data berhasil disimpan ke dalam 3 file .pkl.")

# =====================================================================
# CONTOH FUNGSI PENGGUNAAN HYBRID (TIDAK DIEKSEKUSI SAAT TRAINING)
# =====================================================================
def get_hybrid_recommendation(perfume_idx, df, sim_matrix, top_n=12):
    """
    Sistem Hybrid yang di-improve:
    1. Cari kecocokan CBF berdasarkan fitur gabungan.
    2. Jika kurang dari target (top_n), ambil sisa kuota menggunakan skor CF (rating_score)
       yang sudah dihitung berdasarkan popularitas & ulasan.
    """
    sim_scores = list(enumerate(sim_matrix[perfume_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    cbf_indices = []
    threshold = 0.15 # Batas wajar kemiripan
    
    for i, score in sim_scores[1:]:
        if score > threshold:
            cbf_indices.append(i)
        if len(cbf_indices) == top_n:
            break
            
    # Jika CBF tidak memenuhi kuota 12 parfum
    kekurangan = top_n - len(cbf_indices)
    
    if kekurangan > 0:
        exclude_indices = cbf_indices + [perfume_idx]
        # Menggunakan skor CF (Rating * Log(Count)) sebagai Fallback
        cf_candidates = df.drop(exclude_indices).sort_values(by='rating_score', ascending=False)
        cf_indices = cf_candidates.head(kekurangan).index.tolist()
        cbf_indices.extend(cf_indices)
        
    return df.iloc[cbf_indices]