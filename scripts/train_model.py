import sys
import json
from pathlib import Path

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from _stopwords import STOPWORDS

# Paksa output konsol ke UTF-8 agar emoji (✅/❌) tidak crash di terminal Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- PATH DASAR (robust, tidak bergantung dari mana script dijalankan) ---
# scripts/ berada satu level di bawah root ML, jadi root = parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PARFUM_JSON = BASE_DIR / "dataset" / "parfum" / "perfumes_rows.json"
DATA_INTERAKSI_CF = BASE_DIR / "dataset" / "interaksi" / "dataset_rating_variatif.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("=== 1. TRAINING CONTENT-BASED FILTERING (CBF) ===")
# MENGGUNAKAN JSON YANG SUDAH BERSIH DAN SEMPURNA
try:
    with open(DATA_PARFUM_JSON, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    df_parfum = pd.DataFrame(data_json)
    print(f"✅ Berhasil memuat {len(df_parfum)} data parfum dari JSON.")
except FileNotFoundError:
    print(f"❌ ERROR: File tidak ditemukan: {DATA_PARFUM_JSON}")
    exit()

# Ganti nilai yang kosong (NaN/Null) menjadi string kosong agar tidak error
df_parfum = df_parfum.fillna('')

# Gabungkan fitur untuk CBF.
# [Batch 10] Tambah top/middle/base_notes (terisi 100% & berisi nama aroma asli) —
# sebelumnya kolom 'notes' dipakai tapi 92% kosong, jadi data aroma terkaya tak terpakai.
for _c in ['top_notes', 'middle_notes', 'base_notes']:
    if _c not in df_parfum.columns:
        df_parfum[_c] = ''
df_parfum['cbf_features'] = (
    df_parfum['description'].astype(str) + " " +
    df_parfum['top_notes'].astype(str) + " " +
    df_parfum['middle_notes'].astype(str) + " " +
    df_parfum['base_notes'].astype(str) + " " +
    df_parfum['main_accords'].astype(str) + " " +
    df_parfum['waktu_penggunaan'].astype(str) + " " +
    df_parfum['occasion'].astype(str) + " " +
    df_parfum['gender_target'].astype(str)
).str.lower()

# stop_words: buang kata sambung ID/EN agar skor berbasis kata bermakna (bukan "yang"/"untuk")
tfidf = TfidfVectorizer(stop_words=STOPWORDS, min_df=1)
mat_features = tfidf.fit_transform(df_parfum['cbf_features'])

with open(MODELS_DIR / 'model_tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✅ Model CBF (TF-IDF) berhasil disimpan!")

print("\n=== 2. TRAINING COLLABORATIVE FILTERING (CF) ===")
try:
    # [BATCH 2] Sumber CF diganti ke dataset_rating_variatif.csv.
    # Alasan: ratingnya berkorelasi dengan preferensi user (lebih realistis),
    # menggantikan user_interactions_v2.csv yang ratingnya murni acak.
    df_interaksi = pd.read_csv(DATA_INTERAKSI_CF)
    print(f"✅ Memuat {len(df_interaksi)} baris interaksi dari {DATA_INTERAKSI_CF.name}.")

    # Buat Pivot Table: Baris=Parfum_id, Kolom=User_id, Value=Rating
    cf_matrix = df_interaksi.pivot_table(index='parfum_id', columns='user_id', values='rating').fillna(0)

    # Hitung kemiripan antar ITEM (Parfum) berdasarkan pola rating
    item_similarity = cosine_similarity(cf_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=cf_matrix.index, columns=cf_matrix.index)

    with open(MODELS_DIR / 'model_cf_item.pkl', 'wb') as f:
        pickle.dump(item_similarity_df, f)
    print("✅ Model CF (Item-Based) berhasil disimpan!")

    # ------------------------------------------------------------------
    # [BATCH 5] STATISTIK CF PER PARFUM (untuk "bukti"/penjelasan di web)
    # Dihitung sekali saat training agar runtime (main.py) tetap ringan.
    # ------------------------------------------------------------------
    print("\n=== 3. MENGHITUNG STATISTIK BUKTI CF ===")

    def modus_teratas(series, n=1):
        """Ambil nilai paling sering muncul + jumlahnya (abaikan kosong)."""
        s = series.dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "n/a")]
        if s.empty:
            return []
        vc = s.value_counts()
        return [(idx, int(cnt)) for idx, cnt in vc.head(n).items()]

    cf_stats = {}
    for pid, grup in df_interaksi.groupby("parfum_id"):
        suka = grup[grup["rating"] >= 4]                # "disukai" = rating >= 4
        n_total = int(len(grup))
        n_suka = int(len(suka))
        # konteks dari orang yang MENYUKAI parfum ini
        kolom_occ = "user_occasion_pref" if "user_occasion_pref" in grup.columns else None
        kolom_time = "user_time_pref" if "user_time_pref" in grup.columns else None
        kolom_aroma = "user_aroma_pref" if "user_aroma_pref" in grup.columns else None
        cf_stats[pid] = {
            "jumlah_penilai": n_total,
            "jumlah_suka": n_suka,
            "persen_suka": round(100 * n_suka / n_total, 1) if n_total else 0.0,
            "rata_rata_rating": round(float(grup["rating"].mean()), 2),
            "occasion_populer": modus_teratas(suka[kolom_occ], 2) if kolom_occ else [],
            "waktu_populer": modus_teratas(suka[kolom_time], 2) if kolom_time else [],
            "aroma_populer": modus_teratas(suka[kolom_aroma], 1) if kolom_aroma else [],
        }

    with open(MODELS_DIR / 'cf_stats.pkl', 'wb') as f:
        pickle.dump(cf_stats, f)
    print(f"✅ Statistik bukti CF untuk {len(cf_stats)} parfum disimpan (cf_stats.pkl)!")

except FileNotFoundError:
    print(f"❌ Peringatan: File tidak ditemukan: {DATA_INTERAKSI_CF}. CF Matrix tidak dibuat.")

print("\n=== TAHAP TRAINING SELESAI ===")
