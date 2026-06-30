"""
======================================================================
EVALUASI MODEL REKOMENDASI (Batch 3)
======================================================================
Script ini mengukur akurasi model secara OFFLINE (tanpa perlu server/
Supabase) untuk bahan bab Pengujian/Hasil skripsi.

Metrik yang dihitung:
  - Precision@k : dari k parfum yang direkomendasikan, berapa % yang benar-benar relevan.
  - Recall@k    : dari semua parfum relevan, berapa % yang berhasil masuk top-k.
  - F1@k        : rata-rata harmonik Precision & Recall.

Dua bagian:
  A. EVALUASI CF  -> metode train/test split (held-out). Ground-truth = rating >= 4.
                     Dibandingkan dengan BASELINE POPULARITAS.
  B. EVALUASI CBF -> query dibangun dari preferensi user. Ground-truth = parfum
                     cocok pada >= 2 dari 4 dimensi preferensi. Dibandingkan
                     dengan BASELINE ACAK.

Cara jalan:  python scripts/evaluate_model.py
======================================================================
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ----------------------------------------------------------------------
# PATH & KONFIGURASI
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PARFUM_JSON = BASE_DIR / "dataset" / "parfum" / "perfumes_rows.json"
RATING_CSV = BASE_DIR / "dataset" / "interaksi" / "dataset_rating_variatif.csv"
PROFIL_CSV = BASE_DIR / "dataset" / "interaksi" / "profil_user_dummy.csv"
HASIL_MD = BASE_DIR.parent / "DOKUMENTASI" / "03_HASIL_EVALUASI.md"

SEED = 42
K_LIST = [5, 10]            # nilai k yang diuji
RELEVAN_RATING = 4          # rating >= ini dianggap "user suka" (relevan) untuk CF
MIN_INTERAKSI = 5           # user minimal punya sekian rating agar ikut dievaluasi CF
TEST_RATIO = 0.2            # porsi rating tiap user yang disembunyikan untuk diuji

random.seed(SEED)
np.random.seed(SEED)


# ======================================================================
# UTIL METRIK
# ======================================================================
def precision_recall_at_k(recommended, relevant, k):
    """Hitung precision & recall untuk satu user."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0, 0.0
    hits = len(set(rec_k) & set(relevant))
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def f1(p, r):
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


# ======================================================================
# BAGIAN A — EVALUASI COLLABORATIVE FILTERING (CF)
# ======================================================================
def evaluasi_cf():
    print("\n" + "=" * 60)
    print("BAGIAN A — EVALUASI CF (train/test split, held-out)")
    print("=" * 60)

    df = pd.read_csv(RATING_CSV)[["user_id", "parfum_id", "rating"]].dropna()

    # --- Split per user: sebagian rating disembunyikan jadi data uji ---
    train_rows, test_rows = [], []
    for uid, grup in df.groupby("user_id"):
        grup = grup.sample(frac=1.0, random_state=SEED)  # acak
        if len(grup) < MIN_INTERAKSI:
            continue
        n_test = max(1, int(round(len(grup) * TEST_RATIO)))
        test_rows.append(grup.iloc[:n_test])
        train_rows.append(grup.iloc[n_test:])

    train = pd.concat(train_rows)
    test = pd.concat(test_rows)
    print(f"User dievaluasi : {train.user_id.nunique()}")
    print(f"Interaksi train : {len(train)} | Interaksi test : {len(test)}")

    # --- Bangun similarity antar item HANYA dari data train (hindari kebocoran) ---
    pivot = train.pivot_table(index="parfum_id", columns="user_id", values="rating").fillna(0)
    sim = pd.DataFrame(
        cosine_similarity(pivot), index=pivot.index, columns=pivot.index
    )

    # --- Popularitas (untuk baseline): jumlah rating tinggi per item di train ---
    populer = (
        train[train.rating >= RELEVAN_RATING]["parfum_id"]
        .value_counts()
        .index.tolist()
    )

    hasil = {k: {"cf_p": [], "cf_r": [], "pop_p": [], "pop_r": []} for k in K_LIST}

    test_grup = test.groupby("user_id")
    for uid, train_u in train.groupby("user_id"):
        if uid not in test_grup.groups:
            continue
        test_u = test_grup.get_group(uid)
        relevan = test_u[test_u.rating >= RELEVAN_RATING]["parfum_id"].tolist()
        if not relevan:
            continue  # tidak ada ground-truth positif -> lewati

        sudah_dirating = set(train_u.parfum_id)

        # ---- Skor CF: jumlah similarity ke item yang user-rating tinggi ----
        item_disukai = train_u[train_u.rating >= RELEVAN_RATING]
        skor = pd.Series(dtype=float)
        for _, row in item_disukai.iterrows():
            pid = row.parfum_id
            if pid in sim.index:
                skor = skor.add(sim[pid] * row.rating, fill_value=0)
        # buang item yang sudah dirating user di train
        skor = skor.drop(labels=[p for p in sudah_dirating if p in skor.index], errors="ignore")
        rekom_cf = skor.sort_values(ascending=False).index.tolist()

        # ---- Baseline popularitas: item terpopuler yang belum dirating ----
        rekom_pop = [p for p in populer if p not in sudah_dirating]

        for k in K_LIST:
            p, r = precision_recall_at_k(rekom_cf, relevan, k)
            hasil[k]["cf_p"].append(p)
            hasil[k]["cf_r"].append(r)
            p2, r2 = precision_recall_at_k(rekom_pop, relevan, k)
            hasil[k]["pop_p"].append(p2)
            hasil[k]["pop_r"].append(r2)

    baris = []
    for k in K_LIST:
        cf_p, cf_r = np.mean(hasil[k]["cf_p"]), np.mean(hasil[k]["cf_r"])
        pop_p, pop_r = np.mean(hasil[k]["pop_p"]), np.mean(hasil[k]["pop_r"])
        print(f"\n--- k = {k} ---")
        print(f"CF        : Precision@{k}={cf_p:.3f}  Recall@{k}={cf_r:.3f}  F1@{k}={f1(cf_p, cf_r):.3f}")
        print(f"Baseline  : Precision@{k}={pop_p:.3f}  Recall@{k}={pop_r:.3f}  F1@{k}={f1(pop_p, pop_r):.3f}  (popularitas)")
        baris.append((k, cf_p, cf_r, f1(cf_p, cf_r), pop_p, pop_r, f1(pop_p, pop_r)))
    return baris


# ======================================================================
# BAGIAN B — EVALUASI CONTENT-BASED FILTERING (CBF)
# ======================================================================
def bersih(teks):
    import re
    return re.sub(r"[^a-z0-9\s]", " ", str(teks).lower()).strip()


def evaluasi_cbf():
    print("\n" + "=" * 60)
    print("BAGIAN B — EVALUASI CBF (kecocokan atribut preferensi)")
    print("=" * 60)

    df = pd.DataFrame(json.load(open(PARFUM_JSON, encoding="utf-8"))).fillna("")

    # Fitur teks SAMA PERSIS dengan train_model.py & main.py
    df["features"] = (
        df["description"].astype(str) + " " + df["notes"].astype(str) + " " +
        df["waktu_penggunaan"].astype(str) + " " + df["occasion"].astype(str) + " " +
        df["gender_target"].astype(str) + " " + df["main_accords"].astype(str)
    ).str.lower()

    from _stopwords import STOPWORDS
    tfidf = TfidfVectorizer(stop_words=STOPWORDS, min_df=1)
    mat = tfidf.fit_transform(df["features"])

    # Profil preferensi user
    prof = pd.read_csv(PROFIL_CSV).drop_duplicates(
        subset=["pref_gender", "pref_aroma", "pref_waktu", "pref_kegiatan"]
    )
    print(f"Kombinasi preferensi unik diuji : {len(prof)}")

    def relevan_mask(g, a, w, keg):
        """Parfum relevan jika cocok >= 2 dari 4 dimensi (logika sama dgn generate_dataset)."""
        skor = (
            df["gender_target"].str.contains(g, case=False, na=False).astype(int) +
            df["wangi_vibe"].str.contains(a, case=False, na=False).astype(int) +
            df["waktu_penggunaan"].str.contains(w, case=False, na=False).astype(int) +
            df["occasion"].str.contains(keg, case=False, na=False).astype(int)
        )
        return skor >= 2

    hasil = {k: {"cbf_p": [], "rand_p": []} for k in K_LIST}
    semua_idx = list(range(len(df)))

    for _, p in prof.iterrows():
        g, a, w, keg = p.pref_gender, p.pref_aroma, p.pref_waktu, p.pref_kegiatan
        relevan_idx = set(df.index[relevan_mask(g, a, w, keg)].tolist())
        if not relevan_idx:
            continue

        query = bersih(f"{a} {g} {w} {keg}")
        qvec = tfidf.transform([query])
        skor = cosine_similarity(qvec, mat).flatten()
        rank = np.argsort(skor)[::-1].tolist()

        for k in K_LIST:
            top = rank[:k]
            hits = len(set(top) & relevan_idx)
            hasil[k]["cbf_p"].append(hits / k)
            # baseline acak: ambil k indeks acak
            rnd = random.sample(semua_idx, k)
            hasil[k]["rand_p"].append(len(set(rnd) & relevan_idx) / k)

    baris = []
    for k in K_LIST:
        cbf_p = np.mean(hasil[k]["cbf_p"])
        rand_p = np.mean(hasil[k]["rand_p"])
        print(f"\n--- k = {k} ---")
        print(f"CBF       : Precision@{k}={cbf_p:.3f}")
        print(f"Baseline  : Precision@{k}={rand_p:.3f}  (acak)")
        baris.append((k, cbf_p, rand_p))
    return baris


# ======================================================================
# TULIS HASIL KE MARKDOWN (untuk lampiran skripsi)
# ======================================================================
def tulis_laporan(cf_baris, cbf_baris):
    L = []
    L.append("# 📊 Hasil Evaluasi Model (Batch 3)\n")
    L.append("> Dihasilkan otomatis oleh `ML_SKRIPSI-Parfume/scripts/evaluate_model.py`.")
    L.append(f"> Konfigurasi: seed={SEED}, relevan jika rating ≥ {RELEVAN_RATING}, "
             f"test split = {int(TEST_RATIO*100)}%.\n")

    L.append("## A. Collaborative Filtering (CF) vs Baseline Popularitas\n")
    L.append("| k | Precision@k | Recall@k | F1@k | Baseline P@k | Baseline R@k | Baseline F1@k |")
    L.append("|---|-------------|----------|------|--------------|--------------|---------------|")
    for k, cp, cr, cf1, pp, pr, pf1 in cf_baris:
        L.append(f"| {k} | {cp:.3f} | {cr:.3f} | {cf1:.3f} | {pp:.3f} | {pr:.3f} | {pf1:.3f} |")

    L.append("\n## B. Content-Based Filtering (CBF) vs Baseline Acak\n")
    L.append("| k | Precision@k (CBF) | Precision@k (Acak) |")
    L.append("|---|-------------------|--------------------|")
    for k, cbf_p, rand_p in cbf_baris:
        L.append(f"| {k} | {cbf_p:.3f} | {rand_p:.3f} |")

    L.append("\n## Cara membaca")
    L.append("- **CF lebih tinggi dari baseline popularitas** = model belajar pola selera, "
             "bukan sekadar merekomendasikan yang populer.")
    L.append("- **CBF lebih tinggi dari baseline acak** = pencocokan teks TF-IDF benar-benar "
             "relevan dengan preferensi, bukan kebetulan.")
    L.append("\n> Catatan keterbatasan: kolom `wangi_vibe` (aroma) mayoritas kosong/`N/A`, "
             "sehingga dimensi aroma kurang terukur. Ini batasan data, bukan algoritma.")

    HASIL_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✅ Laporan disimpan ke: {HASIL_MD}")


if __name__ == "__main__":
    cf_baris = evaluasi_cf()
    cbf_baris = evaluasi_cbf()
    tulis_laporan(cf_baris, cbf_baris)
    print("\n=== EVALUASI SELESAI ===")
