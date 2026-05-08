import pandas as pd
import numpy as np
import pickle
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("⏳ 1. Memuat dataset...")
df = pd.read_csv("final_perfume_data.csv", encoding='latin-1')

# Preprocessing Dasar
df['Notes'] = df['Notes'].fillna('').str.lower()
df['Description'] = df['Description'].fillna('').str.lower()
df['Brand'] = df['Brand'].fillna('Unknown')

# =========================
# CUSTOM STOP WORDS PARFUM
# =========================
# Kata-kata industri parfum yang muncul di hampir semua data
# sehingga tidak membantu membedakan satu parfum dengan yang lain
PERFUME_STOP_WORDS = {
    # Istilah industri umum
    'eau', 'de', 'parfum', 'toilette', 'cologne', 'fragrance', 'scent',
    'perfume', 'notes', 'note', 'accord', 'accords', 'blend', 'aroma',
    'aromatic', 'olfactory', 'nose', 'house', 'maison', 'extrait',
    'absolu', 'absolute', 'concentree', 'intense', 'pure',
    # Kata deskriptif umum yang tidak membedakan aroma
    'heart', 'base', 'top', 'middle', 'dry', 'down', 'drydown',
    'opening', 'trail', 'sillage', 'longevity', 'projection',
    'skin', 'wear', 'wearing', 'worn', 'smell', 'smells', 'smelling',
    'like', 'subtle', 'hint', 'touch', 'bit', 'slightly', 'strong',
    'light', 'heavy', 'fresh', 'warm', 'cool', 'cold', 'hot',
    # Kata umum bahasa Inggris tidak relevan
    'very', 'quite', 'rather', 'little', 'great', 'good', 'nice',
    'beautiful', 'elegant', 'luxurious', 'rich', 'deep',
    'unique', 'special', 'interesting', 'wonderful', 'amazing',
    'perfect', 'best', 'well', 'known', 'also', 'even', 'just',
    'one', 'two', 'three', 'first', 'second', 'third', 'new',
    'make', 'made', 'making', 'use', 'used', 'using',
    # Kata marketing
    'inspired', 'inspired by', 'collection', 'limited', 'edition',
    'signature', 'iconic', 'classic', 'modern', 'contemporary',
    'bottle', 'packaging', 'design', 'brand', 'launch', 'release',
    'created', 'crafted', 'designed', 'composed', 'blended',
    'perfumer', 'perfumers', 'creator',
}

# Gabungkan dengan english stop words bawaan sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
all_stop_words = list(ENGLISH_STOP_WORDS.union(PERFUME_STOP_WORDS))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['Notes_clean'] = df['Notes'].apply(clean_text)
df['Desc_clean'] = df['Description'].apply(clean_text)

print("⏳ 2. Membangun Matrix 1 (Notes / Ingredients)...")
tfidf_ingr = TfidfVectorizer(
    stop_words=all_stop_words,  # FIX: pakai custom stop words parfum
    min_df=2,
    max_df=0.60,                # FIX: dari 0.90 → 0.60, buang kata yang muncul >60% data
    ngram_range=(1, 2),
    sublinear_tf=True
)
mat_ingr = tfidf_ingr.fit_transform(df['Notes_clean'])
cs_ingr = cosine_similarity(mat_ingr, mat_ingr)
print(f"   Vocab Notes: {len(tfidf_ingr.vocabulary_)} kata")

print("⏳ 3. Membangun Matrix 2 (Context / Description)...")
tfidf_desc = TfidfVectorizer(
    stop_words=all_stop_words,  # FIX: pakai custom stop words parfum
    min_df=2,
    max_df=0.60,                # FIX: dari 0.90 → 0.60
    ngram_range=(1, 2),
    sublinear_tf=True
)
mat_desc = tfidf_desc.fit_transform(df['Desc_clean'])
cs_desc = cosine_similarity(mat_desc, mat_desc)
print(f"   Vocab Description: {len(tfidf_desc.vocabulary_)} kata")

print("⏳ 4. Blending kedua matrix (NLP Approach)...")
ALPHA = 0.7  # 70% Notes (komposisi bahan), 30% Description (vibe/konteks)
cosine_sim = (ALPHA * cs_ingr) + ((1 - ALPHA) * cs_desc)

# =========================
# EVALUASI & GENERATE REPORT
# =========================
print("⏳ 5. Evaluasi model NLP...")
all_similarity_scores = []
top_n = 5

for idx in range(len(df)):
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    all_similarity_scores.extend([s for _, s in sim_scores])

avg_similarity    = np.mean(all_similarity_scores)
median_similarity = np.median(all_similarity_scores)
max_similarity    = np.max(all_similarity_scores)
min_similarity    = np.min(all_similarity_scores)
pct_above_05      = np.mean(np.array(all_similarity_scores) > 0.5) * 100

if avg_similarity > 0.85:
    kesehatan = "⚠️ TERLALU TINGGI (Mungkin overfitting)"
elif avg_similarity >= 0.70:
    kesehatan = "✅ EXCELLENT — TARGET TERCAPAI"
elif avg_similarity >= 0.40:
    kesehatan = "✅ SANGAT BAGUS (Production Ready)"
elif avg_similarity >= 0.15:
    kesehatan = "🟡 CUKUP BAIK"
else:
    kesehatan = "❌ UNDERFITTING"

# Ambil 3 sampel acak untuk contoh rekomendasi
test_cases = list(np.random.choice(len(df), min(3, len(df)), replace=False))
sample_results = []
for idx in test_cases:
    if idx < len(df):
        ss = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:4]
        recs = [(df['Name'].iloc[i], df['Notes'].iloc[i][:55], round(s, 4)) for i, s in ss]
        sample_results.append((df['Name'].iloc[idx], df['Notes'].iloc[idx][:55], recs))

report_text = (
    "===========================================================\n"
    "      DIAGNOSA & EVALUASI MODEL NLP (DUAL MATRIX TF-IDF)   \n"
    "===========================================================\n"
    f"Total Data Parfum    : {len(df)} item\n"
    f"Arsitektur           : Dual NLP Matrix (Notes + Deskripsi)\n"
    f"  - Matrix 1         : TF-IDF Notes (Spesifik Bahan Baku)\n"
    f"  - Matrix 2         : TF-IDF Description (Konteks, Vibe, Acara)\n"
    f"  - Blend Formula    : {ALPHA} x M_notes + {round(1-ALPHA,1)} x M_desc\n"
    f"  - Custom Stop Words: {len(PERFUME_STOP_WORDS)} kata industri parfum dibuang\n"
    f"  - max_df           : 0.60 (buang kata yang muncul >60% data)\n"
    f"Vocab Notes          : {len(tfidf_ingr.vocabulary_)} kata\n"
    f"Vocab Description    : {len(tfidf_desc.vocabulary_)} kata\n"
    f"Evaluasi Berdasarkan : Top-{top_n} Rekomendasi Terdekat\n\n"

    "--- METRIK KEMIRIPAN (0.0 s/d 1.0) ---\n"
    f"Rata-rata Kemiripan  : {avg_similarity:.4f}\n"
    f"Median Kemiripan     : {median_similarity:.4f}\n"
    f"Kemiripan Tertinggi  : {max_similarity:.4f}\n"
    f"Kemiripan Terendah   : {min_similarity:.4f}\n"
    f"% Skor di atas 0.5   : {pct_above_05:.1f}%\n\n"

    "--- HASIL DIAGNOSA KESEHATAN MODEL ---\n"
    f"Status               : {kesehatan}\n\n"

    "--- CONTOH REKOMENDASI (verifikasi kualitas) ---\n"
)
for query_name, query_notes, recs in sample_results:
    report_text += f"Query: '{query_name}'\n"
    report_text += f"Notes: '{query_notes} ...'\n"
    for i, (name, notes, score) in enumerate(recs, 1):
        report_text += f"  {i}. [{score}] {name}\n"
        report_text += f"     Notes: {notes}...\n"
    report_text += "\n"

report_text += "==========================================================="

with open("evaluasi_model.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)

# =========================
# GENERATE GRAFIK EVALUASI
# =========================
print("⏳ 6. Membuat visualisasi (grafik_evaluasi.png)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Evaluasi Model NLP Dual Matrix — Avg Similarity: {avg_similarity:.4f}',
             fontsize=14, fontweight='bold')

# 1. Histogram distribusi
sns.histplot(all_similarity_scores, bins=40, kde=True, color='#f1c40f', ax=axes[0])
axes[0].axvline(avg_similarity, color='#e74c3c', linestyle='dashed', linewidth=2,
                label=f'Rata-rata: {avg_similarity:.4f}')
axes[0].axvline(median_similarity, color='#3498db', linestyle='dashed', linewidth=2,
                label=f'Median: {median_similarity:.4f}')
axes[0].set_title('Distribusi Skor Kemiripan Top-5')
axes[0].set_xlabel('Skor Kemiripan')
axes[0].set_ylabel('Frekuensi')
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.5)

# 2. Scatter Notes vs Desc vs Blend (100 sampel acak)
sample_100 = np.random.choice(len(df), min(100, len(df)), replace=False)
scores_ingr, scores_desc, scores_blend = [], [], []
for idx in sample_100:
    for mat, store in [(cs_ingr, scores_ingr), (cs_desc, scores_desc), (cosine_sim, scores_blend)]:
        ss = sorted(enumerate(mat[idx]), key=lambda x: x[1], reverse=True)[1:6]
        store.append(np.mean([s for _, s in ss]))

axes[1].scatter(scores_ingr, scores_blend, alpha=0.5, s=20, color='#e74c3c', label='vs Notes Only')
axes[1].scatter(scores_desc, scores_blend, alpha=0.5, s=20, color='#2ecc71', label='vs Desc Only')
axes[1].set_xlabel('Skor Matrix Tunggal')
axes[1].set_ylabel('Skor Blend Final')
axes[1].set_title('Notes/Desc vs Blend\n(100 sampel acak)')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# 3. Distribusi Top Brand
brand_counts = df['Brand'].value_counts().head(12)
brand_counts.plot(kind='barh', ax=axes[2], color='#9b59b6')
axes[2].set_title('Distribusi Top 12 Brand')
axes[2].set_xlabel('Jumlah Parfum')
axes[2].grid(axis='x', alpha=0.5)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("grafik_evaluasi.png", bbox_inches='tight', dpi=120)
plt.close()
print("   Grafik disimpan ke grafik_evaluasi.png")

# =========================
# SIMPAN SEMUA MODEL
# =========================
print("⏳ 7. Menyimpan model ke .pkl...")
with open('tfidf_ingredient.pkl', 'wb') as f: pickle.dump(tfidf_ingr, f)
with open('tfidf_desc.pkl', 'wb')       as f: pickle.dump(tfidf_desc, f)
with open('mat_ingredient.pkl', 'wb')   as f: pickle.dump(mat_ingr, f)
with open('mat_desc.pkl', 'wb')         as f: pickle.dump(mat_desc, f)
with open('df_perfume.pkl', 'wb')       as f: pickle.dump(df[['Name', 'Brand', 'Notes', 'Description']], f)

config = {
    'alpha': ALPHA,
    'avg_similarity': round(float(avg_similarity), 4),
    'vocab_ingredient': len(tfidf_ingr.vocabulary_),
    'vocab_desc': len(tfidf_desc.vocabulary_),
    'total_perfumes': len(df),
}
with open('model_config.json', 'w') as f: json.dump(config, f, indent=2)

print("\n✅ SELESAI! File yang siap digunakan server:")
print("   - tfidf_ingredient.pkl & tfidf_desc.pkl")
print("   - mat_ingredient.pkl & mat_desc.pkl")
print("   - df_perfume.pkl & model_config.json")