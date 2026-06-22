import pandas as pd
import numpy as np
import random

# =========================================================
# LOAD DATA PARFUM
# =========================================================
df_parfum = pd.read_csv('perfumes_rows.csv')

# Pastikan kolom bertipe string
kolom_kriteria = [
    'gender_target',
    'wangi_vibe',
    'waktu_penggunaan',
    'occasion'
]

for col in kolom_kriteria:
    if col in df_parfum.columns:
        df_parfum[col] = df_parfum[col].fillna('').astype(str)

# =========================================================
# PILIHAN PREFERENSI USER
# =========================================================
pilihan_gender = ['Pria', 'Wanita', 'Unisex']
pilihan_aroma = ['Woody', 'Floral', 'Sweet', 'Fresh', 'Spicy']
pilihan_waktu = ['Pagi', 'Siang', 'Malam']
pilihan_kegiatan = ['Kerja', 'Santai', 'Kencan', 'Formal']

# =========================================================
# PARAMETER DATASET
# =========================================================
num_users = 500

data_interaksi = []
data_profil_user = []

print("Mulai generate dataset interaksi user...")

# =========================================================
# GENERATE USER
# =========================================================
for i in range(1, num_users + 1):

    user_id = f"USER_{i:03d}"

    # =====================================================
    # USER PREFERENCE
    # =====================================================
    pref_gender = random.choice(pilihan_gender)
    pref_aroma = random.choice(pilihan_aroma)
    pref_waktu = random.choice(pilihan_waktu)
    pref_kegiatan = random.choice(pilihan_kegiatan)

    # Simpan profil user
    data_profil_user.append({
        'user_id': user_id,
        'pref_gender': pref_gender,
        'pref_aroma': pref_aroma,
        'pref_waktu': pref_waktu,
        'pref_kegiatan': pref_kegiatan
    })

    # =====================================================
    # HITUNG SKOR KECOCOKAN
    # =====================================================
    skor = (
        df_parfum['gender_target'].str.contains(pref_gender, case=False, na=False).astype(int) +
        df_parfum['wangi_vibe'].str.contains(pref_aroma, case=False, na=False).astype(int) +
        df_parfum['waktu_penggunaan'].str.contains(pref_waktu, case=False, na=False).astype(int) +
        df_parfum['occasion'].str.contains(pref_kegiatan, case=False, na=False).astype(int)
    )

    df_parfum['match_score'] = skor

    # =====================================================
    # KELOMPOK PARFUM
    # =====================================================

    # Sangat cocok
    parfum_sangat_cocok = df_parfum[df_parfum['match_score'] >= 3]

    # Cukup cocok
    parfum_cocok = df_parfum[df_parfum['match_score'] == 2]

    # Kurang cocok
    parfum_biasa = df_parfum[df_parfum['match_score'] <= 1]

    # =====================================================
    # JUMLAH REVIEW USER
    # =====================================================
    num_reviews = random.randint(15, 30)

    for _ in range(num_reviews):

        probabilitas = random.random()

        # =================================================
        # 60% -> parfum sangat cocok
        # =================================================
        if probabilitas < 0.6 and len(parfum_sangat_cocok) > 0:

            selected = parfum_sangat_cocok.sample(1).iloc[0]

            rating = np.random.choice(
                [4, 5],
                p=[0.3, 0.7]
            )

        # =================================================
        # 25% -> parfum cukup cocok
        # =================================================
        elif probabilitas < 0.85 and len(parfum_cocok) > 0:

            selected = parfum_cocok.sample(1).iloc[0]

            rating = np.random.choice(
                [3, 4],
                p=[0.6, 0.4]
            )

        # =================================================
        # 15% -> parfum random / kurang cocok
        # =================================================
        else:

            if len(parfum_biasa) > 0:
                selected = parfum_biasa.sample(1).iloc[0]
            else:
                selected = df_parfum.sample(1).iloc[0]

            rating = np.random.choice(
                [1, 2, 3],
                p=[0.5, 0.3, 0.2]
            )

        # =================================================
        # SIMPAN INTERAKSI
        # =================================================
        data_interaksi.append({

            # USER
            'user_id': user_id,

            # PARFUM
            'parfum_id': selected['id'],
            'parfum_name': selected.get('name', ''),
            'brand': selected.get('brand', ''),

            # PREFERENSI USER
            'user_gender_pref': pref_gender,
            'user_aroma_pref': pref_aroma,
            'user_time_pref': pref_waktu,
            'user_occasion_pref': pref_kegiatan,

            # ATRIBUT PARFUM
            'parfum_gender_target': selected.get('gender_target', ''),
            'parfum_wangi_vibe': selected.get('wangi_vibe', ''),
            'parfum_waktu_penggunaan': selected.get('waktu_penggunaan', ''),
            'parfum_occasion': selected.get('occasion', ''),

            # SKOR
            'match_score': selected['match_score'],

            # RATING
            'rating': int(rating)

        })

# =========================================================
# DATAFRAME
# =========================================================
df_dummy = pd.DataFrame(data_interaksi)

# Hapus duplikat user-parfum
df_dummy = df_dummy.drop_duplicates(
    subset=['user_id', 'parfum_id']
)

# =========================================================
# SIMPAN CSV
# =========================================================
df_dummy.to_csv(
    'dataset_rating_variatif.csv',
    index=False
)

df_profil = pd.DataFrame(data_profil_user)

df_profil.to_csv(
    'profil_user_dummy.csv',
    index=False
)

# =========================================================
# HASIL
# =========================================================
print("\n====================================")
print("DATASET BERHASIL DIBUAT")
print("====================================")
print(f"Jumlah User        : {num_users}")
print(f"Jumlah Interaksi   : {len(df_dummy)}")
print("File utama         : dataset_rating_variatif.csv")
print("File profil user   : profil_user_dummy.csv")
print("====================================")