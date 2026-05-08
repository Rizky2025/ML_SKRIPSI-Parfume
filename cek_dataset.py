import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer

print("⏳ Memulai proses training model Machine Learning...")

# 1. Load Dataset
df = pd.read_csv('final_perfume_data.csv')

# 2. Preprocessing Dasar (Handle missing values & jadikan huruf kecil)
df['Notes'] = df['Notes'].fillna('').str.lower()
df['Description'] = df['Description'].fillna('').str.lower()

# 3. Buat DUA Model TF-IDF Vectorizer
# Menggunakan stop_words='english' agar AI membuang kata tak penting seperti "the", "and", "is"
print("🧠 Membangun TF-IDF untuk Ingredients (Notes)...")
tfidf_ingr = TfidfVectorizer(stop_words='english')
mat_ingr = tfidf_ingr.fit_transform(df['Notes'])

print("🧠 Membangun TF-IDF untuk Context (Description)...")
tfidf_desc = TfidfVectorizer(stop_words='english')
mat_desc = tfidf_desc.fit_transform(df['Description'])

# 4. Simpan Model, Matrix, dan Data ke file Pickle
with open('tfidf_ingredient.pkl', 'wb') as f: pickle.dump(tfidf_ingr, f)
with open('tfidf_desc.pkl', 'wb') as f: pickle.dump(tfidf_desc, f)
with open('mat_ingredient.pkl', 'wb') as f: pickle.dump(mat_ingr, f)
with open('mat_desc.pkl', 'wb') as f: pickle.dump(mat_desc, f)

# Simpan kolom yang penting saja agar memori tidak berat
with open('df_perfume.pkl', 'wb') as f: pickle.dump(df[['Name', 'Brand', 'Notes']], f)

# 5. Simpan Konfigurasi (Alpha: Bobot Gabungan)
# 0.5 artinya bobot bahan baku (Notes) dan bobot suasana (Description) sama kuat 50:50
config = {"alpha": 0.5} 
with open('model_config.json', 'w') as f: json.dump(config, f)

print("✅ Training Selesai! Otak AI kamu sekarang jauh lebih pintar!")