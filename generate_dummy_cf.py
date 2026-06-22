import json
import pandas as pd
import random

print("Membaca UUID asli dari perfumes_rows.json...")
with open('perfumes_rows.json', 'r', encoding='utf-8') as f:
    parfum_data = json.load(f)

# Ambil semua ID parfum yang valid
parfum_ids = [p['id'] for p in parfum_data]
user_ids = [f"USER_{str(i).zfill(3)}" for i in range(1, 101)] # Bikin 100 User Dummy

interactions = []
for u in user_ids:
    # Tiap user memberi rating pada 5-20 parfum secara acak
    num_rates = random.randint(5, 20)
    rated_parfums = random.sample(parfum_ids, num_rates)
    for p in rated_parfums:
        interactions.append({
            "user_id": u,
            "parfum_id": p,
            "rating": random.randint(3, 5) # Rating 3 sampai 5
        })

df = pd.DataFrame(interactions)
df.to_csv('user_interactions_v2.csv', index=False)
print(f"✅ Berhasil membuat user_interaction_v2.csv dengan {len(df)} baris data valid!")