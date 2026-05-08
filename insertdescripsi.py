import os
import json
import time
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI

# ==========================================
# CONFIG
# ==========================================
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASS = "your_password"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# CONNECT DB
# ==========================================
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

# ==========================================
# CLEAN NOTES
# ==========================================
def clean_notes(notes):
    if not notes:
        return ""

    notes = notes.strip()

    # remove weird scraped text
    notes = re.sub(r"Click Here.*", "", notes, flags=re.I)
    notes = re.sub(r"Please be aware.*", "", notes, flags=re.I)

    if notes.lower() == "nan":
        return ""

    return notes.strip()


# ==========================================
# GENERATE DESCRIPTION VIA GPT
# ==========================================
def generate_description(name, brand, notes):
    prompt = f"""
Buat deskripsi parfum dalam Bahasa Indonesia.

Nama parfum: {name}
Brand: {brand}
Notes: {notes}

Rules:
- Panjang 2 sampai 3 kalimat
- Elegan dan menarik
- Jelaskan karakter aroma
- Cocok untuk siapa / suasana apa
- Jangan terlalu berlebihan
- Output hanya deskripsi saja
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# ==========================================
# FETCH DATA
# ==========================================
cur = conn.cursor(cursor_factory=RealDictCursor)

cur.execute("""
SELECT id, name, brand, notes
FROM perfumes
WHERE description IS NULL
ORDER BY created_at ASC
""")

rows = cur.fetchall()

print(f"Total data: {len(rows)}")

# ==========================================
# LOOP UPDATE
# ==========================================
for i, row in enumerate(rows, start=1):
    perfume_id = row["id"]
    name = row["name"]
    brand = row["brand"]
    notes = clean_notes(row["notes"])

    try:
        desc = generate_description(name, brand, notes)

        cur.execute("""
            UPDATE perfumes
            SET description = %s
            WHERE id = %s
        """, (desc, perfume_id))

        conn.commit()

        print(f"[{i}/{len(rows)}] Updated: {name}")

        time.sleep(1.2)  # biar aman rate limit

    except Exception as e:
        conn.rollback()
        print(f"ERROR {name}: {e}")

# ==========================================
# DONE
# ==========================================
cur.close()
conn.close()

print("Selesai generate semua description.")