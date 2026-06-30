import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PARFUM_JSON = BASE_DIR / "dataset" / "parfum" / "perfumes_rows.json"
OUTPUT_SQL = BASE_DIR / "dataset" / "sql" / "insert_perfumes_cleaned.sql"

def scrub_ecommerce_spam(text):
    """Fungsi Regex untuk menyapu bersih bahasa iklan marketplace."""
    if not text or str(text).strip() == '' or str(text).lower() in ['n/a', 'nan', 'null']:
        return ""
    
    text_clean = str(text)
    
    # 1. Hapus Harga dan Persentase Diskon
    text_clean = re.sub(r'\b[Rr]p\s?\d+([\.,]\d+)*\b', ' ', text_clean)
    text_clean = re.sub(r'[-+]?\d+%', ' ', text_clean)
    
    # 2. Hapus UI Rating Marketplace (App 4.4 dll)
    text_clean = re.sub(r'[Aa]pp\s\d\.\d\s*\(\d+\)', ' ', text_clean)
    
    # 3. Hapus Karakter Aneh / Emoji yang rusak
    text_clean = re.sub(r'[⭐✔️✅🔥💯]', ' ', text_clean)
    
    # 4. Dictionary Kata Sampah
    spam_phrases = [
        r'(?i)paket parfum', r'(?i)viral tik\s?tok', r'(?i)travel size', r'(?i)bundling', 
        r'(?i)promosi:', r'(?i)beli \d+ diskon', r'(?i)pilihan pengiriman\s?:', 
        r'(?i)dki jakarta', r'(?i)kota jakarta barat', r'(?i)cengkareng', r'(?i)ubah',
        r'(?i)dijamin tiba', r'(?i)standar,with shipping fee', r'(?i)pengembalian & garansi', 
        r'(?i)berubah pikiran', r'(?i)pengembalian gratis', r'(?i)\d+ bulan garansi', 
        r'(?i)pusat layanan resmi setempat', r'(?i)kuantitas:', r'(?i)beli sekarang', 
        r'(?i)tambah ke troli', r'(?i)share like', r'(?i)bisa cod', r'(?i)gratis ongkir', 
        r'(?i)free ongkir', r'(?i)murce', r'(?i)murah berkualitas', r'(?i)isi \d+ pcs',
        r'(?i)\d+ botol', r'(?i)ready stock', r'(?i)original \d+%', r'(?i)terlaris', 
        r'(?i)best seller', r'(?i)bestseller', r'(?i)flash sale', r'(?i)tahan lama',
        r'(?i)ecer', r'(?i)bpom'
    ]
    
    for phrase in spam_phrases:
        text_clean = re.sub(phrase, ' ', text_clean)
        
    return re.sub(r'\s+', ' ', text_clean).strip()

def escape_sql_string(val):
    if val is None or str(val).strip() == '' or str(val).lower() == 'n/a':
        return "NULL"
    safe_val = str(val).replace("'", "''")
    return f"'{safe_val}'"

def generate_sql_from_json():
    print("Membaca file perfumes_rows.json...")
    try:
        with open(PARFUM_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ ERROR: File perfumes_rows.json tidak ditemukan!")
        return

    sql_statements = []
    sql_statements.append("-- ===========================================")
    sql_statements.append("-- SCRIPT INSERT DATA PARFUM BERSIH DARI JSON")
    sql_statements.append("-- ===========================================")
    sql_statements.append("TRUNCATE TABLE public.perfumes RESTART IDENTITY CASCADE;\n")

    # Kolom Canonical sesuai Supabase
    columns = [
        'nama_parfum', 'brand', 'description', 'notes', 'harga',
        'gender_target', 'waktu_penggunaan', 'occasion', 'wangi_vibe',
        'url_foto', 'url_produk', 'rating_value', 'rating_count',
        'concentration', 'top_notes', 'middle_notes', 'base_notes',
        'main_accords', 'cuaca_musim', 'longevity', 'sillage'
    ]

    valid_count = 0
    for row in data:
        # TAHAP CLEANING DATA MARKETPLACE
        row['description'] = scrub_ecommerce_spam(row.get('description', ''))
        row['nama_parfum'] = scrub_ecommerce_spam(row.get('nama_parfum', ''))

        # Jika setelah dibersihkan namanya habis/kosong, lewati data ini
        if not row['nama_parfum']:
            continue

        vals = []
        for col in columns:
            if col in ['harga', 'rating_value', 'rating_count']:
                val = row.get(col, 0)
                if val is None or str(val).strip() == '' or str(val).lower() == 'n/a':
                    val = 0
                try:
                    clean_num = str(val).replace('Rp', '').replace('.', '').replace(',', '.').strip()
                    val = float(clean_num)
                except ValueError:
                    val = 0
                vals.append(str(val))
            else:
                vals.append(escape_sql_string(row.get(col, '')))

        cols_str = ", ".join(columns)
        vals_str = ", ".join(vals)
        sql = f"INSERT INTO public.perfumes ({cols_str}) VALUES ({vals_str});"
        sql_statements.append(sql)
        valid_count += 1

    with open(OUTPUT_SQL, 'w', encoding='utf-8') as f:
        f.write("\n".join(sql_statements))

    print(f"✅ Selesai! Berhasil memproses {valid_count} data dari JSON dan membersihkan spam.")
    print("✅ File 'insert_perfumes_cleaned.sql' siap dieksekusi di Supabase.")

if __name__ == "__main__":
    generate_sql_from_json()