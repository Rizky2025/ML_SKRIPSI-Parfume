import pandas as pd
import re
import sys


# =========================
# KONFIGURASI
# =========================

INPUT_FILE = "dataset1.csv"
OUTPUT_FILE = "data_baru_cleaned.csv"

REQUIRED_COLUMNS = {
    "nama_parfum",
    "description"
}

COLUMN_MAPPING = {
    "Name": "nama_parfum",
    "Brand": "brand",
    "Description": "description",
    "Notes": "notes",
    "Image URL": "url_foto",
    "Product URL": "url_produk",
}


# =========================
# HAPUS SPAM MARKETPLACE
# =========================

SPAM_PATTERNS = [
    r"(?i)beli sekarang",
    r"(?i)tambah ke troli",
    r"(?i)flash sale",
    r"(?i)voucher",
    r"(?i)cashback",
    r"(?i)checkout",
    r"(?i)shipping fee",
    r"(?i)promo",
    r"(?i)diskon",
    r"(?i)gratis ongkir",
    r"(?i)stok tersedia",
]


# =========================
# CLEAN TEXT
# =========================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = value.replace("\n", " ")
    value = value.replace("\r", " ")

    for pattern in SPAM_PATTERNS:
        value = re.sub(pattern, "", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# =========================
# LOAD CSV (ANTI ERROR)
# =========================

def load_csv(path):

    try:

        df = pd.read_csv(
            path,
            dtype=str,
            engine="python",
            sep=",",
            quotechar='"',
            on_bad_lines="skip"
        )

        return df

    except Exception as e:

        print("\nERROR membaca CSV")
        print(e)

        sys.exit()


# =========================
# NORMALISASI KOLOM
# =========================

def normalize_columns(df):

    df.rename(
        columns=COLUMN_MAPPING,
        inplace=True
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    return df


# =========================
# VALIDASI
# =========================

def validate_columns(df):

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:

        raise Exception(
            f"\nKolom wajib tidak ditemukan:\n{missing}"
        )


# =========================
# CLEAN DATAFRAME
# =========================

def clean_dataframe(df):

    for col in df.columns:

        try:
            df[col] = df[col].apply(clean_text)

        except:
            pass

    return df


# =========================
# DROP DUPLIKAT
# =========================

def remove_duplicates(df):

    before = len(df)

    df.drop_duplicates(inplace=True)

    after = len(df)

    print(f"\nDuplicate dihapus: {before-after}")

    return df


# =========================
# SIMPAN
# =========================

def save(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nBerhasil disimpan:")
    print(OUTPUT_FILE)

    print(f"\nTotal data: {len(df)}")


# =========================
# MAIN
# =========================

def main():

    print("\nMEMBACA DATA...")

    df = load_csv(INPUT_FILE)

    print("NORMALISASI KOLOM...")

    df = normalize_columns(df)

    print("VALIDASI...")

    validate_columns(df)

    print("CLEANING...")

    df = clean_dataframe(df)

    print("REMOVE DUPLICATE...")

    df = remove_duplicates(df)

    print("SAVE...")

    save(df)

    print("\nSELESAI")


if __name__ == "__main__":
    main()