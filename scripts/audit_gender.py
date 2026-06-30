"""
======================================================================
AUDIT & PERBAIKAN gender_target DI SUPABASE (sekali-jalan, aman)
======================================================================
Masalah: kolom `gender_target` di tabel perfumes banyak salah label
(hasil scrape marketplace) — mis. parfum bernama "Posh Women" ditandai
"Pria". Ini bikin hasil rekomendasi tidak konsisten dengan filter gender.

Script ini menebak gender yang BENAR dari NAMA + deskripsi parfum, lalu:

  FASE 1 (default / dry-run):
      python scripts/audit_gender.py
    -> TIDAK mengubah apa pun. Hanya membuat file usulan:
       dataset/parfum/gender_review.csv
       (kolom: id, nama_parfum, gender_lama, gender_baru, sinyal, confidence)
    -> Buka CSV itu, periksa. Hapus baris yang kamu TIDAK setuju,
       atau betulkan kolom `gender_baru` secara manual.

  FASE 2 (apply):
      python scripts/audit_gender.py --apply
    -> Membaca gender_review.csv (versi yang sudah kamu edit) dan
       meng-update kolom gender_target di Supabase, baris per baris.
       Hanya baris dengan `gender_baru` terisi yang diproses.

  Cek logika tebakan tanpa internet:
      python scripts/audit_gender.py --selftest

Catatan: setelah apply, sebaiknya sinkron + retrain agar fitur TF-IDF ikut
segar:  python scripts/sync_from_supabase.py && python scripts/generate_dataset.py && python scripts/train_model.py
(opsional — filter gender di main.py sudah baca DB langsung).
======================================================================
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = BASE_DIR / "dataset" / "parfum" / "gender_review.csv"

# Kata gender eksplisit. \b agar "men" tidak ikut match di dalam "women".
FEM_NAME = r"\b(?:women|woman|wanita|ladies|lady|girl|miss|feminine|feminin|cewe|cewek|mademoiselle)\b"
MASC_NAME = r"\b(?:men|man|pria|homme|male|gentleman|masculine|maskulin|cowo|cowok|boy)\b"
FEM_DESC = r"\b(?:women|woman|wanita|feminine|feminin)\b"
MASC_DESC = r"\b(?:men|man|pria|masculine|maskulin|homme)\b"

# Nama brand/selebriti yang jelas-gender tapi TANPA kata generik (tak bisa
# ditangkap regex). Edit/ tambah sesuai pengetahuanmu. Dicocokkan sebagai substring.
OVERRIDE_NAMA = {
    "nagita slavina": "Wanita",
    "aqua kiss": "Wanita",
    "scarlett": "Wanita",
}


def tebak_gender(nama: str, deskripsi: str = ""):
    """Kembalikan (gender, confidence, sinyal) atau (None, None, None) bila tak ada bukti."""
    n = (nama or "").lower()

    for kunci, g in OVERRIDE_NAMA.items():
        if kunci in n:
            return g, "tinggi", f"nama mengandung '{kunci}' (daftar manual)"

    fem = bool(re.search(FEM_NAME, n))
    masc = bool(re.search(MASC_NAME, n))
    if fem and not masc:
        return "Wanita", "tinggi", "nama mengandung kata feminin"
    if masc and not fem:
        return "Pria", "tinggi", "nama mengandung kata maskulin"
    if fem and masc:
        return "Unisex", "sedang", "nama menyebut pria & wanita"

    d = (deskripsi or "").lower()
    fem_d = bool(re.search(FEM_DESC, d))
    masc_d = bool(re.search(MASC_DESC, d))
    if fem_d and not masc_d:
        return "Wanita", "sedang", "deskripsi menyebut wanita"
    if masc_d and not fem_d:
        return "Pria", "sedang", "deskripsi menyebut pria"

    return None, None, None  # tak ada sinyal -> jangan usulkan perubahan


def _norm(g: str) -> str:
    g = (g or "").strip().lower()
    if "wanita" in g:
        return "Wanita"
    if "pria" in g:
        return "Pria"
    if "unisex" in g:
        return "Unisex"
    return g.title()


def selftest():
    assert tebak_gender("Posh Women 150ml")[0] == "Wanita"
    assert tebak_gender("YSL Libre Women")[0] == "Wanita"
    assert tebak_gender("Sauvage Homme for Men")[0] == "Pria"
    # "women" TIDAK boleh ketrigger sebagai maskulin gara-gara substring "men"
    assert tebak_gender("Posh Women")[0] != "Pria"
    # Men's and Women's -> dua-duanya -> Unisex
    assert tebak_gender("Men's and Women's Perfume")[0] == "Unisex"
    # daftar manual
    assert tebak_gender("Nagita Slavina EDP 35ml")[0] == "Wanita"
    # tak ada sinyal -> None (jujur, tak menebak ngawur)
    assert tebak_gender("Aurora Oud Intense")[0] is None
    print("✅ selftest lulus — logika tebakan gender aman.")


def connect() -> Client:
    load_dotenv(BASE_DIR / ".env", override=True)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌ SUPABASE_URL / SUPABASE_KEY tidak ada di .env")
        sys.exit(1)
    return create_client(url, key, options=ClientOptions(schema="public"))


def tarik_semua(supabase: Client):
    data, limit, offset = [], 1000, 0
    while True:
        res = supabase.table("perfumes").select("*").range(offset, offset + limit - 1).execute()
        if not res.data:
            break
        data.extend(res.data)
        offset += limit
    return data


def dry_run():
    supabase = connect()
    print("⏳ Menarik data dari Supabase ...")
    rows = tarik_semua(supabase)
    print(f"✅ {len(rows)} parfum dianalisis.\n")

    usulan = []
    for r in rows:
        nama = r.get("nama_parfum", "")
        baru, conf, sinyal = tebak_gender(nama, r.get("description", ""))
        lama = _norm(r.get("gender_target", ""))
        if baru and baru != lama:
            usulan.append({
                "id": r.get("id"),
                "nama_parfum": nama,
                "gender_lama": r.get("gender_target", ""),
                "gender_baru": baru,
                "sinyal": sinyal,
                "confidence": conf,
            })

    if not usulan:
        print("✅ Tidak ada usulan perubahan — data gender sudah konsisten dgn sinyal nama/deskripsi.")
        return

    df = pd.DataFrame(usulan).sort_values(by=["confidence", "gender_baru"], ascending=[False, True])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"📝 {len(df)} usulan perubahan ditulis ke:\n   {OUT_CSV}\n")
    print(df[["nama_parfum", "gender_lama", "gender_baru", "confidence"]].to_string(index=False, max_colwidth=45))
    print("\nLangkah berikut: buka CSV, periksa/edit, lalu jalankan:")
    print("   python scripts/audit_gender.py --apply")


def apply_csv():
    if not OUT_CSV.exists():
        print(f"❌ {OUT_CSV} belum ada. Jalankan dry-run dulu (tanpa --apply).")
        sys.exit(1)
    supabase = connect()
    df = pd.read_csv(OUT_CSV)
    n = 0
    for _, r in df.iterrows():
        baru = str(r.get("gender_baru", "")).strip()
        gid = r.get("id")
        if not baru or baru.lower() == "nan" or pd.isna(gid):
            continue
        supabase.table("perfumes").update({"gender_target": baru}).eq("id", gid).execute()
        n += 1
        print(f"  ✓ {r.get('nama_parfum','')[:50]:50s} -> {baru}")
    print(f"\n✅ Selesai. {n} baris di-update di Supabase.")
    print("Disarankan retrain: python scripts/sync_from_supabase.py && python scripts/generate_dataset.py && python scripts/train_model.py")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--apply":
        apply_csv()
    else:
        dry_run()
