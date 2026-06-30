"""
Daftar stopword (kata sambung/umum) Bahasa Indonesia + Inggris.
Dipakai TF-IDF agar kata seperti "yang", "untuk", "dengan" TIDAK dihitung
sebagai kecocokan — supaya skor & bukti CBF benar-benar berbasis kata bermakna
(aroma, notes, occasion), bukan kata sambung.

Dipakai oleh: scripts/train_model.py dan scripts/evaluate_model.py
(main.py tidak perlu — ia memuat model TF-IDF yang sudah jadi).
"""

STOPWORDS_ID = [
    "yang", "untuk", "dengan", "dan", "di", "ke", "dari", "pada", "ini", "itu",
    "atau", "juga", "akan", "dalam", "adalah", "sebagai", "oleh", "agar", "saat",
    "saja", "sangat", "lebih", "bisa", "ada", "tidak", "yg", "utk", "dll", "para",
    "karena", "namun", "tetapi", "tapi", "serta", "maupun", "kami", "kita", "anda",
    "saya", "dia", "mereka", "nya", "se", "para", "bagi", "hingga", "sampai",
    "ketika", "sehingga", "secara", "antara", "tanpa", "telah", "sudah", "masih",
]

STOPWORDS_EN = [
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "is",
    "are", "this", "that", "it", "as", "by", "from", "at", "be", "your", "you",
]

STOPWORDS = sorted(set(STOPWORDS_ID + STOPWORDS_EN))
