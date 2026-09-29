import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords jika belum ada
nltk.download('stopwords', quiet=True)


# =====================================================================
# HELPER FUNCTIONS FOR CLEANING
# =====================================================================
def clean_harga(x):
    """Membersihkan kolom harga dan harga_coret."""
    if pd.isna(x) or str(x).strip() == "":
        return 0

    teks = str(x).replace('Rp', '').replace('.', '').strip()

    if '-' in teks:
        teks = teks.split('-')[0].strip()

    try:
        return int(teks)
    except ValueError:
        return 0


def clean_terjual(x):
    """Membersihkan kolom jumlah terjual."""
    if pd.isna(x) or str(x).strip() == "":
        return 0
    teks = str(x).upper().replace('+', '').replace('.', '').replace(' ', '').strip()
    if 'RB' in teks:
        teks = teks.replace('RB', '').replace(',', '.')
        return int(float(teks) * 1000)
    try:
        return int(teks)
    except ValueError:
        return 0


def clean_rating(x):
    """Membersihkan kolom rating."""
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    teks = str(x).replace(',', '.')
    try:
        return float(teks)
    except ValueError:
        return 0.0


def clean_jumlah_rating(x):
    """Membersihkan kolom jumlah rating."""
    if pd.isna(x) or str(x).strip() == "":
        return 0

    teks = str(x).upper().replace('+', '').replace('.', '').replace(' ', '').strip()

    if 'RB' in teks:
        teks = teks.replace('RB', '').replace(',', '.')
        try:
            return int(float(teks) * 1000)
        except ValueError:
            return 0

    try:
        return int(float(teks.replace(',', '.')))
    except ValueError:
        return 0


def remove_noise(text):
    """Menghapus URL, angka, simbol, dan spasi ganda dari teks."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =====================================================================
# MAIN PREPROCESSING PIPELINE
# =====================================================================
def preprocess_data(file_path):
    """Fungsi utama untuk memuat, membersihkan, dan memproses data."""
    # 1. Load Data
    df = pd.read_csv(file_path, sep=';', encoding='cp1252')

    # 2. Cleaning Kolom Numerik
    df['harga_clean'] = df['harga'].apply(clean_harga)
    df['harga_coret_clean'] = df['harga_coret'].apply(clean_harga)
    df.loc[df['harga_coret_clean'] == 0, 'harga_coret_clean'] = df['harga_clean']

    df['terjual_clean'] = df['terjual'].apply(clean_terjual)
    df['rating_clean'] = df['rating'].apply(clean_rating)
    df['jumlah_rating_clean'] = df['jumlah_rating'].apply(clean_jumlah_rating)

    # 3. Penanganan Duplikasi
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Imputasi Missing Values
    kolom_numerik = ['harga_clean', 'terjual_clean', 'rating_clean', 'jumlah_rating_clean']
    for col in kolom_numerik:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df['desc'] = df['desc'].fillna("Tanpa Deskripsi")

    # 5. Labeling Target berdasarkan Threshold
    threshold = 1000
    df['label_target'] = df['terjual_clean'].apply(lambda x: 1 if x >= threshold else 0)

    # 6. Text Preprocessing (NLP)
    # Step 1: Case Folding
    df['text_original'] = df['product_name'].astype(str) + " " + df['desc'].astype(str)
    df['step_1_folding'] = df['text_original'].str.lower()

    # Step 2: Noise Removal
    df['step_2_cleaning'] = df['step_1_folding'].apply(remove_noise)

    # Step 3: Tokenization
    df['step_3_token'] = df['step_2_cleaning'].apply(lambda x: x.split())

    # Step 4: Stopword Removal
    stop_ind = stopwords.words('indonesian')
    sampah_tambahan = [
        'dan', 'yang', 'untuk', 'dengan', 'di', 'anda', 'kami', 'tidak',
        'jika', 'dari', 'akan', 'dalam', 'bisa', 'lebih', 'produk',
        'pengiriman', 'barang', 'ponsel', 'hp', 'model', 'for',
        's', 'c', 'g', 'x', 'y', 'm', 'a'
    ]
    stop_ind.extend(sampah_tambahan)

    df['step_4_stopword'] = df['step_3_token'].apply(
        lambda x: [t for t in x if t not in stop_ind and len(t) > 2]
    )

    # Step 5: Text Re-joining
    df['text_clean'] = df['step_4_stopword'].apply(lambda x: " ".join(x))

    return df


if __name__ == "__main__":
    # Test eksekusi file secara mandiri
    file_path = '97ca96d3-e858-4d82-bcc1-c7a5d86376f0.csv'
    df_clean = preprocess_data(file_path)
    print(f"✅ Data Preprocessing Selesai! Ukuran dataset: {df_clean.shape}")