import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def create_features(df, max_tfidf_features=500):
    """Mengekstrak/membuat fitur numerik dan matriks TF-IDF dari dataframe bersih.

    Mengembalikan X_final (matriks fitur gabungan), y (target), dan tfidf_model.
    """
    df = df.copy()

    # --- 1. BUAT FITUR TURUNAN NUMERIK JIKA BELUM ADA ---
    if 'persen_diskon' not in df.columns:
        # Menghitung % diskon: (harga_coret - harga) / harga_coret * 100
        # Jika harga_coret 0/NaN, beri nilai 0
        df['persen_diskon'] = np.where(
            (df['harga_coret_clean'] > 0)
            & (df['harga_coret_clean'] > df['harga_clean']),
            (
                (df['harga_coret_clean'] - df['harga_clean'])
                / df['harga_coret_clean']
            )
            * 100,
            0,
        )

    if 'panjang_nama' not in df.columns:
        # Menghitung jumlah karakter nama produk (jika ada kolom nama_produk atau text_clean)
        col_nama = (
            'nama_produk' if 'nama_produk' in df.columns else 'text_clean'
        )
        df['panjang_nama'] = df[col_nama].astype(str).str.len()

    if 'panjang_deskripsi' not in df.columns:
        # Menghitung jumlah karakter deskripsi (jika ada kolom deskripsi/text_clean)
        col_desk = (
            'deskripsi' if 'deskripsi' in df.columns else 'text_clean'
        )
        df['panjang_deskripsi'] = df[col_desk].astype(str).str.len()

    # --- 2. PILIH FITUR NUMERIK UTAMA ---
    kolom_numerik = [
        'harga_clean',
        'harga_coret_clean',
        'rating_clean',
        'jumlah_foto',
        'persen_diskon',
        'panjang_nama',
        'panjang_deskripsi',
    ]

    # Pastikan jumlah_foto & rating_clean terisi (fill NaN dengan 0 jika ada)
    for col in kolom_numerik:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    X_num = df[kolom_numerik].values

    # --- 3. EKSTRAKSI FITUR TEKS DENGAN TF-IDF ---
    # Pastikan kolom text_clean string dan tidak null
    text_data = (
        df['text_clean'].fillna('').astype(str)
        if 'text_clean' in df.columns
        else pd.Series([''] * len(df))
    )

    tfidf = TfidfVectorizer(max_features=max_tfidf_features)
    X_tfidf = tfidf.fit_transform(text_data).toarray()

    # Simpan model TF-IDF vectorizer
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

    # --- 4. GABUNGKAN FITUR & AMBIL TARGET ---
    X_final = np.hstack((X_num, X_tfidf))
    y = df['label_target'].values

    return X_final, y, tfidf


if __name__ == '__main__':
    from src.data_preprocessing import preprocess_data

    file_path = 'data/97ca96d3-e858-4d82-bcc1-c7a5d86376f0.csv'
    df_clean = preprocess_data(file_path)
    X_final, y, tfidf_model = create_features(df_clean)
    print(
        f'✅ Feature Engineering Selesai! Ukuran X: {X_final.shape}, Ukuran y: {y.shape}'
    )