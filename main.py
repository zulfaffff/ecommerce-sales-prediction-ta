import time
from src.data_preprocessing import preprocess_data
from src.eda_visualize import visualize_eda
from src.evaluate import evaluate_models
from src.feature_engineering import create_features
from src.feature_importance import plot_feature_importance
from src.train import train_models


def run_pipeline(file_path):
    print('=' * 70)
    print('🚀 MEMULAI PIPELINE EXPERIMENT (SKRIPSI MACHINE LEARNING)')
    print('=' * 70)
    start_time = time.time()

    # STEP 1: Data Preprocessing
    print('\n[1/5] Menjalankan Data Preprocessing...')
    df_clean = preprocess_data(file_path)
    print(f'   ↳ Preprocessing Selesai. Ukuran dataset: {df_clean.shape}')

    # STEP 2: Feature Engineering & TF-IDF
    print('\n[2/5] Menjalankan Feature Engineering & TF-IDF Extraction...')
    X_final, y, tfidf_model = create_features(df_clean, max_tfidf_features=500)
    kolom_numerik = [
        'harga_clean',
        'harga_coret_clean',
        'rating_clean',
        'jumlah_foto',
        'persen_diskon',
        'panjang_nama',
        'panjang_deskripsi',
    ]
    print(
        f'   ↳ Matriks Fitur Siap. Ukuran Matriks X: {X_final.shape}, Target Y: {y.shape}'
    )

    # STEP 3: Model Training (Splitting, SMOTE, & RandomizedSearchCV)
    print(
        '\n[3/5] Menjalankan Model Training (Data Splitting, SMOTE, Hyperparameter Tuning)...'
    )
    models, data_splits = train_models(X_final, y)
    print('   ↳ Model Gradient Boosting & Random Forest berhasil dilatih!')

    # STEP 4: Visualisasi & EDA
    print('\n[4/5] Membuat Visualisasi EDA & Grafik Distribusi...')
    visualize_eda(df_clean, data_splits=data_splits, save_plots=True)
    print('   ↳ Seluruh grafik EDA dan distribusi berhasil dibuat.')

    # STEP 5: Evaluasi & Feature Importance
    print('\n[5/5] Evaluasi Model & Feature Importance Analysis...')
    metrics = evaluate_models(models, data_splits, save_plots=True)
    plot_feature_importance(
        models, tfidf_model, kolom_numerik, top_n=15, save_plots=True
    )

    elapsed_time = time.time() - start_time
    print('\n' + '=' * 70)
    print('✅ EKSEKUSI PIPELINE SELESAI!')
    print(f'⏱️ Total Waktu Eksekusi: {elapsed_time:.2f} detik')
    print('=' * 70)

    # Tampilkan Ringkasan Performa Akhir
    print('\n📊 RINGKASAN PERFORMA MODEL:')
    print(
        f" • Gradient Boosting -> Accuracy: {metrics['gradient_boosting']['accuracy']*100:.2f}% | ROC-AUC: {metrics['gradient_boosting']['roc_auc']:.4f}"
    )
    print(
        f" • Random Forest     -> Accuracy: {metrics['random_forest']['accuracy']*100:.2f}% | ROC-AUC: {metrics['random_forest']['roc_auc']:.4f}"
    )
    print('=' * 70)


if __name__ == '__main__':
    PATH_DATASET = 'data/97ca96d3-e858-4d82-bcc1-c7a5d86376f0.csv'
    run_pipeline(PATH_DATASET)