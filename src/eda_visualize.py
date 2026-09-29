from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Set gaya tampilan Seaborn untuk standar publikasi ilmiah
sns.set_theme(style='whitegrid', font_scale=1.0)


def visualize_eda(df, data_splits=None, save_plots=True):
    """Fungsi untuk menjalankan Analisis Data Eksploratif (EDA) dan visualisasi,

    mulai dari distribusi target, analisis teks, WordCloud, hingga distribusi
    SMOTE.
    """
    # =====================================================================
    # 1. ANALISIS DESKRIPTIF & BOXPLOT
    # =====================================================================
    mean_val = df['terjual_clean'].mean()
    median_val = df['terjual_clean'].median()
    min_val = df['terjual_clean'].min()
    max_val = df['terjual_clean'].max()

    plt.figure(figsize=(10, 4.5))
    sns.boxplot(
        data=df,
        x='terjual_clean',
        color='#91bfdb',
        fliersize=5,
        showmeans=True,
        meanprops={
            'marker': 'o',
            'markerfacecolor': 'red',
            'markeredgecolor': 'red',
            'markersize': '8',
        },
    )

    plt.annotate(
        f'Mean: {mean_val:.2f}',
        xy=(mean_val, 0),
        xytext=(mean_val, -0.25),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1),
    )
    plt.annotate(
        f'Median: {median_val:.0f}',
        xy=(median_val, 0),
        xytext=(median_val, 0.35),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color='black',
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
    )
    plt.text(
        min_val,
        0.45,
        f'Min: {min_val:.0f}',
        ha='center',
        fontsize=9,
        color='#333333',
    )
    plt.text(
        max_val,
        0.45,
        f'Max: {max_val:.0f}',
        ha='center',
        fontsize=9,
        color='#333333',
    )

    plt.title(
        'Boxplot Distribusi Penjualan Produk (terjual_clean)',
        fontsize=12,
        fontweight='bold',
        pad=15,
    )
    plt.xlabel('Jumlah Terjual (Pcs)', fontsize=10)
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig('boxplot_terjual.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # 2. HISTOGRAM DISTRIBUSI PENJUALAN
    # =====================================================================
    plt.figure(figsize=(10, 6))
    sns.histplot(df['terjual_clean'], kde=True, color='skyblue')
    plt.axvline(
        mean_val,
        color='red',
        linestyle='--',
        label=f'Mean (Rata-rata): {mean_val:.0f}',
    )
    plt.axvline(
        median_val,
        color='green',
        linestyle='-',
        label=f'Median: {median_val:.0f}',
    )
    plt.title(
        'Sebaran Data Penjualan (Membuktikan Data Skewed ke Kanan)',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Jumlah Terjual')
    plt.ylabel('Jumlah Produk')
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig('histogram_penjualan.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # 3. COUNTPLOT KATEGORI TARGET
    # =====================================================================
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x='label_target',
        hue='label_target',
        palette='viridis',
        legend=False,
    )
    plt.title(
        'Perbandingan Jumlah Produk Berdasarkan Threshold Penjualan',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    plt.xlabel('Kategori Penjualan', fontsize=12)
    plt.ylabel('Jumlah Produk', fontsize=12)
    plt.xticks(
        ticks=[0, 1],
        labels=[
            'Penjualan kurang dari 1000 pcs',
            'Penjualan ≥ 1000 pcs',
        ],
    )

    total = len(df)
    heights = [p.get_height() for p in ax.patches]
    max_height = max(heights) if heights else 0

    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.annotate(
            f'{int(height)} ({percentage:.1f}%)',
            (p.get_x() + p.get_width() / 2.0, height),
            ha='center',
            va='center',
            xytext=(0, 8),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
        )

    plt.ylim(0, max_height * 1.15)
    plt.tight_layout()
    if save_plots:
        plt.savefig(
            'countplot_kategori_target.png', dpi=300, bbox_inches='tight'
        )
    plt.close()

    # =====================================================================
    # 4. WORDCLOUD SEBELUM DAN SESUDAH PREPROCESSING
    # =====================================================================
    raw_text_combined = ' '.join(df['text_original'].dropna().astype(str))
    clean_text_combined = ' '.join(df['text_clean'].dropna().astype(str))

    wc_original = WordCloud(
        width=1000,
        height=800,
        background_color='white',
        colormap='Greens_r',
        max_words=100,
    ).generate(raw_text_combined)

    wc_clean = WordCloud(
        width=1000,
        height=800,
        background_color='white',
        colormap='Reds_r',
        max_words=100,
    ).generate(clean_text_combined)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(wc_original, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title(
        'Sebelum Preprocessing', fontsize=18, fontweight='bold', pad=15
    )

    axes[1].imshow(wc_clean, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title(
        'Sesudah Preprocessing', fontsize=18, fontweight='bold', pad=15
    )

    plt.subplots_adjust(wspace=0.12)
    if save_plots:
        plt.savefig(
            'wordcloud_comparison_close.png', dpi=300, bbox_inches='tight'
        )
    plt.close()

    # =====================================================================
    # 5. TOP 10 KATA TERBANYAK
    # =====================================================================
    all_tokens_final = [
        token for sublist in df['step_4_stopword'] for token in sublist
    ]
    most_common_10 = Counter(all_tokens_final).most_common(10)
    words, counts = zip(*most_common_10)

    plt.figure(figsize=(8, 5))
    plt.barh(words, counts, color='teal', height=0.7)
    plt.xlabel('Frekuensi Muncul', fontsize=10)
    plt.ylabel('Kata', fontsize=10)
    plt.title('Top 10 Kata Terbanyak Setelah Preprocessing', fontsize=12, pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for i, v in enumerate(counts):
        plt.text(
            v + 10,
            i,
            f' {v:,}',
            color='black',
            va='center',
            fontweight='semibold',
            fontsize=9,
        )

    plt.tight_layout()
    if save_plots:
        plt.savefig('top_10_words.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # 6. TF-IDF WORDCLOUD & TOP 10 FEATURE RANKING
    # =====================================================================
    tfidf = TfidfVectorizer(max_features=500)
    X_tfidf_matrix = tfidf.fit_transform(df['text_clean'])
    nama_fitur = tfidf.get_feature_names_out()
    total_skor = X_tfidf_matrix.sum(axis=0).A1

    frekuensi_kata = dict(zip(nama_fitur, total_skor))

    wordcloud_tfidf = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='Dark2',
        collocations=False,
        prefer_horizontal=0.85,
        max_words=100,
    ).generate_from_frequencies(frekuensi_kata)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_tfidf, interpolation='bilinear')
    plt.axis('off')
    plt.title(
        'WordCloud Fitur Teks Hasil Pembobotan TF-IDF',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    plt.tight_layout()
    if save_plots:
        plt.savefig('wordcloud_tfidf_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # 7. DISTRIBUSI DATA SPLITTING & SMOTE (JIKA DATA SPLIT TERSEDIA)
    # =====================================================================
    if data_splits is not None:
        categories = [
            'Data Latih\n(Sebelum SMOTE)',
            'Data Latih\n(Sesudah SMOTE)',
            'Data Uji\n(Tanpa SMOTE)',
        ]

        kelas_0 = [
            int(np.sum(np.array(data_splits['y_train']) == 0)),
            int(np.sum(np.array(data_splits['y_train_res']) == 0)),
            int(np.sum(np.array(data_splits['y_test']) == 0)),
        ]

        kelas_1 = [
            int(np.sum(np.array(data_splits['y_train']) == 1)),
            int(np.sum(np.array(data_splits['y_train_res']) == 1)),
            int(np.sum(np.array(data_splits['y_test']) == 1)),
        ]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(
            x - width / 2,
            kelas_0,
            width,
            label='Penjualan < 1000 pcs (Kelas 0)',
            color='#4C72B0',
            edgecolor='black',
            linewidth=0.8,
        )
        rects2 = ax.bar(
            x + width / 2,
            kelas_1,
            width,
            label='Penjualan ≥ 1000 pcs (Kelas 1)',
            color='#DD8452',
            edgecolor='black',
            linewidth=0.8,
        )

        ax.set_ylabel(
            'Jumlah Produk (Sampel)', fontsize=12, fontweight='bold'
        )
        ax.set_title(
            'Distribusi Kelas Target Pada Tahap Data Splitting dan Oversampling (SMOTE)',
            fontsize=13,
            fontweight='bold',
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight='bold')
        ax.legend(
            loc='upper right',
            frameon=True,
            facecolor='white',
            edgecolor='none',
            shadow=True,
        )

        max_val_smote = max(max(kelas_0), max(kelas_1))
        ax.set_ylim(0, max_val_smote * 1.15)

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(
                f'{height}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
            )

        plt.tight_layout()
        if save_plots:
            plt.savefig(
                'Gambar_Distribusi_Splitting_dan_SMOTE.png',
                dpi=300,
                bbox_inches='tight',
            )
        plt.close()


if __name__ == '__main__':
    # Test eksekusi file secara mandiri
    from data_preprocessing import preprocess_data

    file_path = '97ca96d3-e858-4d82-bcc1-c7a5d86376f0.csv'
    df_clean = preprocess_data(file_path)

    visualize_eda(df_clean)
    print('✅ Proses Visualisasi EDA Selesai! Semua grafik tersimpan.')