import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_importance(
    models, tfidf_model, kolom_numerik, top_n=15, save_plots=True
):
    """Mengekstrak dan memvisualisasikan fitur terpenting dari model Gradient

    Boosting dan Random Forest.
    """
    # 1. Dapatkan daftar nama fitur lengkap (Numerik + TF-IDF)
    nama_fitur_tfidf = list(tfidf_model.get_feature_names_out())
    semua_nama_fitur = np.array(kolom_numerik + nama_fitur_tfidf)

    # 2. Extract Feature Importance
    gb_importance = models['gradient_boosting'].feature_importances_
    rf_importance = models['random_forest'].feature_importances_

    # 3. Buat DataFrame
    df_importance = pd.DataFrame(
        {
            'Fitur': semua_nama_fitur,
            'Importance_GB': gb_importance,
            'Importance_RF': rf_importance,
        }
    )

    # 4. Visualisasi Top-N Feature Importance Gradient Boosting
    top_gb = df_importance.sort_values(
        by='Importance_GB', ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_gb, x='Importance_GB', y='Fitur', hue='Fitur', palette='Blues_r', legend=False
    )
    plt.title(
        f'Top {top_n} Feature Importance - Gradient Boosting',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Nilai Importance')
    plt.ylabel('Fitur')
    plt.tight_layout()
    if save_plots:
        plt.savefig(
            'feature_importance_gb.png', dpi=300, bbox_inches='tight'
        )
    plt.close()

    # 5. Visualisasi Top-N Feature Importance Random Forest
    top_rf = df_importance.sort_values(
        by='Importance_RF', ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_rf, x='Importance_RF', y='Fitur', hue='Fitur', palette='Oranges_r', legend=False
    )
    plt.title(
        f'Top {top_n} Feature Importance - Random Forest',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Nilai Importance')
    plt.ylabel('Fitur')
    plt.tight_layout()
    if save_plots:
        plt.savefig(
            'feature_importance_rf.png', dpi=300, bbox_inches='tight'
        )
    plt.close()

    return df_importance