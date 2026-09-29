import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_models(models, data_splits, save_plots=True):
    """Fungsi untuk mengevaluasi model pada data uji, menghitung metrik performa,

    serta merekam dan menyimpan visualisasi Confusion Matrix dan Kurva ROC-AUC.
    """
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    model_gb = models['gradient_boosting']
    model_rf = models['random_forest']

    # 1. Prediksi Label & Probabilitas
    y_pred_gb = model_gb.predict(X_test)
    y_prob_gb = model_gb.predict_proba(X_test)[:, 1]

    y_pred_rf = model_rf.predict(X_test)
    y_prob_rf = model_rf.predict_proba(X_test)[:, 1]

    # 2. Kalkulasi Metrik Evaluasi
    metrics = {
        'gradient_boosting': {
            'accuracy': accuracy_score(y_test, y_pred_gb),
            'roc_auc': roc_auc_score(y_test, y_prob_gb),
            'report': classification_report(
                y_test,
                y_pred_gb,
                target_names=['Penjualan < 1000 pcs', 'Penjualan ≥ 1000 pcs'],
                output_dict=True,
            ),
            'y_pred': y_pred_gb,
            'y_prob': y_prob_gb,
        },
        'random_forest': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'roc_auc': roc_auc_score(y_test, y_prob_rf),
            'report': classification_report(
                y_test,
                y_pred_rf,
                target_names=['Penjualan < 1000 pcs', 'Penjualan ≥ 1000 pcs'],
                output_dict=True,
            ),
            'y_pred': y_pred_rf,
            'y_prob': y_prob_rf,
        },
    }

    # 3. Visualisasi Confusion Matrix
    label_names = ['< 1000 pcs', '≥ 1000 pcs']
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(
        cm_gb,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[0],
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    axes[0].set_title(
        'Confusion Matrix - Gradient Boosting',
        fontsize=11,
        fontweight='bold',
        pad=10,
    )
    axes[0].set_xlabel('Prediksi Model', fontsize=10)
    axes[0].set_ylabel('Aktual (Ground Truth)', fontsize=10)

    sns.heatmap(
        cm_rf,
        annot=True,
        fmt='d',
        cmap='Oranges',
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[1],
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    axes[1].set_title(
        'Confusion Matrix - Random Forest',
        fontsize=11,
        fontweight='bold',
        pad=10,
    )
    axes[1].set_xlabel('Prediksi Model', fontsize=10)
    axes[1].set_ylabel('Aktual (Ground Truth)', fontsize=10)

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            'Gambar_Confusion_Matrix_GB_RF.png', dpi=300, bbox_inches='tight'
        )
    plt.close()

    # 4. Visualisasi Kurva ROC-AUC
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
    roc_auc_gb = auc(fpr_gb, tpr_gb)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(6, 4))
    plt.plot(
        fpr_gb,
        tpr_gb,
        color='darkorange',
        lw=1.5,
        label=f'Gradient Boosting (AUC = {roc_auc_gb:.4f})',
    )
    plt.plot(
        fpr_rf,
        tpr_rf,
        color='forestgreen',
        lw=1.5,
        label=f'Random Forest (AUC = {roc_auc_rf:.4f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=9)
    plt.ylabel('True Positive Rate (TPR)', fontsize=9)
    plt.title('Kurva AUC-ROC: GB vs RF', fontsize=11, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_plots:
        plt.savefig('kurva_roc_auc.png', dpi=300, bbox_inches='tight')
    plt.close()

    return metrics


if __name__ == '__main__':
    # Test eksekusi file secara mandiri
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import create_features
    from src.train import train_models

    file_path = '97ca96d3-e858-4d82-bcc1-c7a5d86376f0.csv'
    df_clean = preprocess_data(file_path)
    X_final, y, _ = create_features(df_clean)
    models, data_splits = train_models(X_final, y)

    metrics = evaluate_models(models, data_splits)
    print(
        f"✅ Evaluasi Selesai! Accuracy GB: {metrics['gradient_boosting']['accuracy']*100:.2f}%, Accuracy RF: {metrics['random_forest']['accuracy']*100:.2f}%"
    )