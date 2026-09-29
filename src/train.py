from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def train_models(X, y):
    """Melakukan Data Splitting, SMOTE Oversampling, dan Hyperparameter Tuning

    menggunakan RandomizedSearchCV untuk model Gradient Boosting dan Random
    Forest.
    """
    # 1. Stratified Data Splitting (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Oversampling Kelas Minoritas dengan SMOTE (hanya pada data latih)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 3. Hyperparameter Tuning - Gradient Boosting
    param_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
    }

    gb_base = GradientBoostingClassifier(random_state=42)
    search_gb = RandomizedSearchCV(
        gb_base,
        param_distributions=param_gb,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
    )
    search_gb.fit(X_train_res, y_train_res)
    model_gb_final = search_gb.best_estimator_

    # 4. Hyperparameter Tuning - Random Forest
    param_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf_base = RandomForestClassifier(random_state=42)
    search_rf = RandomizedSearchCV(
        rf_base,
        param_distributions=param_rf,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
    )
    search_rf.fit(X_train_res, y_train_res)
    model_rf_final = search_rf.best_estimator_

    models = {
        'gradient_boosting': model_gb_final,
        'random_forest': model_rf_final,
    }

    data_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_res': X_train_res,
        'y_train_res': y_train_res,
    }

    return models, data_splits


if __name__ == '__main__':
    # Test eksekusi file secara mandiri
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import create_features

    file_path = '97ca96d3-e858-4d82-bcc1-c7a5d86376f0.csv'
    df_clean = preprocess_data(file_path)
    X_final, y, _ = create_features(df_clean)
    models, data_splits = train_models(X_final, y)
    print('✅ Pelatihan Model Berhasil!')