"""Phase 2 Advanced Training with DNN, LSTM, Transformer, Optuna, and Time-Series CV."""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import optuna
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Installing...")

from database.db_manager import DatabaseManager
from pipeline.feature_engineer import TotoFeatureEngineer
from models.simple_models import TotoLightGBMSimple, TotoRandomForestSimple

if TORCH_AVAILABLE:
    from models.deep_learning_models import (
        TotoDNN, TotoLSTM, TotoTransformer,
        DeepLearningTrainer, train_dl_model
    )


def time_series_cv_split(n_samples, n_splits=3, test_size=0.15):
    """
    Time-series cross-validation split.

    Yields train/val indices for each fold.
    """
    total_size = n_samples
    test_samples = int(total_size * test_size)
    train_val_samples = total_size - test_samples

    # Calculate fold sizes
    fold_size = train_val_samples // (n_splits + 1)

    folds = []
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = fold_size * (i + 1)
        val_end = fold_size * (i + 2)

        train_indices = list(range(0, val_start))
        val_indices = list(range(val_start, val_end))

        folds.append((train_indices, val_indices))

    return folds


def optimize_lightgbm_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    """Optimize LightGBM hyperparameters with Optuna."""
    logger = logging.getLogger(__name__)
    logger.info("Optimizing LightGBM with Optuna...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        }

        model = TotoLightGBMSimple(**params, random_state=42)
        model.train(X_train, y_train)

        # Evaluate on validation
        probs = model.predict_proba(X_val)
        preds = np.argsort(probs, axis=1)[:, -6:]

        # Calculate matches
        matches = []
        for pred, true in zip(preds, y_val):
            true_nums = np.where(true == 1)[0]
            match_count = len(set(pred) & set(true_nums))
            matches.append(match_count)

        return np.mean(matches)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"  Best LightGBM params: {study.best_params}")
    logger.info(f"  Best val matches: {study.best_value:.2f}")

    return study.best_params


def optimize_rf_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    """Optimize Random Forest hyperparameters with Optuna."""
    logger = logging.getLogger(__name__)
    logger.info("Optimizing Random Forest with Optuna...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
        }

        model = TotoRandomForestSimple(**params, random_state=42)
        model.train(X_train, y_train)

        # Evaluate
        probs = model.predict_proba(X_val)
        preds = np.argsort(probs, axis=1)[:, -6:]

        matches = []
        for pred, true in zip(preds, y_val):
            true_nums = np.where(true == 1)[0]
            match_count = len(set(pred) & set(true_nums))
            matches.append(match_count)

        return np.mean(matches)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"  Best RF params: {study.best_params}")
    logger.info(f"  Best val matches: {study.best_value:.2f}")

    return study.best_params


def main():
    """Main Phase 2 training pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("PHASE 2 ADVANCED TRAINING - DL MODELS + OPTUNA + TIME-SERIES CV")
    print("=" * 80)

    # Check PyTorch
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Please install: pip install torch")
        logger.info("Training only classical ML models...")

    # 1. Load data
    logger.info("\n📦 Loading data from database...")
    db = DatabaseManager('data/toto.db')
    draws = db.get_all_draws()
    df_raw = pd.DataFrame(draws)

    logger.info(f"  Total draws: {len(df_raw)}")
    logger.info(f"  Date range: {df_raw['draw_date'].min()} to {df_raw['draw_date'].max()}")

    # Filter for 2023 training data
    df_2023 = df_raw[df_raw['draw_date'] < '2024-01-01']
    logger.info(f"  2023 training data: {len(df_2023)} draws")

    # 2. Engineer features
    logger.info("\n🔧 Engineering features...")
    engineer = TotoFeatureEngineer(lookback_windows=[5, 10, 20])
    df_features = engineer.engineer_features(df_raw)
    targets = engineer.prepare_target(df_raw)

    # Select feature columns
    exclude_cols = ['draw_id', 'draw_number', 'draw_date', 'day_of_week',
                   'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6',
                   'additional_number', 'prize_pool', 'group_1_winners', 'group_2_winners',
                   'group_3_winners', 'group_4_winners', 'draw_type', 'is_rollover',
                   'created_at', 'updated_at']

    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    logger.info(f"  Features: {len(feature_cols)}")

    # 3. Prepare data splits
    # Use 2023 for training, 2024 for validation, 2025-Jan for testing
    train_mask = df_raw['draw_date'] < '2024-01-01'
    val_mask = (df_raw['draw_date'] >= '2024-01-01') & (df_raw['draw_date'] < '2025-01-01')
    test_mask = df_raw['draw_date'] >= '2025-01-01'

    X_train = df_features[feature_cols][train_mask].fillna(0).values
    y_train = targets[train_mask]

    X_val = df_features[feature_cols][val_mask].fillna(0).values
    y_val = targets[val_mask]

    X_test = df_features[feature_cols][test_mask].fillna(0).values
    y_test = targets[test_mask]

    logger.info(f"\n📊 Data splits:")
    logger.info(f"  Train (2023): {X_train.shape[0]} samples")
    logger.info(f"  Val (2024):   {X_val.shape[0]} samples")
    logger.info(f"  Test (2025):  {X_test.shape[0]} samples")

    # 4. Time-Series Cross-Validation
    logger.info("\n🔄 Time-Series Cross-Validation Setup")
    cv_folds = time_series_cv_split(len(X_train), n_splits=3, test_size=0.15)
    logger.info(f"  CV Folds: {len(cv_folds)}")

    for i, (train_idx, val_idx) in enumerate(cv_folds, 1):
        logger.info(f"  Fold {i}: train={len(train_idx)}, val={len(val_idx)}")

    # 5. Train LightGBM with Optuna
    print("\n" + "=" * 80)
    logger.info("🚀 Training LightGBM with Optuna Optimization")
    print("=" * 80)

    try:
        best_lgb_params = optimize_lightgbm_optuna(
            X_train, y_train, X_val, y_val, n_trials=15
        )

        lgb_model = TotoLightGBMSimple(**best_lgb_params, random_state=42)
        lgb_model.train(X_train, y_train, feature_names=feature_cols)
        logger.info("✓ LightGBM training complete")

    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        # Fallback to default params
        lgb_model = TotoLightGBMSimple(n_estimators=100, max_depth=6, random_state=42)
        lgb_model.train(X_train, y_train, feature_names=feature_cols)

    # 6. Train Random Forest with Optuna
    print("\n" + "=" * 80)
    logger.info("🚀 Training Random Forest with Optuna Optimization")
    print("=" * 80)

    try:
        best_rf_params = optimize_rf_optuna(
            X_train, y_train, X_val, y_val, n_trials=15
        )

        rf_model = TotoRandomForestSimple(**best_rf_params, random_state=42)
        rf_model.train(X_train, y_train, feature_names=feature_cols)
        logger.info("✓ Random Forest training complete")

    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        rf_model = TotoRandomForestSimple(n_estimators=100, max_depth=10, random_state=42)
        rf_model.train(X_train, y_train, feature_names=feature_cols)

    # 7. Train Deep Learning Models (if PyTorch available)
    dl_models = {}

    if TORCH_AVAILABLE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"\n💻 Using device: {device}")

        # DNN
        print("\n" + "=" * 80)
        logger.info("🚀 Training Deep Neural Network")
        print("=" * 80)

        try:
            dnn_params = {
                'input_dim': X_train.shape[1],
                'hidden_dims': [256, 128, 64],
                'dropout': 0.3
            }
            dnn_trainer, dnn_history = train_dl_model(
                TotoDNN, dnn_params,
                X_train.astype(np.float32), y_train.astype(np.float32),
                X_val.astype(np.float32), y_val.astype(np.float32),
                epochs=50, batch_size=16, learning_rate=0.001,
                early_stopping_patience=10, device=device
            )
            dl_models['DNN'] = dnn_trainer
            logger.info("✓ DNN training complete")

        except Exception as e:
            logger.error(f"DNN training failed: {e}")

        # LSTM
        print("\n" + "=" * 80)
        logger.info("🚀 Training LSTM with Attention")
        print("=" * 80)

        try:
            # Reshape for LSTM (add sequence dimension)
            X_train_seq = X_train.astype(np.float32)[:, np.newaxis, :]
            X_val_seq = X_val.astype(np.float32)[:, np.newaxis, :]

            lstm_params = {
                'input_dim': X_train.shape[1],
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3
            }
            lstm_trainer, lstm_history = train_dl_model(
                TotoLSTM, lstm_params,
                X_train_seq, y_train.astype(np.float32),
                X_val_seq, y_val.astype(np.float32),
                epochs=50, batch_size=16, learning_rate=0.001,
                early_stopping_patience=10, device=device
            )
            dl_models['LSTM'] = lstm_trainer
            logger.info("✓ LSTM training complete")

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")

        # Transformer
        print("\n" + "=" * 80)
        logger.info("🚀 Training Transformer")
        print("=" * 80)

        try:
            X_train_seq = X_train.astype(np.float32)[:, np.newaxis, :]
            X_val_seq = X_val.astype(np.float32)[:, np.newaxis, :]

            transformer_params = {
                'input_dim': X_train.shape[1],
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1
            }
            transformer_trainer, transformer_history = train_dl_model(
                TotoTransformer, transformer_params,
                X_train_seq, y_train.astype(np.float32),
                X_val_seq, y_val.astype(np.float32),
                epochs=50, batch_size=16, learning_rate=0.0005,
                early_stopping_patience=10, device=device
            )
            dl_models['Transformer'] = transformer_trainer
            logger.info("✓ Transformer training complete")

        except Exception as e:
            logger.error(f"Transformer training failed: {e}")

    # 8. Save models
    logger.info("\n💾 Saving Phase 2 models...")
    model_dir = Path('models/checkpoints/phase2')
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lgb_model, model_dir / 'lightgbm_optimized.pkl')
    joblib.dump(rf_model, model_dir / 'rf_optimized.pkl')
    joblib.dump(feature_cols, model_dir / 'feature_cols.pkl')

    if dl_models:
        for name, trainer in dl_models.items():
            torch.save(trainer.model.state_dict(), model_dir / f'{name.lower()}_model.pth')

    logger.info(f"  Saved {2 + len(dl_models)} models to {model_dir}")

    # 9. Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 2 TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTrained Models:")
    print(f"  ✓ LightGBM (Optuna optimized)")
    print(f"  ✓ Random Forest (Optuna optimized)")
    if TORCH_AVAILABLE:
        for name in dl_models.keys():
            print(f"  ✓ {name}")

    print(f"\nTraining Data: 2023 ({X_train.shape[0]} draws)")
    print(f"Validation Data: 2024 ({X_val.shape[0]} draws)")
    print(f"Test Data: Jan 2025 ({X_test.shape[0]} draws)")

    print(f"\nModels saved to: {model_dir}/")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
