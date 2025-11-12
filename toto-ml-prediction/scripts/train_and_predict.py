"""Train models and generate predictions for next TOTO draw."""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

from database.db_manager import DatabaseManager
from pipeline.feature_engineer import TotoFeatureEngineer
from models.simple_models import (
    TotoRandomForestSimple,
    TotoLightGBMSimple,
    SimpleEnsemble,
    calculate_matches
)


def main():
    """Main training and prediction pipeline."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("TOTO ML PREDICTION SYSTEM - TRAINING & PREDICTION")
    print("=" * 70)

    # 1. Load data from database
    logger.info("Loading data from database...")
    db = DatabaseManager('data/toto.db')
    draws = db.get_all_draws()
    df_raw = pd.DataFrame(draws)

    logger.info(f"Loaded {len(df_raw)} draws from {df_raw['draw_date'].min()} to {df_raw['draw_date'].max()}")

    # 2. Engineer features
    logger.info("\nEngineering features...")
    engineer = TotoFeatureEngineer(lookback_windows=[5, 10, 20])
    df_features = engineer.engineer_features(df_raw)

    # Create targets
    targets = engineer.prepare_target(df_raw)

    logger.info(f"Features shape: {df_features.shape}")
    logger.info(f"Targets shape: {targets.shape}")

    # 3. Select feature columns (remove metadata)
    exclude_cols = ['draw_id', 'draw_number', 'draw_date', 'day_of_week',
                   'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6',
                   'additional_number', 'prize_pool', 'group_1_winners', 'group_2_winners',
                   'group_3_winners', 'group_4_winners', 'draw_type', 'is_rollover',
                   'created_at', 'updated_at']

    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # 4. Split data (time-series split: last 10 draws for testing)
    n_test = 10
    n_train = len(df_features) - n_test

    X_train = df_features[feature_cols].iloc[:n_train].fillna(0).values
    y_train = targets[:n_train]

    X_test = df_features[feature_cols].iloc[n_train:].fillna(0).values
    y_test = targets[n_train:]

    logger.info(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 5. Train Random Forest
    print("\n" + "=" * 70)
    logger.info("Training Random Forest Model...")
    print("=" * 70)

    rf_model = TotoRandomForestSimple(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.train(X_train, y_train, feature_names=feature_cols)

    # 6. Train LightGBM
    print("\n" + "=" * 70)
    logger.info("Training LightGBM Model...")
    print("=" * 70)

    lgb_model = TotoLightGBMSimple(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    lgb_model.train(X_train, y_train, feature_names=feature_cols)

    # 7. Create ensemble
    logger.info("\nCreating ensemble...")
    ensemble = SimpleEnsemble([rf_model, lgb_model])

    # 8. Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    rf_matches = []
    lgb_matches = []
    ensemble_matches = []

    test_draws = df_raw.iloc[n_train:].reset_index(drop=True)

    for i in range(len(X_test)):
        # Get actual numbers
        actual = [test_draws.iloc[i][f'number_{j}'] for j in range(1, 7)]

        # Predictions
        rf_pred = rf_model.predict_top_k(X_test[i:i+1], k=6)[0]
        lgb_pred = lgb_model.predict_top_k(X_test[i:i+1], k=6)[0]
        ensemble_pred = ensemble.predict_top_k(X_test[i:i+1], k=6)[0]

        # Calculate matches
        rf_match_count, rf_match_nums = calculate_matches(rf_pred, actual)
        lgb_match_count, lgb_match_nums = calculate_matches(lgb_pred, actual)
        ensemble_match_count, ensemble_match_nums = calculate_matches(ensemble_pred, actual)

        rf_matches.append(rf_match_count)
        lgb_matches.append(lgb_match_count)
        ensemble_matches.append(ensemble_match_count)

        print(f"\nDraw {test_draws.iloc[i]['draw_date']}")
        print(f"  Actual:    {sorted(actual)}")
        print(f"  RF:        {sorted(rf_pred.tolist())} [{rf_match_count} matches: {rf_match_nums}]")
        print(f"  LightGBM:  {sorted(lgb_pred.tolist())} [{lgb_match_count} matches: {lgb_match_nums}]")
        print(f"  Ensemble:  {sorted(ensemble_pred.tolist())} [{ensemble_match_count} matches: {ensemble_match_nums}]")

    # Summary statistics
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"\nRandom Forest:")
    print(f"  Average matches: {np.mean(rf_matches):.2f}")
    print(f"  Best: {np.max(rf_matches)} matches")
    print(f"  Distribution: {dict(zip(*np.unique(rf_matches, return_counts=True)))}")

    print(f"\nLightGBM:")
    print(f"  Average matches: {np.mean(lgb_matches):.2f}")
    print(f"  Best: {np.max(lgb_matches)} matches")
    print(f"  Distribution: {dict(zip(*np.unique(lgb_matches, return_counts=True)))}")

    print(f"\nEnsemble:")
    print(f"  Average matches: {np.mean(ensemble_matches):.2f}")
    print(f"  Best: {np.max(ensemble_matches)} matches")
    print(f"  Distribution: {dict(zip(*np.unique(ensemble_matches, return_counts=True)))}")

    print(f"\nRandom Baseline Expected: 0.73 matches")

    # 9. Generate prediction for next draw
    print("\n" + "=" * 70)
    print("PREDICTION FOR NEXT DRAW")
    print("=" * 70)

    # Use last draw's features
    X_next = df_features[feature_cols].iloc[-1:].fillna(0).values

    rf_next = rf_model.predict_top_k(X_next, k=6)[0]
    lgb_next = lgb_model.predict_top_k(X_next, k=6)[0]
    ensemble_next = ensemble.predict_top_k(X_next, k=6)[0]

    print(f"\nBased on data up to: {df_raw.iloc[-1]['draw_date']}")
    print(f"\nPredicted numbers for next draw:")
    print(f"\n  Random Forest:  {sorted(rf_next.tolist())}")
    print(f"  LightGBM:       {sorted(lgb_next.tolist())}")
    print(f"  Ensemble:       {sorted(ensemble_next.tolist())} ⭐ RECOMMENDED")

    # Get probability scores
    ensemble_probs = ensemble.predict_top_k(X_next, k=6)
    rf_probs = rf_model.predict_proba(X_next)[0]
    lgb_probs = lgb_model.predict_proba(X_next)[0]
    avg_probs = (rf_probs + lgb_probs) / 2

    print(f"\n  Top 10 numbers by probability:")
    top_10_idx = np.argsort(avg_probs)[-10:][::-1]
    for idx in top_10_idx:
        number = idx + 1
        prob = avg_probs[idx]
        print(f"    {number:2d}: {prob:.3f}")

    # 10. Save models
    print("\n" + "=" * 70)
    logger.info("Saving models...")

    model_dir = Path('models/checkpoints')
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, model_dir / 'rf_model.pkl')
    joblib.dump(lgb_model, model_dir / 'lgb_model.pkl')
    joblib.dump(ensemble, model_dir / 'ensemble.pkl')
    joblib.dump(feature_cols, model_dir / 'feature_cols.pkl')

    logger.info("Models saved successfully!")

    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 70)
    print("\nThis is a machine learning research project.")
    print("TOTO draws are random and cannot be reliably predicted.")
    print("These predictions are for educational purposes only.")
    print("Do not use for actual gambling. Always gamble responsibly.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
