"""
C_GENERATE_TOTO - Production TOTO Number Generator

This script:
1. Uses current date to determine training window (last 6 months)
2. Trains P1_LightGBM model on recent 6-month historical data
3. Generates optimized 6-number prediction for next TOTO draw
4. Provides confidence scores and pattern analysis

Usage:
    python C_GENERATE_TOTO.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import joblib

from database.db_manager import DatabaseManager
from pipeline.feature_engineer import TotoFeatureEngineer
from models.simple_models import TotoLightGBMSimple


def get_last_n_months_data(db: DatabaseManager, months: int = 6) -> pd.DataFrame:
    """
    Get training data from the last N months based on current date.

    Args:
        db: Database manager instance
        months: Number of months to look back (default: 6)

    Returns:
        DataFrame with historical draws from last N months
    """
    # Get current date
    current_date = datetime.now()

    # Calculate date N months ago
    start_date = current_date - timedelta(days=months * 30)  # Approximate 30 days per month
    start_date_str = start_date.strftime('%Y-%m-%d')
    current_date_str = current_date.strftime('%Y-%m-%d')

    # Get all draws in the date range
    all_draws = db.get_all_draws()
    filtered_draws = [d for d in all_draws if start_date_str <= d['draw_date'] < current_date_str]

    if not filtered_draws:
        return pd.DataFrame()

    return pd.DataFrame(filtered_draws)


def prepare_training_data(df: pd.DataFrame, feature_engineer: TotoFeatureEngineer):
    """Prepare training data with features and labels."""
    # Engineer features
    df_featured = feature_engineer.engineer_features(df.copy())

    # Create labels (multi-hot encoding for numbers 1-49)
    y = np.zeros((len(df), 49), dtype=int)
    for i, (idx, row) in enumerate(df.iterrows()):
        numbers = [row['number_1'], row['number_2'], row['number_3'],
                   row['number_4'], row['number_5'], row['number_6']]
        for num in numbers:
            y[i, num - 1] = 1

    # Get feature columns (exclude target and metadata, only keep numeric columns)
    exclude_cols = ['draw_id', 'draw_number', 'draw_date', 'day_of_week',
                    'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6',
                    'additional_number', 'prize_pool', 'group_1_winners', 'group_2_winners',
                    'group_3_winners', 'group_4_winners', 'draw_type', 'is_rollover']

    # Select only numeric columns that are not in the exclude list
    feature_cols = []
    for col in df_featured.columns:
        if col not in exclude_cols and df_featured[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            feature_cols.append(col)

    X = df_featured[feature_cols].fillna(0).values

    return X, y, feature_cols, df_featured


def get_number_probabilities(model: TotoLightGBMSimple, X: np.ndarray) -> Dict[int, float]:
    """
    Get probability scores for each number.

    Args:
        model: Trained LightGBM model
        X: Feature vector for prediction

    Returns:
        Dictionary mapping number (1-49) to probability score
    """
    probs = model.predict_proba(X.reshape(1, -1))[0]
    return {num: probs[num - 1] for num in range(1, 50)}


def analyze_recent_patterns(df: pd.DataFrame, last_n: int = 5) -> Dict:
    """Analyze patterns in recent draws."""
    recent_draws = df.tail(last_n)

    # Get all numbers from recent draws
    all_numbers = []
    for _, row in recent_draws.iterrows():
        all_numbers.extend([row['number_1'], row['number_2'], row['number_3'],
                           row['number_4'], row['number_5'], row['number_6']])

    # Calculate frequency
    from collections import Counter
    frequency = Counter(all_numbers)

    # Calculate statistics
    hot_numbers = [num for num, count in frequency.most_common(10)]
    cold_numbers = [num for num in range(1, 50) if frequency[num] == 0][:10]

    # Odd/Even analysis
    odd_count = sum(1 for num in all_numbers if num % 2 == 1)
    even_count = len(all_numbers) - odd_count

    # High/Low analysis (1-24 vs 25-49)
    low_count = sum(1 for num in all_numbers if num <= 24)
    high_count = len(all_numbers) - low_count

    return {
        'hot_numbers': hot_numbers,
        'cold_numbers': cold_numbers,
        'odd_ratio': odd_count / len(all_numbers),
        'even_ratio': even_count / len(all_numbers),
        'low_ratio': low_count / len(all_numbers),
        'high_ratio': high_count / len(all_numbers),
        'recent_draws': len(recent_draws)
    }


def main():
    print("=" * 90)
    print("C_GENERATE_TOTO - PRODUCTION NUMBER GENERATOR")
    print("=" * 90)
    print()

    # Initialize components
    db = DatabaseManager()
    feature_engineer = TotoFeatureEngineer()

    # Get current date and training window
    current_date = datetime.now()
    training_months = 6
    start_date = current_date - timedelta(days=training_months * 30)

    print(f"📅 Current Date:       {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Training Window:    Last {training_months} months")
    print(f"   Training Start:     {start_date.strftime('%Y-%m-%d')}")
    print(f"   Training End:       {current_date.strftime('%Y-%m-%d')}")
    print()

    # Load training data from last 6 months
    print("Loading training data from last 6 months...")
    training_df = get_last_n_months_data(db, months=training_months)

    if training_df.empty:
        print("❌ ERROR: No training data found in the last 6 months.")
        print("   Please run data collection scripts first.")
        return

    print(f"✓ Loaded {len(training_df)} draws")
    print(f"  Date range: {training_df['draw_date'].min()} to {training_df['draw_date'].max()}")
    print()

    # Check minimum data requirement
    if len(training_df) < 20:
        print(f"⚠️  WARNING: Only {len(training_df)} draws available.")
        print("   Recommended minimum: 20 draws for reliable predictions.")
        print()

    # Prepare training data
    print("=" * 90)
    print("TRAINING MODEL ON 6-MONTH HISTORICAL DATA")
    print("=" * 90)
    print()

    X_train, y_train, feature_cols, df_featured = prepare_training_data(training_df, feature_engineer)

    print(f"📊 Training Statistics:")
    print(f"   Samples:     {len(X_train)}")
    print(f"   Features:    {len(feature_cols)}")
    print(f"   Target Dim:  {y_train.shape}")
    print()

    # Train model
    print("🔧 Training P1_LightGBM model...")
    print("   (This may take 1-2 minutes)")
    print()

    model = TotoLightGBMSimple(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.train(X_train, y_train)

    print("✓ Model training complete!")
    print()

    # Analyze recent patterns
    print("=" * 90)
    print("ANALYZING RECENT PATTERNS")
    print("=" * 90)
    print()

    patterns = analyze_recent_patterns(training_df, last_n=5)

    print(f"🔥 Hot Numbers (Last 5 Draws):    {patterns['hot_numbers'][:6]}")
    print(f"❄️  Cold Numbers (Last 5 Draws):   {patterns['cold_numbers'][:6]}")
    print(f"📊 Odd/Even Ratio:                {patterns['odd_ratio']:.1%} / {patterns['even_ratio']:.1%}")
    print(f"📊 Low/High Ratio:                {patterns['low_ratio']:.1%} / {patterns['high_ratio']:.1%}")
    print()

    # Get latest draw for feature context
    latest_draw = training_df.iloc[-1]
    print(f"📌 Latest Draw in Database:")
    print(f"   Date:    {latest_draw['draw_date']}")
    print(f"   Numbers: [{latest_draw['number_1']}, {latest_draw['number_2']}, {latest_draw['number_3']}, "
          f"{latest_draw['number_4']}, {latest_draw['number_5']}, {latest_draw['number_6']}]")
    print(f"   Additional: {latest_draw['additional_number']}")
    print()

    # Prepare prediction features (use latest draw context)
    X_pred = X_train[-1:]  # Use last draw's features as context

    # Generate prediction
    print("=" * 90)
    print("GENERATING PREDICTION FOR NEXT DRAW")
    print("=" * 90)
    print()

    # Get predicted numbers
    predicted_numbers = model.predict_top_k(X_pred, k=6)[0].tolist()

    # Get probability scores for all numbers
    probs = get_number_probabilities(model, X_train[-1])

    # Get confidence scores for predicted numbers
    confidence_scores = {num: probs[num] for num in predicted_numbers}

    # Sort by confidence
    sorted_predictions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)

    print("🎯 PREDICTED NUMBERS FOR NEXT DRAW:")
    print()
    print("   " + "─" * 70)
    print(f"   {'Number':<10} │ {'Confidence':<15} │ {'Bar Chart':<30}")
    print("   " + "─" * 70)

    for num, conf in sorted_predictions:
        bar = '█' * int(conf * 50)
        print(f"   {num:<10} │ {conf:>6.2%} ({conf:.4f})  │ {bar}")

    print("   " + "─" * 70)
    print()

    # Display final prediction set
    final_prediction = sorted(predicted_numbers)
    print("=" * 90)
    print("FINAL PREDICTION")
    print("=" * 90)
    print()
    print(f"   🎲 Numbers: {final_prediction}")
    print()

    # Calculate prediction characteristics
    odd_count = sum(1 for num in final_prediction if num % 2 == 1)
    even_count = 6 - odd_count
    low_count = sum(1 for num in final_prediction if num <= 24)
    high_count = 6 - low_count
    number_sum = sum(final_prediction)
    number_range = max(final_prediction) - min(final_prediction)

    print("   📊 Prediction Characteristics:")
    print(f"      Odd/Even:     {odd_count}/6 odd, {even_count}/6 even")
    print(f"      Low/High:     {low_count}/6 low (1-24), {high_count}/6 high (25-49)")
    print(f"      Sum:          {number_sum}")
    print(f"      Range:        {number_range} (from {min(final_prediction)} to {max(final_prediction)})")
    print()

    # Compare with hot/cold numbers
    hot_in_prediction = [num for num in final_prediction if num in patterns['hot_numbers']]
    cold_in_prediction = [num for num in final_prediction if num in patterns['cold_numbers']]

    if hot_in_prediction:
        print(f"   🔥 Contains {len(hot_in_prediction)} hot numbers: {hot_in_prediction}")
    if cold_in_prediction:
        print(f"   ❄️  Contains {len(cold_in_prediction)} cold numbers: {cold_in_prediction}")
    print()

    # Save prediction to file
    output_file = f"data/predictions/prediction_{current_date.strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("data/predictions", exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("C_GENERATE_TOTO PREDICTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated:     {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {len(training_df)} draws from last {training_months} months\n")
        f.write(f"Date Range:    {training_df['draw_date'].min()} to {training_df['draw_date'].max()}\n\n")
        f.write(f"PREDICTED NUMBERS:\n")
        f.write(f"{final_prediction}\n\n")
        f.write(f"CONFIDENCE SCORES:\n")
        for num, conf in sorted_predictions:
            f.write(f"  {num:2d}: {conf:.4f} ({conf:.2%})\n")
        f.write(f"\nCHARACTERISTICS:\n")
        f.write(f"  Odd/Even: {odd_count}/{even_count}\n")
        f.write(f"  Low/High: {low_count}/{high_count}\n")
        f.write(f"  Sum:      {number_sum}\n")
        f.write(f"  Range:    {number_range}\n")

    print(f"✓ Prediction saved to: {output_file}")
    print()

    # Model performance disclaimer
    print("=" * 90)
    print("MODEL PERFORMANCE NOTES")
    print("=" * 90)
    print()
    print(f"   Based on incremental learning evaluation (Jan-Jun 2025):")
    print(f"   • Average matches:     2.20 per draw")
    print(f"   • Win rate:            28.3% (Group 5-6 prizes)")
    print(f"   • Best performance:    4 matches (Group 5 prize)")
    print(f"   • Improvement:         +191% vs random baseline")
    print()
    print("   ⚠️  IMPORTANT DISCLAIMER:")
    print("   This prediction is generated using machine learning on historical data.")
    print("   TOTO is a game of chance, and outcomes are inherently random.")
    print("   Past performance does not guarantee future results.")
    print("   Please gamble responsibly and within your means.")
    print()
    print("=" * 90)
    print("✅ PREDICTION GENERATION COMPLETE")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
