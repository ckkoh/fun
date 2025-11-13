"""
C_GENERATE_4D - Production 4D Number Generator

This script:
1. Uses current date to determine training window (last 6 months)
2. Trains models on recent 6-month historical data
3. Generates optimized 4D number predictions
4. Provides confidence scores and pattern analysis

Usage:
    python C_GENERATE_4D.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

from database.db_manager import FourDDatabaseManager
from pipeline.feature_engineer import FourDFeatureEngineer
from models.digit_models import FourDLightGBM, FourDRandomForest, FourDEnsemble


# Prize structure
PRIZE_VALUES = {
    'first': {'big': 3000, 'small': 2000},
    'second': {'big': 2000, 'small': 1000},
    'third': {'big': 1000, 'small': 500},
    'starter': {'big': 250, 'small': 0},
    'consolation': {'big': 60, 'small': 0}
}


def get_last_n_months_data(db: FourDDatabaseManager, months: int = 6) -> pd.DataFrame:
    """Get training data from the last N months."""
    current_date = datetime.now()
    start_date = current_date - timedelta(days=months * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    current_date_str = current_date.strftime('%Y-%m-%d')

    all_draws = db.get_all_draws()
    filtered_draws = [d for d in all_draws if start_date_str <= d['draw_date'] < current_date_str]

    if not filtered_draws:
        return pd.DataFrame()

    return pd.DataFrame(filtered_draws)


def analyze_hot_cold_numbers(df: pd.DataFrame, last_n: int = 10) -> Dict:
    """Analyze hot and cold 4D numbers."""
    recent_draws = df.tail(last_n)

    # Get all first prize numbers
    first_prizes = recent_draws['first_prize'].tolist()

    # Analyze digit frequency per position
    hot_digits = {}
    for pos in range(4):
        digits = [int(num[pos]) for num in first_prizes if len(num) == 4]
        from collections import Counter
        freq = Counter(digits)
        hot_digits[pos] = freq.most_common(3)

    return {
        'recent_first_prizes': first_prizes,
        'hot_digits_by_position': hot_digits
    }


def calculate_expected_value(number: str, confidence: float, bet_type: str = 'big') -> float:
    """Calculate expected value for a prediction."""
    # Simplified EV calculation based on confidence
    # Assumes uniform distribution for prize categories given a hit

    if bet_type == 'big':
        # Can win any of 23 prizes
        ev = confidence * (
            PRIZE_VALUES['first'][bet_type] * 0.04 +  # 1 in 23
            PRIZE_VALUES['second'][bet_type] * 0.04 +
            PRIZE_VALUES['third'][bet_type] * 0.04 +
            PRIZE_VALUES['starter'][bet_type] * 0.43 +  # 10 in 23
            PRIZE_VALUES['consolation'][bet_type] * 0.43
        ) - 1  # Bet cost

    else:  # small
        # Can only win top 3
        ev = confidence * (
            PRIZE_VALUES['first'][bet_type] * 0.33 +
            PRIZE_VALUES['second'][bet_type] * 0.33 +
            PRIZE_VALUES['third'][bet_type] * 0.33
        ) - 1

    return ev


def main():
    print("=" * 90)
    print("C_GENERATE_4D - PRODUCTION 4D NUMBER GENERATOR")
    print("=" * 90)
    print()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize components
    db = FourDDatabaseManager()
    feature_engineer = FourDFeatureEngineer()

    # Get current date and training window
    current_date = datetime.now()
    training_months = 6
    start_date = current_date - timedelta(days=training_months * 30)

    print(f"📅 Current Date:       {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Training Window:    Last {training_months} months")
    print(f"   Training Start:     {start_date.strftime('%Y-%m-%d')}")
    print(f"   Training End:       {current_date.strftime('%Y-%m-%d')}")
    print()

    # Load training data
    print("Loading training data from last 6 months...")
    training_df = get_last_n_months_data(db, months=training_months)

    if training_df.empty:
        print("❌ ERROR: No training data found in the last 6 months.")
        print("   Please run data collection script first:")
        print("   python scripts/collect_data.py")
        return

    print(f"✓ Loaded {len(training_df)} draws")
    print(f"  Date range: {training_df['draw_date'].min()} to {training_df['draw_date'].max()}")
    print()

    # Check minimum data requirement
    if len(training_df) < 30:
        print(f"⚠️  WARNING: Only {len(training_df)} draws available.")
        print("   Recommended minimum: 30 draws for reliable predictions.")
        print()

    # Feature engineering
    print("=" * 90)
    print("ENGINEERING FEATURES")
    print("=" * 90)
    print()

    df_featured = feature_engineer.engineer_features(training_df)
    print(f"✓ Feature engineering complete. Features: {df_featured.shape[1]}")
    print()

    # Prepare training data for each digit position
    print("=" * 90)
    print("TRAINING MODELS ON 6-MONTH HISTORICAL DATA")
    print("=" * 90)
    print()

    # Prepare labels for all positions
    y_dict = {}
    for pos in range(4):
        _, y, feature_cols = feature_engineer.prepare_training_data(df_featured, pos)
        y_dict[pos] = y

    # Get features (same for all positions)
    X, _, feature_cols = feature_engineer.prepare_training_data(df_featured, 0)

    print(f"📊 Training Statistics:")
    print(f"   Samples:     {len(X)}")
    print(f"   Features:    {len(feature_cols)}")
    print(f"   Positions:   4 (thousands, hundreds, tens, ones)")
    print()

    # Train model
    print("🔧 Training LightGBM model...")
    print("   (This may take 1-2 minutes)")
    print()

    model = FourDLightGBM(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.train(X, y_dict)

    print("✓ Model training complete!")
    print()

    # Analyze recent patterns
    print("=" * 90)
    print("ANALYZING RECENT PATTERNS")
    print("=" * 90)
    print()

    patterns = analyze_hot_cold_numbers(training_df, last_n=10)

    print(f"🔥 Recent First Prizes (Last 10 Draws):")
    for i, num in enumerate(patterns['recent_first_prizes'][-5:], 1):
        print(f"   {num}")
    print()

    print(f"🔥 Hot Digits by Position (Last 10 Draws):")
    pos_names = ['Thousands', 'Hundreds', 'Tens', 'Ones']
    for pos in range(4):
        hot_digits = patterns['hot_digits_by_position'][pos]
        digits_str = ', '.join([f"{d}({c}×)" for d, c in hot_digits])
        print(f"   {pos_names[pos]:12s}: {digits_str}")
    print()

    # Get latest draw for context
    latest_draw = training_df.iloc[-1]
    print(f"📌 Latest Draw in Database:")
    print(f"   Date:        {latest_draw['draw_date']}")
    print(f"   1st Prize:   {latest_draw['first_prize']}")
    print(f"   2nd Prize:   {latest_draw['second_prize']}")
    print(f"   3rd Prize:   {latest_draw['third_prize']}")
    print()

    # Generate predictions
    print("=" * 90)
    print("GENERATING PREDICTIONS FOR NEXT DRAW")
    print("=" * 90)
    print()

    # Use latest draw features as context
    X_pred = X[-1:]

    # Generate top 10 predictions
    predictions = model.generate_4d_numbers(X_pred[0], top_k=10)

    print("🎯 TOP 10 PREDICTED 4D NUMBERS:")
    print()
    print("   " + "─" * 80)
    print(f"   {'Rank':<6} │ {'Number':<8} │ {'Confidence':<15} │ {'Expected Value (Big)':<20}")
    print("   " + "─" * 80)

    for rank, (number, confidence) in enumerate(predictions, 1):
        ev = calculate_expected_value(number, confidence, bet_type='big')
        bar = '█' * int(confidence * 50)
        print(f"   #{rank:<5} │ {number:<8} │ {confidence:>6.2%} ({confidence:.4f}) │ ${ev:>6.2f} │ {bar}")

    print("   " + "─" * 80)
    print()

    # Display top prediction details
    top_number, top_conf = predictions[0]
    print("=" * 90)
    print("RECOMMENDED PREDICTION")
    print("=" * 90)
    print()
    print(f"   🎲 4D Number:  {top_number}")
    print(f"   📊 Confidence: {top_conf:.2%}")
    print()

    # Analyze top prediction
    digits = [int(d) for d in top_number]
    print("   📊 Number Characteristics:")
    print(f"      Sum of digits:    {sum(digits)}")
    print(f"      Even digits:      {sum(1 for d in digits if d % 2 == 0)}/4")
    print(f"      Odd digits:       {sum(1 for d in digits if d % 2 == 1)}/4")
    print(f"      Has 8 or 9:       {'Yes' if 8 in digits or 9 in digits else 'No'}")
    print(f"      Is palindrome:    {'Yes' if top_number == top_number[::-1] else 'No'}")
    print()

    # Expected value analysis
    ev_big = calculate_expected_value(top_number, top_conf, 'big')
    ev_small = calculate_expected_value(top_number, top_conf, 'small')

    print("   💰 Expected Value Analysis:")
    print(f"      Big Bet ($1):     ${ev_big:+.2f}")
    print(f"      Small Bet ($1):   ${ev_small:+.2f}")
    print(f"      Recommended:      {'Big Bet' if ev_big > ev_small else 'Small Bet'}")
    print()

    # Save prediction to file
    output_dir = "data/predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/4d_prediction_{current_date.strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("C_GENERATE_4D PREDICTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated:     {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {len(training_df)} draws from last {training_months} months\n")
        f.write(f"Date Range:    {training_df['draw_date'].min()} to {training_df['draw_date'].max()}\n\n")
        f.write(f"TOP 10 PREDICTIONS:\n")
        for rank, (number, confidence) in enumerate(predictions, 1):
            ev = calculate_expected_value(number, confidence, 'big')
            f.write(f"  #{rank}: {number} (Confidence: {confidence:.4f}, EV: ${ev:+.2f})\n")
        f.write(f"\nRECOMMENDED: {top_number}\n")
        f.write(f"Confidence: {top_conf:.4f} ({top_conf:.2%})\n")
        f.write(f"Expected Value (Big): ${ev_big:+.2f}\n")
        f.write(f"Expected Value (Small): ${ev_small:+.2f}\n")

    print(f"✓ Prediction saved to: {output_file}")
    print()

    # Model performance disclaimer
    print("=" * 90)
    print("BETTING STRATEGY & DISCLAIMERS")
    print("=" * 90)
    print()
    print("   💡 Recommended Strategy:")
    print("      - Use Big Bet for higher win probability (23 winning numbers)")
    print("      - Use Small Bet for higher payouts (only top 3)")
    print("      - Consider betting on top 3-5 predictions to diversify")
    print("      - Set a budget and stick to it")
    print()
    print("   ⚠️  IMPORTANT DISCLAIMER:")
    print("      This prediction is generated using machine learning on historical data.")
    print("      4D lottery is a game of chance, and outcomes are inherently random.")
    print("      Past performance does not guarantee future results.")
    print("      Expected values are theoretical and assume uniform prize distribution.")
    print("      Please gamble responsibly and within your means.")
    print("      Only bet what you can afford to lose.")
    print()
    print("=" * 90)
    print("✅ PREDICTION GENERATION COMPLETE")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
