"""
C_GENERATE_4D_P1 - Phase 1 Production 4D Number Generator

This script uses Phase 1 methodology (inspired by TOTO P1_LightGBM):
1. Direct 4D number prediction (10,000-way classification)
2. Multi-hot encoding for all 23 winning numbers per draw
3. Simple LightGBM with default parameters
4. 6-month rolling training window
5. Top-K selection based on probability scores

Phase 1 treats each 4D number as an atomic entity, rather than
breaking it down into digit-level predictions.

Usage:
    python C_GENERATE_4D_P1.py
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
from models.phase1_models import FourDLightGBM_Phase1, prepare_phase1_training_data


def get_last_n_months_data(db: FourDDatabaseManager, months: int = 6) -> pd.DataFrame:
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
    start_date = current_date - timedelta(days=months * 30)  # Approximate
    start_date_str = start_date.strftime('%Y-%m-%d')
    current_date_str = current_date.strftime('%Y-%m-%d')

    # Get all draws in the date range
    all_draws = db.get_all_draws()
    filtered_draws = [d for d in all_draws if start_date_str <= d['draw_date'] < current_date_str]

    if not filtered_draws:
        return pd.DataFrame()

    return pd.DataFrame(filtered_draws)


def analyze_recent_patterns(df: pd.DataFrame, last_n: int = 5) -> Dict:
    """Analyze patterns in recent draws."""
    recent_draws = df.tail(last_n)

    # Get all first prize numbers from recent draws
    all_numbers = []
    for _, row in recent_draws.iterrows():
        all_numbers.append(row['first_prize'])

    # Calculate digit frequency
    from collections import Counter
    digit_freq = Counter()
    for num_str in all_numbers:
        for digit in num_str:
            digit_freq[digit] += 1

    # Pattern analysis
    sum_values = [sum(int(d) for d in num) for num in all_numbers]
    avg_sum = np.mean(sum_values)

    # Repeating digits
    repeating_count = sum(1 for num in all_numbers if len(set(num)) < 4)

    return {
        'recent_numbers': all_numbers,
        'digit_frequency': dict(digit_freq.most_common()),
        'avg_sum': avg_sum,
        'repeating_ratio': repeating_count / len(all_numbers),
        'recent_draws': len(recent_draws)
    }


def calculate_expected_value(number: str, confidence: float, bet_type: str = 'big') -> float:
    """
    Calculate expected value for a 4D prediction.

    Args:
        number: 4D number (0000-9999)
        confidence: Model confidence score (0-1)
        bet_type: 'big' or 'small'

    Returns:
        Expected value in dollars
    """
    if bet_type == 'big':
        # Big Bet prizes
        prizes = {
            'first': 3000,
            'second': 2000,
            'third': 1000,
            'starter': 250,  # 10 numbers
            'consolation': 60  # 10 numbers
        }
        # Probability of winning any prize (23 out of 10,000)
        win_prob = 23 / 10000
        # Weighted average prize
        avg_prize = (prizes['first'] + prizes['second'] + prizes['third'] +
                     10 * prizes['starter'] + 10 * prizes['consolation']) / 23
    else:
        # Small Bet prizes (top 3 only)
        prizes = {
            'first': 4500,
            'second': 3500,
            'third': 2500
        }
        win_prob = 3 / 10000
        avg_prize = (prizes['first'] + prizes['second'] + prizes['third']) / 3

    # Expected value = (win probability × average prize) - bet cost
    bet_cost = 1  # $1 bet
    ev = (win_prob * confidence * avg_prize) - bet_cost

    return ev


def main():
    print("=" * 100)
    print("C_GENERATE_4D_P1 - PHASE 1 PRODUCTION NUMBER GENERATOR")
    print("=" * 100)
    print()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

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

    # Prepare training data using Phase 1 methodology
    print("=" * 100)
    print("PREPARING TRAINING DATA - PHASE 1 METHODOLOGY")
    print("=" * 100)
    print()
    print("Phase 1 Approach:")
    print("  • Direct 4D number prediction (10,000-way classification)")
    print("  • Multi-hot encoding: 23 winning numbers per draw")
    print("  • Each 4D number treated as atomic entity")
    print("  • Simple LightGBM with default parameters")
    print()

    X_train, y_train, feature_cols = prepare_phase1_training_data(training_df, feature_engineer)

    print(f"📊 Training Statistics:")
    print(f"   Samples:       {len(X_train)}")
    print(f"   Features:      {len(feature_cols)}")
    print(f"   Target Shape:  {y_train.shape}")
    print(f"   Active Numbers: {np.sum(y_train.sum(axis=0) > 0)} / 10,000")
    print()

    # Train Phase 1 model
    print("=" * 100)
    print("TRAINING PHASE 1 MODEL")
    print("=" * 100)
    print()
    print("🔧 Training P1_LightGBM on historical data...")
    print("   (This may take 2-3 minutes due to 10,000 binary classifiers)")
    print()

    model = FourDLightGBM_Phase1(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05
    )

    model.train(X_train, y_train, feature_names=feature_cols)

    print()
    print("✓ Phase 1 model training complete!")
    print()

    # Analyze recent patterns
    print("=" * 100)
    print("ANALYZING RECENT PATTERNS")
    print("=" * 100)
    print()

    patterns = analyze_recent_patterns(training_df, last_n=5)

    print(f"🔢 Recent First Prizes (Last 5 Draws):")
    for num in patterns['recent_numbers']:
        print(f"   {num}")
    print()
    print(f"📊 Digit Frequency (Last 5 Draws):")
    for digit, count in sorted(patterns['digit_frequency'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * count
        print(f"   {digit}: {bar} ({count})")
    print()
    print(f"📊 Average Sum:  {patterns['avg_sum']:.1f}")
    print(f"📊 Repeating Digits: {patterns['repeating_ratio']:.1%}")
    print()

    # Get latest draw for feature context
    latest_draw = training_df.iloc[-1]
    print(f"📌 Latest Draw in Database:")
    print(f"   Date:       {latest_draw['draw_date']}")
    print(f"   1st Prize:  {latest_draw['first_prize']}")
    print(f"   2nd Prize:  {latest_draw['second_prize']}")
    print(f"   3rd Prize:  {latest_draw['third_prize']}")
    print()

    # Prepare prediction features (use latest draw context)
    X_pred = X_train[-1:]  # Use last draw's features as context

    # Generate prediction
    print("=" * 100)
    print("GENERATING PREDICTIONS FOR NEXT DRAW")
    print("=" * 100)
    print()

    # Get top-10 predicted numbers with confidence scores
    predictions = model.predict_top_k(X_pred, k=10)[0]

    print("🎯 TOP 10 PREDICTED 4D NUMBERS:")
    print()
    print("   " + "─" * 80)
    print(f"   {'Rank':<6} │ {'Number':<8} │ {'Confidence':<15} │ {'Expected Value':<15} │ {'Bar Chart':<20}")
    print("   " + "─" * 80)

    for rank, (number, confidence) in enumerate(predictions, 1):
        ev_big = calculate_expected_value(number, confidence, 'big')
        bar = '█' * int(confidence * 50)
        ev_str = f"${ev_big:+.2f}" if ev_big != 0 else " $0.00"

        print(f"   {rank:<6} │ {number:<8} │ {confidence:>6.2%} ({confidence:.4f}) │ {ev_str:>15} │ {bar}")

    print("   " + "─" * 80)
    print()

    # Display top recommendation
    top_number, top_confidence = predictions[0]
    print("=" * 100)
    print("TOP RECOMMENDATION")
    print("=" * 100)
    print()
    print(f"   🎲 Number:      {top_number}")
    print(f"   📊 Confidence:  {top_confidence:.2%}")
    print()

    # Calculate characteristics for top prediction
    digit_sum = sum(int(d) for d in top_number)
    unique_digits = len(set(top_number))
    is_palindrome = top_number == top_number[::-1]
    is_sequential = all(int(top_number[i+1]) - int(top_number[i]) == 1 for i in range(3))

    print("   📊 Characteristics:")
    print(f"      Sum of Digits:    {digit_sum}")
    print(f"      Unique Digits:    {unique_digits}/4")
    print(f"      Palindrome:       {'Yes' if is_palindrome else 'No'}")
    print(f"      Sequential:       {'Yes' if is_sequential else 'No'}")
    print()

    # Expected value analysis
    ev_big = calculate_expected_value(top_number, top_confidence, 'big')
    ev_small = calculate_expected_value(top_number, top_confidence, 'small')

    print("   💰 Expected Value Analysis (per $1 bet):")
    print(f"      Big Bet:    ${ev_big:+.2f}")
    print(f"      Small Bet:  ${ev_small:+.2f}")
    print()

    if ev_big > 0:
        print("   ✅ Positive expected value on Big Bet!")
    else:
        print("   ⚠️  Negative expected value (as expected for lottery)")

    print()

    # Save prediction to file
    output_file = f"data/predictions/prediction_P1_{current_date.strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("data/predictions", exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("C_GENERATE_4D_P1 - PHASE 1 PREDICTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated:     {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {len(training_df)} draws from last {training_months} months\n")
        f.write(f"Date Range:    {training_df['draw_date'].min()} to {training_df['draw_date'].max()}\n")
        f.write(f"Methodology:   Phase 1 (Direct 4D number classification)\n\n")

        f.write(f"TOP RECOMMENDATION:\n")
        f.write(f"  Number:      {top_number}\n")
        f.write(f"  Confidence:  {top_confidence:.4f} ({top_confidence:.2%})\n\n")

        f.write(f"TOP 10 PREDICTIONS:\n")
        for rank, (number, confidence) in enumerate(predictions, 1):
            ev = calculate_expected_value(number, confidence, 'big')
            f.write(f"  {rank:2d}. {number} - {confidence:.4f} ({confidence:.2%}) - EV: ${ev:+.2f}\n")

        f.write(f"\nCHARACTERISTICS:\n")
        f.write(f"  Sum: {digit_sum}\n")
        f.write(f"  Unique Digits: {unique_digits}\n")
        f.write(f"  Palindrome: {'Yes' if is_palindrome else 'No'}\n")

    print(f"✓ Prediction saved to: {output_file}")
    print()

    # Disclaimer
    print("=" * 100)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 100)
    print()
    print("   Phase 1 Methodology:")
    print("   • Direct 4D number classification (10,000-way)")
    print("   • Trained on limited historical data")
    print("   • Simple baseline approach without optimization")
    print()
    print("   4D Lottery Reality:")
    print("   • Win probability: 0.23% (23 in 10,000)")
    print("   • Outcomes are inherently random")
    print("   • Past performance does not guarantee future results")
    print()
    print("   Please gamble responsibly and within your means.")
    print("   This is an educational demonstration of ML methodology.")
    print()
    print("=" * 100)
    print("✅ PHASE 1 PREDICTION GENERATION COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
