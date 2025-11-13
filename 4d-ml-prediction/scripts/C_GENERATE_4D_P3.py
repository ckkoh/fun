"""
C_GENERATE_4D_P3 - Phase 3 Production 4D Number Generator

Phase 3A: Statistical + Pattern-Based Approaches

Methods:
1. Statistical Frequency: Pure statistical analysis (no ML)
2. Pattern Mining: Rule-based with domain knowledge

Advantages over Phase 1 & 2:
- No training required (instant predictions)
- No overfitting (statistics-based)
- Highly interpretable
- Fast execution (<1 second)
- Naturally handles class imbalance

Usage:
    python C_GENERATE_4D_P3.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from database.db_manager import FourDDatabaseManager
from models.phase3_models import StatisticalFrequencyPredictor, PatternMiningPredictor


def get_last_n_months_data(db: FourDDatabaseManager, months: int = 6) -> pd.DataFrame:
    """Get training data from the last N months."""
    current_date = datetime.now()
    start_date = current_date - timedelta(days=months * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    current_date_str = current_date.strftime('%Y-%m-%d')

    all_draws = db.get_all_draws()
    filtered_draws = [d for d in all_draws if start_date_str <= d['draw_date'] < current_date_str]

    return pd.DataFrame(filtered_draws) if filtered_draws else pd.DataFrame()


def main():
    print("=" * 100)
    print("C_GENERATE_4D_P3 - PHASE 3 STATISTICAL + PATTERN-BASED GENERATOR")
    print("=" * 100)
    print()

    # Setup
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    db = FourDDatabaseManager()

    # Get historical data
    current_date = datetime.now()
    training_months = 6

    print(f"📅 Current Date:       {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Analysis Window:    Last {training_months} months")
    print()

    print("Loading historical data...")
    historical_df = get_last_n_months_data(db, months=training_months)

    if historical_df.empty:
        print("❌ ERROR: No historical data found.")
        return

    print(f"✓ Loaded {len(historical_df)} draws")
    print(f"  Date range: {historical_df['draw_date'].min()} to {historical_df['draw_date'].max()}")
    print()

    # Get latest draw info
    latest_draw = historical_df.iloc[-1]
    print(f"📌 Latest Draw:")
    print(f"   Date:       {latest_draw['draw_date']}")
    print(f"   1st Prize:  {latest_draw['first_prize']}")
    print(f"   2nd Prize:  {latest_draw['second_prize']}")
    print(f"   3rd Prize:  {latest_draw['third_prize']}")
    print()

    # Method 1: Statistical Frequency
    print("=" * 100)
    print("METHOD 1: STATISTICAL FREQUENCY ANALYSIS")
    print("=" * 100)
    print()
    print("Analyzing historical frequencies, recency, position patterns...")
    print()

    stat_predictor = StatisticalFrequencyPredictor(
        freq_weight=0.3,
        recency_weight=0.3,
        position_weight=0.2,
        pattern_weight=0.2
    )

    stat_predictions = stat_predictor.predict_top_k(
        historical_df,
        current_date.strftime('%Y-%m-%d'),
        k=10
    )

    print("🎯 TOP 10 PREDICTIONS (Statistical Frequency):")
    print()
    print("   " + "─" * 80)
    print(f"   {'Rank':<6} │ {'Number':<8} │ {'Score':<15} │ {'Bar Chart':<30}")
    print("   " + "─" * 80)

    for rank, (number, score) in enumerate(stat_predictions, 1):
        bar = '█' * int(score * 50)
        print(f"   {rank:<6} │ {number:<8} │ {score:>6.4f} ({score:.2%})  │ {bar}")

    print("   " + "─" * 80)
    print()

    # Method 2: Pattern Mining
    print("=" * 100)
    print("METHOD 2: PATTERN MINING")
    print("=" * 100)
    print()
    print("Mining patterns: palindromes, sequences, digit pairs, sum ranges...")
    print()

    pattern_predictor = PatternMiningPredictor()
    pattern_predictions = pattern_predictor.predict_top_k(historical_df, k=10)

    print("🎯 TOP 10 PREDICTIONS (Pattern Mining):")
    print()
    print("   " + "─" * 80)
    print(f"   {'Rank':<6} │ {'Number':<8} │ {'Confidence':<15} │ {'Bar Chart':<30}")
    print("   " + "─" * 80)

    for rank, (number, conf) in enumerate(pattern_predictions, 1):
        bar = '█' * int(conf * 50)
        print(f"   {rank:<6} │ {number:<8} │ {conf:>6.4f} ({conf:.2%})  │ {bar}")

    print("   " + "─" * 80)
    print()

    # Combined recommendation
    print("=" * 100)
    print("COMBINED RECOMMENDATIONS")
    print("=" * 100)
    print()

    # Merge predictions with weighted average
    combined_scores = {}

    # Add statistical predictions (weight 0.5)
    for number, score in stat_predictions:
        combined_scores[number] = combined_scores.get(number, 0) + 0.5 * score

    # Add pattern predictions (weight 0.5)
    for number, conf in pattern_predictions:
        combined_scores[number] = combined_scores.get(number, 0) + 0.5 * conf

    # Sort by combined score
    combined_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print("🎯 TOP 10 COMBINED PREDICTIONS:")
    print()
    print("   " + "─" * 80)
    print(f"   {'Rank':<6} │ {'Number':<8} │ {'Combined Score':<15} │ {'Bar Chart':<30}")
    print("   " + "─" * 80)

    for rank, (number, score) in enumerate(combined_predictions, 1):
        bar = '█' * int(score * 50)
        print(f"   {rank:<6} │ {number:<8} │ {score:>6.4f} ({score:.2%})  │ {bar}")

    print("   " + "─" * 80)
    print()

    # Top recommendation analysis
    top_number, top_score = combined_predictions[0]

    print("=" * 100)
    print("TOP RECOMMENDATION")
    print("=" * 100)
    print()
    print(f"   🎲 Number:         {top_number}")
    print(f"   📊 Combined Score: {top_score:.4f} ({top_score:.2%})")
    print()

    # Analyze top number
    digits = [int(d) for d in top_number]
    digit_sum = sum(digits)
    unique_digits = len(set(top_number))
    is_palindrome = top_number == top_number[::-1]

    print("   📊 Number Characteristics:")
    print(f"      Sum of Digits:    {digit_sum}")
    print(f"      Unique Digits:    {unique_digits}/4")
    print(f"      Palindrome:       {'Yes' if is_palindrome else 'No'}")
    print(f"      Even Digits:      {sum(1 for d in digits if d % 2 == 0)}/4")
    print()

    # Check if top number appeared recently
    if top_number in stat_predictor.number_last_seen:
        last_seen = stat_predictor.number_last_seen[top_number]
        days_ago = (pd.to_datetime(current_date) - pd.to_datetime(last_seen)).days
        appearances = stat_predictor.number_frequencies[top_number]
        print(f"   📈 Historical Performance:")
        print(f"      Total Appearances: {appearances}")
        print(f"      Last Seen:         {last_seen} ({days_ago} days ago)")
    else:
        print(f"   📈 Historical Performance:")
        print(f"      This number has never appeared in the dataset")
        print(f"      (Cold number - could be due for appearance)")

    print()

    # Save predictions
    output_file = f"data/predictions/prediction_P3_{current_date.strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("data/predictions", exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("C_GENERATE_4D_P3 - PHASE 3 PREDICTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated:     {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis Data: {len(historical_df)} draws from last {training_months} months\n")
        f.write(f"Date Range:    {historical_df['draw_date'].min()} to {historical_df['draw_date'].max()}\n")
        f.write(f"Methodology:   Phase 3 (Statistical + Pattern Mining)\n\n")

        f.write(f"TOP RECOMMENDATION:\n")
        f.write(f"  Number:         {top_number}\n")
        f.write(f"  Combined Score: {top_score:.4f} ({top_score:.2%})\n\n")

        f.write(f"TOP 10 COMBINED PREDICTIONS:\n")
        for rank, (number, score) in enumerate(combined_predictions, 1):
            f.write(f"  {rank:2d}. {number} - {score:.4f} ({score:.2%})\n")

        f.write(f"\nTOP 10 STATISTICAL:\n")
        for rank, (number, score) in enumerate(stat_predictions, 1):
            f.write(f"  {rank:2d}. {number} - {score:.4f}\n")

        f.write(f"\nTOP 10 PATTERN MINING:\n")
        for rank, (number, conf) in enumerate(pattern_predictions, 1):
            f.write(f"  {rank:2d}. {number} - {conf:.4f}\n")

    print(f"✓ Prediction saved to: {output_file}")
    print()

    # Performance notes
    print("=" * 100)
    print("⚡ PERFORMANCE ADVANTAGES")
    print("=" * 100)
    print()
    print("   Phase 3 vs Phase 1 & 2:")
    print("   ✅ No training required (instant predictions)")
    print("   ✅ No overfitting to sparse signals")
    print("   ✅ Highly interpretable (clear statistical logic)")
    print("   ✅ Fast execution (<1 second)")
    print("   ✅ Naturally handles extreme class imbalance")
    print()
    print("   Prediction Strategy:")
    print("   • Statistical: Frequency + Recency + Position + Pattern")
    print("   • Pattern Mining: Active patterns from recent trends")
    print("   • Combined: Weighted average of both methods")
    print()

    print("=" * 100)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 100)
    print()
    print("   Phase 3 Methodology:")
    print("   • Statistical analysis without ML")
    print("   • Pattern-based rule mining")
    print("   • No guarantee of predictive power")
    print()
    print("   4D Lottery Reality:")
    print("   • Win probability: 0.23% (23 in 10,000)")
    print("   • Outcomes are inherently random")
    print("   • Past patterns don't guarantee future results")
    print()
    print("   Please gamble responsibly and within your means.")
    print("   This is an educational demonstration of alternative methods.")
    print()
    print("=" * 100)
    print("✅ PHASE 3 PREDICTION GENERATION COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
