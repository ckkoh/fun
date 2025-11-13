"""
Incremental Learning Evaluation for 4D - January 2025

This script:
1. For each draw in January 2025:
   - Trains model on all historical data before that draw
   - Generates top prediction
   - Evaluates against actual results
   - Calculates winnings (assuming $1 Big Bet per prediction)
2. Tabulates all results with costs and winnings
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging

from database.db_manager import FourDDatabaseManager
from pipeline.feature_engineer import FourDFeatureEngineer
from models.digit_models import FourDLightGBM


# Prize values for $1 Big Bet
PRIZE_VALUES_BIG = {
    'first': 3000,
    'second': 2000,
    'third': 1000,
    'starter': 250,
    'consolation': 60
}

BET_COST = 1  # $1 per prediction


def get_training_data_up_to_date(db: FourDDatabaseManager, end_date: str) -> pd.DataFrame:
    """Get all historical data before a specific date."""
    all_draws = db.get_all_draws()
    filtered_draws = [d for d in all_draws if d['draw_date'] < end_date]

    if not filtered_draws:
        return pd.DataFrame()

    return pd.DataFrame(filtered_draws)


def check_prediction_win(predicted: str, actual_draw: Dict) -> Tuple[bool, str, int]:
    """
    Check if prediction matches any winning number.

    Returns:
        (is_win, prize_category, payout)
    """
    # Check first prize
    if predicted == actual_draw['first_prize']:
        return True, 'first', PRIZE_VALUES_BIG['first']

    # Check second prize
    if predicted == actual_draw['second_prize']:
        return True, 'second', PRIZE_VALUES_BIG['second']

    # Check third prize
    if predicted == actual_draw['third_prize']:
        return True, 'third', PRIZE_VALUES_BIG['third']

    # Check starter prizes
    for i in range(1, 11):
        key = f'starter_{i}'
        if key in actual_draw and actual_draw[key] == predicted:
            return True, 'starter', PRIZE_VALUES_BIG['starter']

    # Check consolation prizes
    for i in range(1, 11):
        key = f'consolation_{i}'
        if key in actual_draw and actual_draw[key] == predicted:
            return True, 'consolation', PRIZE_VALUES_BIG['consolation']

    return False, 'no_prize', 0


def main():
    print("=" * 100)
    print("4D INCREMENTAL LEARNING EVALUATION - JANUARY 2025")
    print("=" * 100)
    print()

    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    # Initialize components
    db = FourDDatabaseManager()
    feature_engineer = FourDFeatureEngineer()

    # Get January 2025 draws
    start_date = "2025-01-01"
    end_date = "2025-01-31"

    test_draws = db.get_draws_in_date_range(start_date, end_date)

    if not test_draws:
        print("❌ ERROR: No draws found for January 2025")
        print("   Please ensure data is collected for January 2025")
        return

    test_df = pd.DataFrame(test_draws)
    print(f"📊 Found {len(test_df)} draws in January 2025")
    print(f"   Date range: {test_df['draw_date'].min()} to {test_df['draw_date'].max()}")
    print()

    # Track results
    all_results = []

    print("=" * 100)
    print("INCREMENTAL PREDICTIONS - TRAIN → PREDICT → EVALUATE → RETRAIN")
    print("=" * 100)
    print()

    for idx, test_row in test_df.iterrows():
        draw_date = test_row['draw_date']
        draw_number = test_row['draw_number']

        print(f"{'─' * 100}")
        print(f"DRAW #{idx + 1} - {draw_date} (Draw Number: {draw_number})")
        print(f"{'─' * 100}")

        # Get training data (all draws before this one)
        training_df = get_training_data_up_to_date(db, draw_date)

        if len(training_df) < 20:
            print(f"⚠️  Insufficient training data ({len(training_df)} draws). Skipping...")
            print()
            continue

        print(f"📚 Training data: {len(training_df)} draws (up to {training_df['draw_date'].max()})")

        # Feature engineering
        df_featured = feature_engineer.engineer_features(training_df)

        # Prepare training data for each digit position
        y_dict = {}
        for pos in range(4):
            _, y, feature_cols = feature_engineer.prepare_training_data(df_featured, pos)
            y_dict[pos] = y

        # Get features
        X, _, feature_cols = feature_engineer.prepare_training_data(df_featured, 0)

        # Train model
        print(f"🔧 Training LightGBM on {len(X)} historical draws...")
        model = FourDLightGBM(n_estimators=100, max_depth=6, learning_rate=0.05)
        model.train(X, y_dict)

        # Generate prediction
        X_pred = X[-1:]  # Use latest draw features as context
        predictions = model.generate_4d_numbers(X_pred[0], top_k=1)
        predicted_number, confidence = predictions[0]

        # Check if it's a winner
        is_win, prize_category, payout = check_prediction_win(predicted_number, test_row)
        net_profit = payout - BET_COST

        # Display results
        print(f"🎯 Predicted:   {predicted_number} (Confidence: {confidence:.2%})")
        print(f"✓  1st Prize:   {test_row['first_prize']}")
        print(f"✓  2nd Prize:   {test_row['second_prize']}")
        print(f"✓  3rd Prize:   {test_row['third_prize']}")

        if is_win:
            print(f"🏆 WINNER! Prize: {prize_category.upper()} - ${payout:,}")
            print(f"💰 Net Profit:  ${net_profit:+,} (${payout:,} - ${BET_COST})")
        else:
            print(f"   Result:      No win")
            print(f"💰 Net Profit:  ${net_profit:+,} (-${BET_COST})")

        print()

        # Store result
        all_results.append({
            'draw_number': draw_number,
            'draw_date': draw_date,
            'predicted': predicted_number,
            'confidence': confidence,
            'first_prize': test_row['first_prize'],
            'second_prize': test_row['second_prize'],
            'third_prize': test_row['third_prize'],
            'is_win': is_win,
            'prize_category': prize_category,
            'payout': payout,
            'bet_cost': BET_COST,
            'net_profit': net_profit,
            'training_size': len(training_df)
        })

    # Summary statistics
    print()
    print("=" * 100)
    print("📈 SUMMARY STATISTICS - JANUARY 2025")
    print("=" * 100)
    print()

    if not all_results:
        print("No results to summarize.")
        return

    total_predictions = len(all_results)
    total_wins = sum(1 for r in all_results if r['is_win'])
    total_cost = total_predictions * BET_COST
    total_payout = sum(r['payout'] for r in all_results)
    total_profit = total_payout - total_cost
    win_rate = (total_wins / total_predictions) * 100 if total_predictions > 0 else 0
    roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0

    print(f"Total Predictions:   {total_predictions}")
    print(f"Total Wins:          {total_wins}/{total_predictions} ({win_rate:.1f}%)")
    print(f"Total Cost:          ${total_cost:,} ({total_predictions} × ${BET_COST})")
    print(f"Total Payout:        ${total_payout:,}")
    print(f"Net Profit/Loss:     ${total_profit:+,}")
    print(f"ROI:                 {roi:+.1f}%")
    print()

    # Prize breakdown
    print("Prize Breakdown:")
    prize_counts = {}
    prize_payouts = {}

    for r in all_results:
        cat = r['prize_category']
        prize_counts[cat] = prize_counts.get(cat, 0) + 1
        prize_payouts[cat] = prize_payouts.get(cat, 0) + r['payout']

    for category in ['first', 'second', 'third', 'starter', 'consolation', 'no_prize']:
        count = prize_counts.get(category, 0)
        payout = prize_payouts.get(category, 0)
        pct = (count / total_predictions) * 100

        if category == 'no_prize':
            print(f"  {'No Prize':15s}: {count:2d} ({pct:5.1f}%) - ${payout:,} payout")
        else:
            prize_value = PRIZE_VALUES_BIG.get(category, 0)
            print(f"  {category.title():15s}: {count:2d} ({pct:5.1f}%) - ${payout:,} payout (${prize_value:,} each)")

    print()

    # Detailed results table
    print("=" * 100)
    print("📋 DETAILED RESULTS TABLE")
    print("=" * 100)
    print()

    print(f"{'Date':<12} │ {'Predicted':<10} │ {'Confidence':<12} │ {'Result':<15} │ {'Payout':<10} │ {'Net Profit':<12}")
    print("─" * 100)

    for r in all_results:
        result_str = r['prize_category'].title() if r['is_win'] else 'No Win'
        profit_str = f"${r['net_profit']:+,}"
        payout_str = f"${r['payout']:,}" if r['payout'] > 0 else "-"

        print(f"{r['draw_date']:<12} │ {r['predicted']:<10} │ {r['confidence']:>6.2%} ({r['confidence']:.4f}) │ "
              f"{result_str:<15} │ {payout_str:<10} │ {profit_str:<12}")

    print()

    # Best performances
    if total_wins > 0:
        print("=" * 100)
        print("🏆 WINNING PREDICTIONS")
        print("=" * 100)
        print()

        winning_results = [r for r in all_results if r['is_win']]
        winning_results.sort(key=lambda x: x['payout'], reverse=True)

        for i, r in enumerate(winning_results, 1):
            print(f"{i}. {r['draw_date']} - {r['predicted']} → {r['prize_category'].upper()} Prize")
            print(f"   Payout: ${r['payout']:,} | Net Profit: ${r['net_profit']:+,} | Confidence: {r['confidence']:.2%}")
            print()

    # Comparison with random baseline
    random_win_probability = 23 / 10000  # 23 winning numbers out of 10,000
    random_expected_wins = total_predictions * random_win_probability
    random_expected_payout = random_expected_wins * ((PRIZE_VALUES_BIG['first'] + PRIZE_VALUES_BIG['second'] + PRIZE_VALUES_BIG['third'] +
                                                      10 * PRIZE_VALUES_BIG['starter'] + 10 * PRIZE_VALUES_BIG['consolation']) / 23)
    random_expected_profit = random_expected_payout - total_cost

    print("=" * 100)
    print("📊 COMPARISON WITH RANDOM BASELINE")
    print("=" * 100)
    print()
    print(f"ML Model Performance:")
    print(f"  Win Rate:     {win_rate:.2f}%")
    print(f"  Net Profit:   ${total_profit:+,}")
    print(f"  ROI:          {roi:+.1f}%")
    print()
    print(f"Random Baseline (selecting any 4D number):")
    print(f"  Win Rate:     {random_win_probability * 100:.2f}% ({23} in {10000})")
    print(f"  Expected Wins: {random_expected_wins:.2f}")
    print(f"  Expected Profit: ${random_expected_profit:+,.2f}")
    print(f"  Expected ROI: {(random_expected_profit / total_cost * 100):+.1f}%")
    print()

    if total_wins > random_expected_wins:
        improvement = ((total_wins - random_expected_wins) / random_expected_wins) * 100
        print(f"✅ ML model won {total_wins} times vs {random_expected_wins:.2f} expected (Random)")
        print(f"   Improvement: +{improvement:.1f}% more wins than random")
    else:
        print(f"⚠️  ML model won {total_wins} times vs {random_expected_wins:.2f} expected (Random)")

    print()
    print("=" * 100)
    print("✅ INCREMENTAL LEARNING EVALUATION COMPLETE")
    print("=" * 100)
    print()
    print("⚠️  DISCLAIMER: Results based on mock data for educational purposes.")
    print("   Real 4D lottery outcomes are random and cannot be reliably predicted.")
    print("   This analysis demonstrates the methodology, not guaranteed returns.")
    print()


if __name__ == "__main__":
    main()
