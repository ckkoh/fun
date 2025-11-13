"""
Incremental Learning Evaluation for 4D - February to June 2025

This script:
1. For each draw in Feb-Jun 2025:
   - Trains model on all historical data before that draw
   - Generates top prediction
   - Evaluates against actual results
   - Calculates winnings (assuming $1 Big Bet per prediction)
2. Tabulates results per month and overall summary
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

MONTHS = [
    ("2025-02-01", "2025-02-28", "February"),
    ("2025-03-01", "2025-03-31", "March"),
    ("2025-04-01", "2025-04-30", "April"),
    ("2025-05-01", "2025-05-31", "May"),
    ("2025-06-01", "2025-06-30", "June")
]


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


def print_monthly_summary(month_name: str, results: List[Dict]):
    """Print summary statistics for a single month."""
    if not results:
        print(f"\n⚠️  No results for {month_name}")
        return

    total_predictions = len(results)
    total_wins = sum(1 for r in results if r['is_win'])
    total_cost = total_predictions * BET_COST
    total_payout = sum(r['payout'] for r in results)
    total_profit = total_payout - total_cost
    win_rate = (total_wins / total_predictions) * 100 if total_predictions > 0 else 0
    roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0

    print(f"\n{'=' * 100}")
    print(f"📈 SUMMARY - {month_name.upper()} 2025")
    print(f"{'=' * 100}\n")

    print(f"Predictions:  {total_predictions}")
    print(f"Wins:         {total_wins}/{total_predictions} ({win_rate:.1f}%)")
    print(f"Cost:         ${total_cost:,}")
    print(f"Payout:       ${total_payout:,}")
    print(f"Net Profit:   ${total_profit:+,}")
    print(f"ROI:          {roi:+.1f}%")

    # Prize breakdown
    prize_counts = {}
    for r in results:
        cat = r['prize_category']
        prize_counts[cat] = prize_counts.get(cat, 0) + 1

    if total_wins > 0:
        print(f"\nWinning Categories:")
        for category in ['first', 'second', 'third', 'starter', 'consolation']:
            count = prize_counts.get(category, 0)
            if count > 0:
                print(f"  {category.title():12s}: {count} × ${PRIZE_VALUES_BIG[category]:,} = ${count * PRIZE_VALUES_BIG[category]:,}")


def evaluate_month(db: FourDDatabaseManager, feature_engineer: FourDFeatureEngineer,
                   start_date: str, end_date: str, month_name: str) -> List[Dict]:
    """Evaluate predictions for a single month."""
    print(f"\n{'=' * 100}")
    print(f"🗓️  EVALUATING {month_name.upper()} 2025")
    print(f"{'=' * 100}\n")

    # Get test draws for this month
    test_draws = db.get_draws_in_date_range(start_date, end_date)

    if not test_draws:
        print(f"⚠️  No draws found for {month_name} 2025")
        return []

    test_df = pd.DataFrame(test_draws)
    print(f"📊 Found {len(test_df)} draws ({test_df['draw_date'].min()} to {test_df['draw_date'].max()})")

    # Track results
    month_results = []

    for idx, test_row in test_df.iterrows():
        draw_date = test_row['draw_date']
        draw_number = test_row['draw_number']

        # Get training data (all draws before this one)
        training_df = get_training_data_up_to_date(db, draw_date)

        if len(training_df) < 20:
            print(f"⚠️  Draw {draw_date}: Insufficient training data. Skipping...")
            continue

        # Feature engineering (suppress output)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_featured = feature_engineer.engineer_features(training_df)

        # Prepare training data for each digit position
        y_dict = {}
        for pos in range(4):
            _, y, feature_cols = feature_engineer.prepare_training_data(df_featured, pos)
            y_dict[pos] = y

        # Get features
        X, _, feature_cols = feature_engineer.prepare_training_data(df_featured, 0)

        # Train model
        model = FourDLightGBM(n_estimators=100, max_depth=6, learning_rate=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.train(X, y_dict)

        # Generate prediction
        X_pred = X[-1:]  # Use latest draw features as context
        predictions = model.generate_4d_numbers(X_pred[0], top_k=1)
        predicted_number, confidence = predictions[0]

        # Check if it's a winner
        is_win, prize_category, payout = check_prediction_win(predicted_number, test_row)
        net_profit = payout - BET_COST

        # Display compact result
        win_indicator = "🏆 WIN " if is_win else "   LOSS"
        prize_str = f"${payout:,}".rjust(6) if is_win else "   $0"
        print(f"{win_indicator} │ {draw_date} │ Pred: {predicted_number} │ 1st: {test_row['first_prize']} │ "
              f"Payout: {prize_str} │ Net: ${net_profit:+3d} │ Conf: {confidence:>6.2%}")

        # Store result
        month_results.append({
            'month': month_name,
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

    return month_results


def print_overall_summary(all_results: List[Dict]):
    """Print comprehensive summary across all months."""
    print(f"\n\n{'=' * 100}")
    print(f"📊 OVERALL SUMMARY - FEBRUARY TO JUNE 2025")
    print(f"{'=' * 100}\n")

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
    print(f"ROI:                 {roi:+.1f}%\n")

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
            print(f"  {'No Prize':15s}: {count:3d} ({pct:5.1f}%) - ${payout:,} payout")
        else:
            prize_value = PRIZE_VALUES_BIG.get(category, 0)
            print(f"  {category.title():15s}: {count:3d} ({pct:5.1f}%) - ${payout:,} payout (${prize_value:,} each)")

    # Monthly breakdown
    print(f"\n{'=' * 100}")
    print(f"📅 MONTHLY BREAKDOWN")
    print(f"{'=' * 100}\n")

    print(f"{'Month':<12} │ {'Predictions':<12} │ {'Wins':<8} │ {'Win Rate':<10} │ {'Net Profit':<12} │ {'ROI':<10}")
    print("─" * 100)

    for month_name in ["February", "March", "April", "May", "June"]:
        month_results = [r for r in all_results if r['month'] == month_name]
        if not month_results:
            print(f"{month_name:<12} │ {'0':<12} │ {'0':<8} │ {'-':<10} │ {'$0':<12} │ {'-':<10}")
            continue

        m_predictions = len(month_results)
        m_wins = sum(1 for r in month_results if r['is_win'])
        m_cost = m_predictions * BET_COST
        m_payout = sum(r['payout'] for r in month_results)
        m_profit = m_payout - m_cost
        m_win_rate = (m_wins / m_predictions) * 100 if m_predictions > 0 else 0
        m_roi = (m_profit / m_cost) * 100 if m_cost > 0 else 0

        print(f"{month_name:<12} │ {m_predictions:<12} │ {m_wins:>3}/{m_predictions:<3} │ {m_win_rate:>6.1f}%   │ ${m_profit:>+10,} │ {m_roi:>+7.1f}%")

    # Winning predictions
    if total_wins > 0:
        print(f"\n{'=' * 100}")
        print(f"🏆 ALL WINNING PREDICTIONS")
        print(f"{'=' * 100}\n")

        winning_results = [r for r in all_results if r['is_win']]
        winning_results.sort(key=lambda x: x['payout'], reverse=True)

        for i, r in enumerate(winning_results, 1):
            print(f"{i}. {r['month']:<10} {r['draw_date']} - Predicted: {r['predicted']} → {r['prize_category'].upper()} Prize")
            print(f"   Payout: ${r['payout']:,} | Net Profit: ${r['net_profit']:+,} | Confidence: {r['confidence']:.2%}")
            print(f"   Actual Winners: 1st:{r['first_prize']}, 2nd:{r['second_prize']}, 3rd:{r['third_prize']}\n")

    # Comparison with random baseline
    random_win_probability = 23 / 10000  # 23 winning numbers out of 10,000
    random_expected_wins = total_predictions * random_win_probability
    random_expected_payout = random_expected_wins * ((PRIZE_VALUES_BIG['first'] + PRIZE_VALUES_BIG['second'] + PRIZE_VALUES_BIG['third'] +
                                                      10 * PRIZE_VALUES_BIG['starter'] + 10 * PRIZE_VALUES_BIG['consolation']) / 23)
    random_expected_profit = random_expected_payout - total_cost

    print(f"\n{'=' * 100}")
    print(f"📊 COMPARISON WITH RANDOM BASELINE")
    print(f"{'=' * 100}\n")
    print(f"ML Model Performance:")
    print(f"  Win Rate:     {win_rate:.2f}%")
    print(f"  Actual Wins:  {total_wins}")
    print(f"  Net Profit:   ${total_profit:+,}")
    print(f"  ROI:          {roi:+.1f}%\n")
    print(f"Random Baseline (selecting any 4D number):")
    print(f"  Win Rate:     {random_win_probability * 100:.2f}% (23 in 10,000)")
    print(f"  Expected Wins: {random_expected_wins:.2f}")
    print(f"  Expected Profit: ${random_expected_profit:+,.2f}")
    print(f"  Expected ROI: {(random_expected_profit / total_cost * 100):+.1f}%\n")

    if total_wins > random_expected_wins:
        improvement = ((total_wins - random_expected_wins) / random_expected_wins) * 100
        print(f"✅ ML model won {total_wins} times vs {random_expected_wins:.2f} expected (Random)")
        print(f"   Improvement: +{improvement:.1f}% more wins than random")
    else:
        print(f"⚠️  ML model won {total_wins} times vs {random_expected_wins:.2f} expected (Random)")


def main():
    print("=" * 100)
    print("4D INCREMENTAL LEARNING EVALUATION - FEBRUARY TO JUNE 2025")
    print("=" * 100)

    # Setup logging to reduce noise
    logging.basicConfig(level=logging.ERROR)

    # Initialize components
    db = FourDDatabaseManager()
    feature_engineer = FourDFeatureEngineer()

    # Store all results
    all_results = []

    # Evaluate each month
    for start_date, end_date, month_name in MONTHS:
        month_results = evaluate_month(db, feature_engineer, start_date, end_date, month_name)
        print_monthly_summary(month_name, month_results)
        all_results.extend(month_results)

    # Overall summary
    if all_results:
        print_overall_summary(all_results)
    else:
        print("\n⚠️  No results to summarize across all months.")

    print(f"\n{'=' * 100}")
    print("✅ INCREMENTAL LEARNING EVALUATION COMPLETE")
    print(f"{'=' * 100}\n")
    print("⚠️  DISCLAIMER: Results based on mock data for educational purposes.")
    print("   Real 4D lottery outcomes are random and cannot be reliably predicted.")
    print("   This analysis demonstrates the methodology, not guaranteed returns.\n")


if __name__ == "__main__":
    main()
