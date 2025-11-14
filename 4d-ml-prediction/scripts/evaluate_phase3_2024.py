"""
Phase 3 Evaluation on 2024 4D Draws

Evaluate Phase 3 (Statistical + Pattern-Based) generator on all 2024 draws.
Track wins and losses with $1 bet per draw.
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
from models.phase3_models import StatisticalFrequencyPredictor, PatternMiningPredictor

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
    """Check if prediction matches any winning number."""
    if predicted == actual_draw['first_prize']:
        return True, 'first', PRIZE_VALUES_BIG['first']
    if predicted == actual_draw['second_prize']:
        return True, 'second', PRIZE_VALUES_BIG['second']
    if predicted == actual_draw['third_prize']:
        return True, 'third', PRIZE_VALUES_BIG['third']

    for i in range(1, 11):
        if actual_draw.get(f'starter_{i}') == predicted:
            return True, 'starter', PRIZE_VALUES_BIG['starter']

    for i in range(1, 11):
        if actual_draw.get(f'consolation_{i}') == predicted:
            return True, 'consolation', PRIZE_VALUES_BIG['consolation']

    return False, 'no_prize', 0


def main():
    print("=" * 100)
    print("PHASE 3 EVALUATION - 2024 4D DRAWS")
    print("=" * 100)
    print()

    # Setup
    logging.basicConfig(level=logging.ERROR)
    db = FourDDatabaseManager()

    # Get 2024 draws
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    test_draws = db.get_draws_in_date_range(start_date, end_date)

    if not test_draws:
        print("❌ ERROR: No draws found for 2024")
        print("   Please ensure data is collected for 2024")
        return

    test_df = pd.DataFrame(test_draws)
    print(f"📊 Found {len(test_df)} draws in 2024")
    print(f"   Date range: {test_df['draw_date'].min()} to {test_df['draw_date'].max()}")
    print()

    # Track results
    all_results = []

    print("=" * 100)
    print("INCREMENTAL PREDICTIONS - TRAIN → PREDICT → EVALUATE")
    print("=" * 100)
    print()

    for idx, test_row in test_df.iterrows():
        draw_date = test_row['draw_date']
        draw_number = test_row['draw_number']

        # Get training data (all draws before this one)
        training_df = get_training_data_up_to_date(db, draw_date)

        if len(training_df) < 10:
            print(f"⚠️  {draw_date}: Insufficient training data ({len(training_df)} draws). Skipping...")
            continue

        # Phase 3A: Statistical Frequency
        stat_predictor = StatisticalFrequencyPredictor(
            freq_weight=0.3,
            recency_weight=0.3,
            position_weight=0.2,
            pattern_weight=0.2
        )

        stat_predictions = stat_predictor.predict_top_k(
            training_df,
            draw_date,
            k=1
        )
        stat_number, stat_score = stat_predictions[0]

        # Phase 3B: Pattern Mining
        pattern_predictor = PatternMiningPredictor()
        pattern_predictions = pattern_predictor.predict_top_k(training_df, k=1)
        pattern_number, pattern_conf = pattern_predictions[0]

        # Phase 3C: Combined (weighted average)
        combined_scores = {}
        combined_scores[stat_number] = combined_scores.get(stat_number, 0) + 0.5 * stat_score
        combined_scores[pattern_number] = combined_scores.get(pattern_number, 0) + 0.5 * pattern_conf

        combined_number = max(combined_scores.items(), key=lambda x: x[1])[0]
        combined_score = combined_scores[combined_number]

        # Check if it's a winner
        is_win, prize_category, payout = check_prediction_win(combined_number, test_row)
        net_profit = payout - BET_COST

        # Display compact result
        win_marker = "🏆 WIN " if is_win else "   LOSS"
        prize_str = f"{prize_category.upper()}" if is_win else "No Prize"
        payout_str = f"${payout:,}".rjust(6) if is_win else "   $0"

        print(f"{win_marker} │ {draw_date} │ Pred: {combined_number} ({combined_score:.1%}) │ "
              f"1st: {test_row['first_prize']} │ Prize: {prize_str:12s} │ Payout: {payout_str} │ Net: ${net_profit:+4d}")

        # Store result
        all_results.append({
            'draw_number': draw_number,
            'draw_date': draw_date,
            'predicted': combined_number,
            'confidence': combined_score,
            'first_prize': test_row['first_prize'],
            'second_prize': test_row['second_prize'],
            'third_prize': test_row['third_prize'],
            'is_win': is_win,
            'prize_category': prize_category,
            'payout': payout,
            'bet_cost': BET_COST,
            'net_profit': net_profit,
            'training_size': len(training_df),
            'stat_pred': stat_number,
            'pattern_pred': pattern_number
        })

    # Summary statistics
    print()
    print("=" * 100)
    print("📈 SUMMARY STATISTICS - 2024")
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
            print(f"  {'No Prize':15s}: {count:3d} ({pct:5.1f}%) - ${payout:,} payout")
        else:
            prize_value = PRIZE_VALUES_BIG.get(category, 0)
            print(f"  {category.title():15s}: {count:3d} ({pct:5.1f}%) - ${payout:,} payout (${prize_value:,} each)")

    # Winning predictions detail
    if total_wins > 0:
        print()
        print("=" * 100)
        print("🏆 WINNING PREDICTIONS")
        print("=" * 100)
        print()

        winning_results = [r for r in all_results if r['is_win']]
        winning_results.sort(key=lambda x: x['payout'], reverse=True)

        print(f"{'Date':<12} │ {'Predicted':<8} │ {'1st Prize':<8} │ {'Prize Category':<15} │ {'Payout':<10} │ {'Net Profit':<12}")
        print("─" * 100)

        for r in winning_results:
            print(f"{r['draw_date']:<12} │ {r['predicted']:<8} │ {r['first_prize']:<8} │ "
                  f"{r['prize_category'].title():<15} │ ${r['payout']:<9,} │ ${r['net_profit']:+11,}")

    # Detailed results table
    print()
    print("=" * 100)
    print("📋 DETAILED RESULTS TABLE (All Predictions)")
    print("=" * 100)
    print()

    print(f"{'Date':<12} │ {'Predicted':<8} │ {'Confidence':<12} │ {'1st Prize':<8} │ {'Result':<15} │ {'Payout':<10} │ {'Net Profit':<12}")
    print("─" * 100)

    for r in all_results:
        result_str = r['prize_category'].title() if r['is_win'] else 'No Win'
        profit_str = f"${r['net_profit']:+,}"
        payout_str = f"${r['payout']:,}" if r['payout'] > 0 else "-"

        print(f"{r['draw_date']:<12} │ {r['predicted']:<8} │ {r['confidence']:>6.2%} ({r['confidence']:.4f}) │ "
              f"{r['first_prize']:<8} │ {result_str:<15} │ {payout_str:<10} │ {profit_str:<12}")

    # Monthly breakdown
    print()
    print("=" * 100)
    print("📅 MONTHLY BREAKDOWN - 2024")
    print("=" * 100)
    print()

    # Group by month
    monthly_results = {}
    for r in all_results:
        month = r['draw_date'][:7]  # YYYY-MM
        if month not in monthly_results:
            monthly_results[month] = []
        monthly_results[month].append(r)

    print(f"{'Month':<12} │ {'Predictions':<12} │ {'Wins':<8} │ {'Win Rate':<10} │ {'Net Profit':<12} │ {'ROI':<10}")
    print("─" * 100)

    for month in sorted(monthly_results.keys()):
        month_data = monthly_results[month]
        m_predictions = len(month_data)
        m_wins = sum(1 for r in month_data if r['is_win'])
        m_cost = m_predictions * BET_COST
        m_payout = sum(r['payout'] for r in month_data)
        m_profit = m_payout - m_cost
        m_win_rate = (m_wins / m_predictions) * 100 if m_predictions > 0 else 0
        m_roi = (m_profit / m_cost) * 100 if m_cost > 0 else 0

        print(f"{month:<12} │ {m_predictions:<12} │ {m_wins:>3}/{m_predictions:<3} │ {m_win_rate:>8.1f}% │ ${m_profit:>+10,} │ {m_roi:>+8.1f}%")

    # Comparison with random baseline
    random_win_probability = 23 / 10000  # 23 winning numbers out of 10,000
    random_expected_wins = total_predictions * random_win_probability
    random_expected_payout = random_expected_wins * ((PRIZE_VALUES_BIG['first'] + PRIZE_VALUES_BIG['second'] + PRIZE_VALUES_BIG['third'] +
                                                      10 * PRIZE_VALUES_BIG['starter'] + 10 * PRIZE_VALUES_BIG['consolation']) / 23)
    random_expected_profit = random_expected_payout - total_cost

    print()
    print("=" * 100)
    print("📊 COMPARISON WITH RANDOM BASELINE")
    print("=" * 100)
    print()
    print(f"Phase 3 Performance:")
    print(f"  Win Rate:     {win_rate:.2f}%")
    print(f"  Actual Wins:  {total_wins}")
    print(f"  Net Profit:   ${total_profit:+,}")
    print(f"  ROI:          {roi:+.1f}%")
    print()
    print(f"Random Baseline (selecting any 4D number):")
    print(f"  Win Rate:     {random_win_probability * 100:.2f}% (23 in 10,000)")
    print(f"  Expected Wins: {random_expected_wins:.2f}")
    print(f"  Expected Profit: ${random_expected_profit:+,.2f}")
    print(f"  Expected ROI: {(random_expected_profit / total_cost * 100):+.1f}%")
    print()

    if total_wins > random_expected_wins:
        improvement = ((total_wins - random_expected_wins) / random_expected_wins) * 100
        print(f"✅ Phase 3 won {total_wins} times vs {random_expected_wins:.2f} expected (Random)")
        print(f"   Improvement: +{improvement:.1f}% more wins than random")
    else:
        print(f"⚠️  Phase 3 won {total_wins} times vs {random_expected_wins:.2f} expected (Random)")

    print()
    print("=" * 100)
    print("✅ PHASE 3 EVALUATION COMPLETE - 2024")
    print("=" * 100)
    print()
    print("⚠️  DISCLAIMER: Results based on mock data for educational purposes.")
    print("   Real 4D lottery outcomes are random and cannot be reliably predicted.")
    print("   This analysis demonstrates Phase 3 methodology on historical data.")
    print()


if __name__ == "__main__":
    main()
