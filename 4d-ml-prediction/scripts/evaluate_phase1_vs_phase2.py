"""
Phase 1 vs Phase 2 Model Comparison - February to June 2025

This script compares two 4D prediction methodologies:

Phase 1 (P1_LightGBM):
- Direct 4D number classification (10,000-way)
- Multi-hot encoding for all 23 winning numbers
- Simple LightGBM with default parameters
- Treats each number as atomic entity

Phase 2 (Digit-by-Digit):
- 4 separate digit position classifiers (4 × 10-way)
- Predicts each digit independently
- Combines digit probabilities
- More granular approach

Evaluation:
- Incremental learning: Train on all data before each draw
- Compare top predictions from each model
- Track wins, confidence scores, and performance
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import warnings

from database.db_manager import FourDDatabaseManager
from pipeline.feature_engineer import FourDFeatureEngineer
from models.phase1_models import FourDLightGBM_Phase1, prepare_phase1_training_data
from models.digit_models import FourDLightGBM

# Prize values for $1 Big Bet
PRIZE_VALUES_BIG = {
    'first': 3000,
    'second': 2000,
    'third': 1000,
    'starter': 250,
    'consolation': 60
}

BET_COST = 1

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


def train_phase1_model(training_df: pd.DataFrame, feature_engineer: FourDFeatureEngineer):
    """Train Phase 1 model (direct 4D classification)."""
    X, y, feature_cols = prepare_phase1_training_data(training_df, feature_engineer)

    model = FourDLightGBM_Phase1(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(X, y, feature_names=feature_cols)

    return model, X, feature_cols


def train_phase2_model(training_df: pd.DataFrame, feature_engineer: FourDFeatureEngineer):
    """Train Phase 2 model (digit-by-digit)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_featured = feature_engineer.engineer_features(training_df.copy())

    # Prepare training data for each digit position
    y_dict = {}
    for pos in range(4):
        _, y, feature_cols = feature_engineer.prepare_training_data(df_featured, pos)
        y_dict[pos] = y

    X, _, feature_cols = feature_engineer.prepare_training_data(df_featured, 0)

    model = FourDLightGBM(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(X, y_dict)

    return model, X, feature_cols


def main():
    print("=" * 100)
    print("PHASE 1 vs PHASE 2 MODEL COMPARISON - FEBRUARY TO JUNE 2025")
    print("=" * 100)
    print()

    # Setup
    logging.basicConfig(level=logging.ERROR)
    db = FourDDatabaseManager()
    feature_engineer = FourDFeatureEngineer()

    # Results storage
    all_results = {
        'P1_LightGBM': [],
        'P2_DigitByDigit': []
    }

    draw_info = []

    # Evaluate each month
    for start_date, end_date, month_name in MONTHS:
        print(f"\n{'=' * 100}")
        print(f"📅 EVALUATING {month_name.upper()} 2025")
        print(f"{'=' * 100}\n")

        test_draws = db.get_draws_in_date_range(start_date, end_date)

        if not test_draws:
            print(f"⚠️  No draws found for {month_name}")
            continue

        test_df = pd.DataFrame(test_draws)
        print(f"📊 Found {len(test_df)} draws in {month_name} 2025\n")

        for idx, test_row in test_df.iterrows():
            draw_date = test_row['draw_date']
            draw_number = test_row['draw_number']

            # Get training data
            training_df = get_training_data_up_to_date(db, draw_date)

            if len(training_df) < 20:
                print(f"⚠️  {draw_date}: Insufficient training data. Skipping...")
                continue

            print(f"Draw {draw_date} (Training: {len(training_df)} draws)")

            # Train both models
            print("  Training P1 (10,000-way)...", end=" ")
            p1_model, p1_X, p1_features = train_phase1_model(training_df, feature_engineer)
            p1_predictions = p1_model.predict_top_k(p1_X[-1:], k=1)[0]
            p1_number, p1_confidence = p1_predictions[0]
            print(f"✓ Pred: {p1_number} ({p1_confidence:.2%})")

            print("  Training P2 (4×10-way)...", end=" ")
            p2_model, p2_X, p2_features = train_phase2_model(training_df, feature_engineer)
            p2_predictions = p2_model.generate_4d_numbers(p2_X[-1], top_k=1)
            p2_number, p2_confidence = p2_predictions[0]
            print(f"✓ Pred: {p2_number} ({p2_confidence:.2%})")

            # Check results
            p1_win, p1_category, p1_payout = check_prediction_win(p1_number, test_row)
            p2_win, p2_category, p2_payout = check_prediction_win(p2_number, test_row)

            # Display results
            actual_1st = test_row['first_prize']
            p1_marker = "🏆" if p1_win else "  "
            p2_marker = "🏆" if p2_win else "  "

            print(f"  Actual 1st: {actual_1st}")
            print(f"  {p1_marker} P1: {'WIN' if p1_win else 'LOSS':4s} - ${p1_payout:,}")
            print(f"  {p2_marker} P2: {'WIN' if p2_win else 'LOSS':4s} - ${p2_payout:,}")
            print()

            # Store results
            draw_info.append({
                'month': month_name,
                'draw_date': draw_date,
                'draw_number': draw_number,
                'actual_1st': actual_1st,
                'actual_2nd': test_row['second_prize'],
                'actual_3rd': test_row['third_prize']
            })

            all_results['P1_LightGBM'].append({
                'draw_date': draw_date,
                'predicted': p1_number,
                'confidence': p1_confidence,
                'is_win': p1_win,
                'category': p1_category,
                'payout': p1_payout,
                'net_profit': p1_payout - BET_COST,
                'training_size': len(training_df)
            })

            all_results['P2_DigitByDigit'].append({
                'draw_date': draw_date,
                'predicted': p2_number,
                'confidence': p2_confidence,
                'is_win': p2_win,
                'category': p2_category,
                'payout': p2_payout,
                'net_profit': p2_payout - BET_COST,
                'training_size': len(training_df)
            })

    # Overall Summary
    print("\n" + "=" * 100)
    print("📊 OVERALL COMPARISON - FEBRUARY TO JUNE 2025")
    print("=" * 100)
    print()

    for model_name in ['P1_LightGBM', 'P2_DigitByDigit']:
        results = all_results[model_name]

        if not results:
            continue

        total_predictions = len(results)
        total_wins = sum(1 for r in results if r['is_win'])
        total_payout = sum(r['payout'] for r in results)
        total_cost = total_predictions * BET_COST
        total_profit = total_payout - total_cost
        win_rate = (total_wins / total_predictions) * 100 if total_predictions > 0 else 0
        roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in results])

        phase_label = "PHASE 1 (Direct 4D Classification)" if model_name == 'P1_LightGBM' else "PHASE 2 (Digit-by-Digit)"

        print(f"🏅 {phase_label}")
        print(f"{'─' * 100}")
        print(f"  Predictions:       {total_predictions}")
        print(f"  Wins:              {total_wins}/{total_predictions} ({win_rate:.1f}%)")
        print(f"  Avg Confidence:    {avg_confidence:.2%}")
        print(f"  Total Cost:        ${total_cost:,}")
        print(f"  Total Payout:      ${total_payout:,}")
        print(f"  Net Profit:        ${total_profit:+,}")
        print(f"  ROI:               {roi:+.1f}%")

        # Prize breakdown
        if total_wins > 0:
            prize_counts = {}
            for r in results:
                if r['is_win']:
                    prize_counts[r['category']] = prize_counts.get(r['category'], 0) + 1

            print(f"\n  Winning Categories:")
            for category in ['first', 'second', 'third', 'starter', 'consolation']:
                count = prize_counts.get(category, 0)
                if count > 0:
                    print(f"    {category.title():12s}: {count} × ${PRIZE_VALUES_BIG[category]:,} = ${count * PRIZE_VALUES_BIG[category]:,}")

        print()

    # Head-to-head comparison
    print("\n" + "=" * 100)
    print("🔍 HEAD-TO-HEAD COMPARISON")
    print("=" * 100)
    print()

    p1_results = all_results['P1_LightGBM']
    p2_results = all_results['P2_DigitByDigit']

    if p1_results and p2_results:
        p1_wins = sum(1 for r in p1_results if r['is_win'])
        p2_wins = sum(1 for r in p2_results if r['is_win'])
        p1_avg_conf = np.mean([r['confidence'] for r in p1_results])
        p2_avg_conf = np.mean([r['confidence'] for r in p2_results])
        p1_profit = sum(r['net_profit'] for r in p1_results)
        p2_profit = sum(r['net_profit'] for r in p2_results)

        print(f"{'Metric':<25} │ {'Phase 1':<20} │ {'Phase 2':<20} │ {'Winner':<15}")
        print("─" * 100)
        print(f"{'Total Wins':<25} │ {p1_wins:<20} │ {p2_wins:<20} │ {('Phase 1' if p1_wins > p2_wins else 'Phase 2' if p2_wins > p1_wins else 'Tie'):<15}")
        print(f"{'Average Confidence':<25} │ {p1_avg_conf:<20.2%} │ {p2_avg_conf:<20.2%} │ {('Phase 1' if p1_avg_conf > p2_avg_conf else 'Phase 2'):<15}")
        print(f"{'Net Profit':<25} │ ${p1_profit:<19,} │ ${p2_profit:<19,} │ {('Phase 1' if p1_profit > p2_profit else 'Phase 2' if p2_profit > p1_profit else 'Tie'):<15}")

        # Monthly breakdown
        print("\n" + "=" * 100)
        print("📅 MONTHLY BREAKDOWN")
        print("=" * 100)
        print()

        print(f"{'Month':<12} │ {'P1 Wins':<10} │ {'P2 Wins':<10} │ {'P1 Conf':<12} │ {'P2 Conf':<12} │ {'Better Model':<15}")
        print("─" * 100)

        for month_name in ["February", "March", "April", "May", "June"]:
            p1_month = [r for r, d in zip(p1_results, draw_info) if d['month'] == month_name]
            p2_month = [r for r, d in zip(p2_results, draw_info) if d['month'] == month_name]

            if not p1_month:
                continue

            p1_m_wins = sum(1 for r in p1_month if r['is_win'])
            p2_m_wins = sum(1 for r in p2_month if r['is_win'])
            p1_m_conf = np.mean([r['confidence'] for r in p1_month])
            p2_m_conf = np.mean([r['confidence'] for r in p2_month])

            better = "Phase 1" if p1_m_wins > p2_m_wins else "Phase 2" if p2_m_wins > p1_m_wins else "Tie"

            print(f"{month_name:<12} │ {p1_m_wins:>3}/{len(p1_month):<5} │ {p2_m_wins:>3}/{len(p2_month):<5} │ {p1_m_conf:>10.2%} │ {p2_m_conf:>10.2%} │ {better:<15}")

        # Draw-by-draw comparison
        print("\n" + "=" * 100)
        print("📋 DRAW-BY-DRAW COMPARISON")
        print("=" * 100)
        print()

        print(f"{'Date':<12} │ {'Actual':<8} │ {'P1 Pred':<8} │ {'P1 Win':<8} │ {'P2 Pred':<8} │ {'P2 Win':<8}")
        print("─" * 100)

        for i, draw in enumerate(draw_info):
            p1_r = p1_results[i]
            p2_r = p2_results[i]

            p1_win_str = "✓" if p1_r['is_win'] else "✗"
            p2_win_str = "✓" if p2_r['is_win'] else "✗"

            print(f"{draw['draw_date']:<12} │ {draw['actual_1st']:<8} │ {p1_r['predicted']:<8} │ {p1_win_str:<8} │ {p2_r['predicted']:<8} │ {p2_win_str:<8}")

        # Conclusion
        print("\n" + "=" * 100)
        print("📝 CONCLUSIONS")
        print("=" * 100)
        print()

        print("Methodology Comparison:")
        print(f"  Phase 1: Direct 4D classification (10,000-way multi-output)")
        print(f"  Phase 2: Digit-by-digit prediction (4 × 10-way classification)")
        print()

        if p1_wins > p2_wins:
            print(f"✅ WINNER: Phase 1 (Direct Classification)")
            print(f"   Phase 1 won {p1_wins} times vs Phase 2's {p2_wins} times")
            print(f"   Advantage: +{p1_wins - p2_wins} wins ({((p1_wins - p2_wins) / len(p1_results) * 100):.1f}% improvement)")
        elif p2_wins > p1_wins:
            print(f"✅ WINNER: Phase 2 (Digit-by-Digit)")
            print(f"   Phase 2 won {p2_wins} times vs Phase 1's {p1_wins} times")
            print(f"   Advantage: +{p2_wins - p1_wins} wins ({((p2_wins - p1_wins) / len(p2_results) * 100):.1f}% improvement)")
        else:
            print(f"🤝 TIE: Both models performed equally")
            print(f"   Both won {p1_wins} times out of {len(p1_results)} predictions")

        print()
        print("Confidence Analysis:")
        if p1_avg_conf > p2_avg_conf:
            print(f"  Phase 1 had higher average confidence: {p1_avg_conf:.2%} vs {p2_avg_conf:.2%}")
        else:
            print(f"  Phase 2 had higher average confidence: {p2_avg_conf:.2%} vs {p1_avg_conf:.2%}")

    print("\n" + "=" * 100)
    print("⚠️  DISCLAIMER")
    print("=" * 100)
    print()
    print("Results based on mock 4D data for educational purposes.")
    print("Real 4D lottery outcomes are random and cannot be reliably predicted.")
    print("This comparison demonstrates methodological differences, not guaranteed returns.")
    print()


if __name__ == "__main__":
    main()
