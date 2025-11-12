"""
Incremental Learning Evaluation - January to June 2025

This script implements an incremental learning approach:
1. Start with P1_LightGBM model trained on 2024 data
2. For each week in Jan-Jun 2025:
   - Generate predictions for upcoming draws
   - Evaluate against actual results
   - Retrain model with new data
   - Continue to next week

This simulates real-world usage where the model learns from new draws.
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


def calculate_matches_and_prize(predicted: List[int], actual: List[int], additional: int) -> Tuple[int, str, int]:
    """Calculate number of matches and prize group."""
    matches = len(set(predicted) & set(actual))
    has_additional = additional in predicted

    # Prize structure
    if matches == 6:
        prize_group = "Group 1"
        reward = 1000000
    elif matches == 5 and has_additional:
        prize_group = "Group 2"
        reward = 50000
    elif matches == 5:
        prize_group = "Group 3"
        reward = 1000
    elif matches == 4 and has_additional:
        prize_group = "Group 4"
        reward = 500
    elif matches == 4:
        prize_group = "Group 5"
        reward = 50
    elif matches == 3 and has_additional:
        prize_group = "Group 6"
        reward = 25
    elif matches == 3:
        prize_group = "Group 6"
        reward = 25
    else:
        prize_group = "No Prize"
        reward = 0

    return matches, prize_group, reward


def get_weekly_draws(db: DatabaseManager, start_date: str, end_date: str) -> pd.DataFrame:
    """Get all draws within a date range."""
    draws = db.get_draws_in_date_range(start_date, end_date)

    if not draws:
        return pd.DataFrame()

    return pd.DataFrame(draws)


def get_training_data_up_to_date(db: DatabaseManager, end_date: str) -> pd.DataFrame:
    """Get all historical data up to a specific date for training."""
    # Get all draws
    all_draws = db.get_all_draws()

    # Filter to only include draws before end_date
    filtered_draws = [d for d in all_draws if d['draw_date'] < end_date]

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

    # Get feature columns (exclude target and metadata, and only keep numeric columns)
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

    return X, y, feature_cols


def main():
    print("=" * 90)
    print("INCREMENTAL LEARNING EVALUATION - JANUARY TO JUNE 2025")
    print("=" * 90)
    print()

    # Initialize components
    db = DatabaseManager()
    feature_engineer = TotoFeatureEngineer()

    # Track all results
    all_results = []

    # Define date ranges for Jan-Jun 2025
    # TOTO draws are Monday and Thursday
    start_date = "2025-01-01"
    end_date = "2025-06-30"

    # Get all draws in Jan-Jun 2025 for testing
    test_draws_df = get_weekly_draws(db, start_date, end_date)

    if test_draws_df.empty:
        print("⚠️  No draws found for Jan-Jun 2025 period")
        return

    print(f"📊 Found {len(test_draws_df)} draws from {start_date} to {end_date}")
    print()

    # Initialize model (will retrain for each prediction)
    model = TotoLightGBMSimple(n_estimators=100, max_depth=6, learning_rate=0.05)

    # Process each draw incrementally
    print("=" * 90)
    print("INCREMENTAL PREDICTIONS - TRAIN → PREDICT → EVALUATE → RETRAIN")
    print("=" * 90)
    print()

    for idx, test_row in test_draws_df.iterrows():
        draw_date = test_row['draw_date']
        draw_number = test_row['draw_number']

        print(f"{'─' * 90}")
        print(f"DRAW #{idx + 1} - {draw_date} (Draw Number: {draw_number})")
        print(f"{'─' * 90}")

        # Step 1: Get all training data up to this draw (exclusive)
        training_df = get_training_data_up_to_date(db, draw_date)

        if len(training_df) < 20:
            print(f"⚠️  Insufficient training data ({len(training_df)} draws). Skipping...")
            print()
            continue

        print(f"📚 Training data: {len(training_df)} draws (up to {training_df['draw_date'].max()})")

        # Step 2: Prepare training data
        X_train, y_train, feature_cols = prepare_training_data(training_df, feature_engineer)

        # Step 3: Train model on historical data
        print(f"🔧 Training model on {len(X_train)} historical draws...")
        model.train(X_train, y_train)

        # Step 4: Prepare test data (single draw)
        test_df_single = pd.DataFrame([test_row])
        X_test, _, _ = prepare_training_data(test_df_single, feature_engineer)

        # Step 5: Generate prediction
        predicted_numbers = model.predict_top_k(X_test[0].reshape(1, -1), k=6)[0].tolist()

        # Step 6: Get actual numbers
        actual_numbers = sorted([
            int(test_row['number_1']), int(test_row['number_2']), int(test_row['number_3']),
            int(test_row['number_4']), int(test_row['number_5']), int(test_row['number_6'])
        ])
        additional_number = int(test_row['additional_number'])

        # Step 7: Calculate performance
        matches, prize_group, reward = calculate_matches_and_prize(
            predicted_numbers, actual_numbers, additional_number
        )

        # Display results
        print(f"🎯 Predicted:  {predicted_numbers}")
        print(f"✓  Actual:     {actual_numbers} + {additional_number}")
        print(f"📊 Matches:    {matches}/6")
        if reward > 0:
            print(f"🏆 Prize:      {prize_group} - ${reward:,}")
        else:
            print(f"   Prize:      {prize_group}")
        print()

        # Store result
        all_results.append({
            'draw_number': draw_number,
            'draw_date': draw_date,
            'predicted': predicted_numbers,
            'actual': actual_numbers,
            'additional': additional_number,
            'matches': matches,
            'prize_group': prize_group,
            'reward': reward,
            'training_size': len(training_df)
        })

    # Summary statistics
    print()
    print("=" * 90)
    print("📈 SUMMARY STATISTICS - JANUARY TO JUNE 2025")
    print("=" * 90)
    print()

    if not all_results:
        print("No results to summarize.")
        return

    # Overall statistics
    total_predictions = len(all_results)
    total_matches = sum(r['matches'] for r in all_results)
    avg_matches = total_matches / total_predictions
    total_wins = sum(1 for r in all_results if r['reward'] > 0)
    total_rewards = sum(r['reward'] for r in all_results)
    win_rate = (total_wins / total_predictions) * 100

    print(f"Total Predictions:   {total_predictions}")
    print(f"Total Matches:       {total_matches}")
    print(f"Average Matches:     {avg_matches:.2f}")
    print(f"Total Wins:          {total_wins}/{total_predictions} ({win_rate:.1f}%)")
    print(f"Total Rewards:       ${total_rewards:,}")
    print()

    # Breakdown by number of matches
    print("Match Distribution:")
    for i in range(7):
        count = sum(1 for r in all_results if r['matches'] == i)
        pct = (count / total_predictions) * 100
        bar = '█' * int(pct / 2)
        print(f"  {i} matches: {count:3d} ({pct:5.1f}%) {bar}")
    print()

    # Prize breakdown
    print("Prize Distribution:")
    prize_counts = {}
    for r in all_results:
        pg = r['prize_group']
        prize_counts[pg] = prize_counts.get(pg, 0) + 1

    for prize_group in ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6', 'No Prize']:
        count = prize_counts.get(prize_group, 0)
        pct = (count / total_predictions) * 100
        print(f"  {prize_group:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Monthly breakdown
    print("=" * 90)
    print("📅 MONTHLY PERFORMANCE BREAKDOWN")
    print("=" * 90)
    print()

    monthly_stats = {}
    for r in all_results:
        month = r['draw_date'][:7]  # YYYY-MM
        if month not in monthly_stats:
            monthly_stats[month] = {
                'count': 0,
                'matches': 0,
                'wins': 0,
                'rewards': 0
            }
        monthly_stats[month]['count'] += 1
        monthly_stats[month]['matches'] += r['matches']
        if r['reward'] > 0:
            monthly_stats[month]['wins'] += 1
        monthly_stats[month]['rewards'] += r['reward']

    print(f"{'Month':<12} │ {'Draws':<6} │ {'Avg Matches':<12} │ {'Wins':<8} │ {'Rewards':<12}")
    print(f"{'─' * 12}─┼─{'─' * 6}─┼─{'─' * 12}─┼─{'─' * 8}─┼─{'─' * 12}")

    for month in sorted(monthly_stats.keys()):
        stats = monthly_stats[month]
        avg_matches = stats['matches'] / stats['count']
        win_rate = (stats['wins'] / stats['count']) * 100
        print(f"{month:<12} │ {stats['count']:<6} │ {avg_matches:<12.2f} │ {stats['wins']}/{stats['count']} ({win_rate:.0f}%) │ ${stats['rewards']:<11,}")

    print()

    # Best and worst draws
    print("=" * 90)
    print("🏆 BEST PERFORMANCES")
    print("=" * 90)
    print()

    best_results = sorted(all_results, key=lambda x: (x['matches'], x['reward']), reverse=True)[:5]
    for i, r in enumerate(best_results, 1):
        print(f"{i}. {r['draw_date']} - {r['matches']} matches - {r['prize_group']} - ${r['reward']:,}")
        print(f"   Predicted: {r['predicted']}")
        print(f"   Actual:    {r['actual']} + {r['additional']}")
        print()

    # Performance comparison with baseline
    baseline_avg_matches = 0.73  # Random baseline
    improvement = ((avg_matches - baseline_avg_matches) / baseline_avg_matches) * 100

    print("=" * 90)
    print("📊 COMPARISON WITH BASELINE")
    print("=" * 90)
    print()
    print(f"P1_LightGBM (Incremental):  {avg_matches:.2f} avg matches")
    print(f"Random Baseline:            {baseline_avg_matches:.2f} avg matches")
    print(f"Improvement:                {improvement:+.1f}%")
    print()

    print("=" * 90)
    print("✅ INCREMENTAL LEARNING EVALUATION COMPLETE")
    print("=" * 90)
    print()
    print("⚠️  DISCLAIMER: Results based on mock data for educational purposes.")
    print("   Real lottery outcomes are random and cannot be reliably predicted.")
    print()


if __name__ == "__main__":
    main()
