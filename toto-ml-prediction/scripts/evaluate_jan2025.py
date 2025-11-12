"""Generate 8 sets of predictions and compare with Jan 2025 actual results."""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from collections import defaultdict

from database.db_manager import DatabaseManager
from pipeline.feature_engineer import TotoFeatureEngineer
from models.simple_models import calculate_matches


def calculate_reward(matches, has_additional=False):
    """Calculate reward based on match count."""
    if matches == 6:
        return 1000000, "Group 1 (Jackpot!)"
    elif matches == 5 and has_additional:
        return 100000, "Group 2"
    elif matches == 5:
        return 10000, "Group 3"
    elif matches == 4:
        return 1000, "Group 4"
    elif matches == 3 and has_additional:
        return 50, "Group 5"
    elif matches == 3:
        return 25, "Group 6"
    else:
        return 0, "No Prize"


def main():
    """Main evaluation script."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("TOTO ML MODEL EVALUATION - JANUARY 2025 RESULTS")
    print("=" * 80)

    # 1. Load models
    logger.info("Loading trained models...")
    model_dir = Path('models/checkpoints')

    rf_model = joblib.load(model_dir / 'rf_model.pkl')
    lgb_model = joblib.load(model_dir / 'lgb_model.pkl')
    ensemble = joblib.load(model_dir / 'ensemble.pkl')
    feature_cols = joblib.load(model_dir / 'feature_cols.pkl')

    models = {
        'Random Forest': rf_model,
        'LightGBM': lgb_model,
        'Ensemble': ensemble
    }

    logger.info(f"Loaded {len(models)} models")

    # 2. Load January 2025 data
    logger.info("\nLoading January 2025 draws...")
    db = DatabaseManager('data/toto.db')

    # Get all draws and filter for Jan 2025
    all_draws = db.get_draws_in_date_range('2025-01-01', '2025-01-31')
    jan_draws = pd.DataFrame(all_draws)

    if len(jan_draws) == 0:
        print("\n❌ No January 2025 draws found in database!")
        return

    print(f"\n✓ Found {len(jan_draws)} draws in January 2025")
    print(f"  Date range: {jan_draws['draw_date'].min()} to {jan_draws['draw_date'].max()}")

    # 3. Load all data for feature engineering
    all_data = db.get_all_draws()
    df_all = pd.DataFrame(all_data)

    # Engineer features for entire dataset
    logger.info("\nEngineering features...")
    engineer = TotoFeatureEngineer(lookback_windows=[5, 10, 20])
    df_features = engineer.engineer_features(df_all)

    # 4. Generate predictions for each January 2025 draw
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS FOR 8 JANUARY 2025 DRAWS")
    print("=" * 80)

    results = defaultdict(lambda: {
        'predictions': [],
        'matches': [],
        'match_numbers': [],
        'rewards': [],
        'prize_groups': []
    })

    # Get indices of January draws in the full dataset
    jan_indices = df_all[df_all['draw_date'].isin(jan_draws['draw_date'])].index.tolist()

    for idx, jan_idx in enumerate(jan_indices, 1):
        draw_data = df_all.iloc[jan_idx]
        draw_date = draw_data['draw_date']

        # Get actual numbers
        actual = [int(draw_data[f'number_{j}']) for j in range(1, 7)]
        additional = int(draw_data['additional_number'])

        print(f"\n{'─' * 80}")
        print(f"DRAW #{idx} - {draw_date}")
        print(f"{'─' * 80}")
        print(f"Actual Numbers:     {sorted(actual)}")
        print(f"Additional Number:  {additional}")
        print()

        # Use features up to (but not including) this draw for prediction
        if jan_idx > 0:
            X = df_features[feature_cols].iloc[jan_idx-1:jan_idx].fillna(0).values
        else:
            # For first draw, use first available features
            X = df_features[feature_cols].iloc[0:1].fillna(0).values

        # Generate predictions from each model
        for model_name, model in models.items():
            pred = model.predict_top_k(X, k=6)[0]
            pred_sorted = sorted(pred.tolist())

            # Calculate matches
            match_count, match_nums = calculate_matches(pred, actual)

            # Check if additional number is in prediction
            has_additional = additional in pred

            # Calculate reward
            reward, prize_group = calculate_reward(match_count, has_additional)

            # Store results
            results[model_name]['predictions'].append(pred_sorted)
            results[model_name]['matches'].append(match_count)
            results[model_name]['match_numbers'].append(match_nums)
            results[model_name]['rewards'].append(reward)
            results[model_name]['prize_groups'].append(prize_group)

            # Display
            match_display = f"{'✓' * match_count}{'✗' * (6 - match_count)}"
            additional_marker = "🎯" if has_additional else ""

            print(f"{model_name:15s}: {pred_sorted}  │ {match_display} │ {match_count}/6 {additional_marker}")
            if match_nums:
                print(f"{'':17s}Matched: {match_nums}  │ {prize_group}")
            if reward > 0:
                print(f"{'':17s}Reward: ${reward:,}")

    # 5. Summary statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY - JANUARY 2025 (8 DRAWS)")
    print("=" * 80)

    summary_data = []

    for model_name in models.keys():
        matches = results[model_name]['matches']
        rewards = results[model_name]['rewards']

        avg_matches = np.mean(matches)
        max_matches = np.max(matches)
        min_matches = np.min(matches)
        total_reward = np.sum(rewards)
        wins = sum(1 for r in rewards if r > 0)

        match_dist = {}
        for m in range(7):
            count = matches.count(m)
            if count > 0:
                match_dist[m] = count

        summary_data.append({
            'Model': model_name,
            'Avg Matches': avg_matches,
            'Max': max_matches,
            'Min': min_matches,
            'Total Reward': total_reward,
            'Wins': wins,
            'Distribution': match_dist
        })

    # Sort by average matches (descending)
    summary_data.sort(key=lambda x: x['Avg Matches'], reverse=True)

    print()
    for rank, data in enumerate(summary_data, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."

        print(f"{medal} {data['Model']}")
        print(f"   Average Matches:  {data['Avg Matches']:.2f}")
        print(f"   Range:            {data['Min']} - {data['Max']}")
        print(f"   Total Reward:     ${data['Total Reward']:,}")
        print(f"   Prize Wins:       {data['Wins']}/8 draws")
        print(f"   Distribution:     {data['Distribution']}")
        print()

    # 6. Detailed comparison table
    print("=" * 80)
    print("DETAILED MATCH COMPARISON")
    print("=" * 80)
    print()

    # Create comparison table
    print(f"{'Draw Date':<12s} │ {'RF':<3s} │ {'LGB':<3s} │ {'ENS':<3s} │ Best Model")
    print("─" * 80)

    for idx, draw_date in enumerate(jan_draws['draw_date']):
        rf_matches = results['Random Forest']['matches'][idx]
        lgb_matches = results['LightGBM']['matches'][idx]
        ens_matches = results['Ensemble']['matches'][idx]

        best_match = max(rf_matches, lgb_matches, ens_matches)
        best_models = []
        if rf_matches == best_match:
            best_models.append('RF')
        if lgb_matches == best_match:
            best_models.append('LGB')
        if ens_matches == best_match:
            best_models.append('ENS')

        best_str = ', '.join(best_models)

        print(f"{draw_date:<12s} │ {rf_matches:<3d} │ {lgb_matches:<3d} │ {ens_matches:<3d} │ {best_str}")

    # 7. Statistical comparison to random baseline
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE vs RANDOM BASELINE")
    print("=" * 80)
    print()

    random_baseline = 0.7347  # Expected matches for random selection

    for data in summary_data:
        model_name = data['Model']
        avg_matches = data['Avg Matches']
        improvement = ((avg_matches / random_baseline) - 1) * 100

        print(f"{model_name:15s}: {avg_matches:.2f} matches (baseline: {random_baseline:.2f})")
        print(f"{'':17s}Improvement: {improvement:+.1f}%")
        print()

    # 8. Winner announcement
    print("=" * 80)
    print("🏆 MODEL RANKING - JANUARY 2025")
    print("=" * 80)
    print()

    for rank, data in enumerate(summary_data, 1):
        if rank == 1:
            print(f"🥇 WINNER: {data['Model']}")
            print(f"   Average: {data['Avg Matches']:.2f} matches per draw")
            print(f"   Prizes Won: {data['Wins']}/8 draws")
            print(f"   Total Winnings: ${data['Total Reward']:,}")
        elif rank == 2:
            print(f"\n🥈 Runner-up: {data['Model']}")
            print(f"   Average: {data['Avg Matches']:.2f} matches")
        elif rank == 3:
            print(f"\n🥉 Third Place: {data['Model']}")
            print(f"   Average: {data['Avg Matches']:.2f} matches")

    print("\n" + "=" * 80)
    print("⚠️  DISCLAIMER: Mock data used for demonstration purposes.")
    print("Real TOTO draws are random and cannot be reliably predicted.")
    print("=" * 80)


if __name__ == "__main__":
    main()
