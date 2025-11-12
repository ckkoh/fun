"""Compare Phase 1 vs Phase 2 models on January 2025 results."""

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

    print("=" * 90)
    print("PHASE 1 vs PHASE 2 MODEL COMPARISON - JANUARY 2025 RESULTS")
    print("=" * 90)

    # 1. Load Phase 1 models
    logger.info("Loading Phase 1 models...")
    phase1_dir = Path('models/checkpoints')

    phase1_models = {}
    try:
        phase1_models['P1_RF'] = joblib.load(phase1_dir / 'rf_model.pkl')
        phase1_models['P1_LightGBM'] = joblib.load(phase1_dir / 'lgb_model.pkl')
        phase1_models['P1_Ensemble'] = joblib.load(phase1_dir / 'ensemble.pkl')
        logger.info(f"  Loaded {len(phase1_models)} Phase 1 models")
    except Exception as e:
        logger.error(f"Error loading Phase 1 models: {e}")
        return

    # 2. Load Phase 2 models
    logger.info("Loading Phase 2 models...")
    phase2_dir = Path('models/checkpoints/phase2')

    phase2_models = {}
    try:
        phase2_models['P2_RF_Opt'] = joblib.load(phase2_dir / 'rf_optimized.pkl')
        phase2_models['P2_LGB_Opt'] = joblib.load(phase2_dir / 'lightgbm_optimized.pkl')
        feature_cols = joblib.load(phase2_dir / 'feature_cols.pkl')
        logger.info(f"  Loaded {len(phase2_models)} Phase 2 models")
    except Exception as e:
        logger.error(f"Error loading Phase 2 models: {e}")
        return

    # Combine all models
    all_models = {**phase1_models, **phase2_models}

    # 3. Load January 2025 data
    logger.info("\nLoading January 2025 test data...")
    db = DatabaseManager('data/toto.db')
    jan_draws = db.get_draws_in_date_range('2025-01-01', '2025-01-31')
    jan_df = pd.DataFrame(jan_draws)

    if len(jan_df) == 0:
        logger.error("No January 2025 draws found!")
        return

    logger.info(f"  Found {len(jan_df)} draws in January 2025")

    # 4. Engineer features
    logger.info("Engineering features...")
    all_data = db.get_all_draws()
    df_all = pd.DataFrame(all_data)

    engineer = TotoFeatureEngineer(lookback_windows=[5, 10, 20])
    df_features = engineer.engineer_features(df_all)

    # 5. Generate predictions for each January draw
    print("\n" + "=" * 90)
    print("GENERATING PREDICTIONS - 8 DRAWS × 5 MODELS = 40 PREDICTIONS")
    print("=" * 90)

    results = defaultdict(lambda: {
        'predictions': [],
        'matches': [],
        'match_numbers': [],
        'rewards': [],
        'prize_groups': []
    })

    jan_indices = df_all[df_all['draw_date'].isin(jan_df['draw_date'])].index.tolist()

    for idx, jan_idx in enumerate(jan_indices, 1):
        draw_data = df_all.iloc[jan_idx]
        draw_date = draw_data['draw_date']

        actual = [int(draw_data[f'number_{j}']) for j in range(1, 7)]
        additional = int(draw_data['additional_number'])

        print(f"\n{'─' * 90}")
        print(f"DRAW #{idx} - {draw_date}")
        print(f"{'─' * 90}")
        print(f"Actual Numbers:     {sorted(actual)}")
        print(f"Additional Number:  {additional}")
        print()

        # Use features up to this draw for prediction
        if jan_idx > 0:
            X = df_features[feature_cols].iloc[jan_idx-1:jan_idx].fillna(0).values
        else:
            X = df_features[feature_cols].iloc[0:1].fillna(0).values

        # Generate predictions from each model
        for model_name, model in all_models.items():
            pred = model.predict_top_k(X, k=6)[0]
            pred_sorted = sorted(pred.tolist())

            # Calculate matches
            match_count, match_nums = calculate_matches(pred, actual)
            has_additional = additional in pred
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

            phase_tag = "P1" if model_name.startswith('P1') else "P2"
            print(f"[{phase_tag}] {model_name:15s}: {pred_sorted}  │ {match_display} │ {match_count}/6 {additional_marker}")

            if match_nums:
                print(f"{'':20s}Matched: {match_nums}  │ {prize_group}")
            if reward > 0:
                print(f"{'':20s}Reward: ${reward:,}")

    # 6. Summary statistics by phase
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY - JANUARY 2025 (8 DRAWS)")
    print("=" * 90)

    phase1_summary = []
    phase2_summary = []

    for model_name in all_models.keys():
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

        summary = {
            'Model': model_name,
            'Avg Matches': avg_matches,
            'Max': max_matches,
            'Min': min_matches,
            'Total Reward': total_reward,
            'Wins': wins,
            'Distribution': match_dist
        }

        if model_name.startswith('P1'):
            phase1_summary.append(summary)
        else:
            phase2_summary.append(summary)

    # Sort by average matches
    phase1_summary.sort(key=lambda x: x['Avg Matches'], reverse=True)
    phase2_summary.sort(key=lambda x: x['Avg Matches'], reverse=True)

    print("\n🏅 PHASE 1 MODELS (Trained on 2024 data)")
    print("─" * 90)
    for rank, data in enumerate(phase1_summary, 1):
        print(f"{rank}. {data['Model']}")
        print(f"   Avg Matches: {data['Avg Matches']:.2f} │ Range: {data['Min']}-{data['Max']} │ Wins: {data['Wins']}/8 │ Reward: ${data['Total Reward']:,}")

    print("\n🏅 PHASE 2 MODELS (Trained on 2023 data + Optimized)")
    print("─" * 90)
    for rank, data in enumerate(phase2_summary, 1):
        print(f"{rank}. {data['Model']}")
        print(f"   Avg Matches: {data['Avg Matches']:.2f} │ Range: {data['Min']}-{data['Max']} │ Wins: {data['Wins']}/8 │ Reward: ${data['Total Reward']:,}")

    # 7. Overall winner
    all_summary = phase1_summary + phase2_summary
    all_summary.sort(key=lambda x: x['Avg Matches'], reverse=True)

    print("\n" + "=" * 90)
    print("🏆 OVERALL WINNER - ALL MODELS")
    print("=" * 90)

    for rank, data in enumerate(all_summary[:3], 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        phase = "Phase 1" if data['Model'].startswith('P1') else "Phase 2"

        print(f"{medal} {data['Model']} ({phase})")
        print(f"   Average: {data['Avg Matches']:.2f} matches │ Wins: {data['Wins']}/8 draws │ Total: ${data['Total Reward']:,}")

    # 8. Phase comparison
    print("\n" + "=" * 90)
    print("📊 PHASE 1 vs PHASE 2 COMPARISON")
    print("=" * 90)

    p1_avg = np.mean([s['Avg Matches'] for s in phase1_summary])
    p2_avg = np.mean([s['Avg Matches'] for s in phase2_summary])
    p1_wins = sum([s['Wins'] for s in phase1_summary])
    p2_wins = sum([s['Wins'] for s in phase2_summary])
    p1_reward = sum([s['Total Reward'] for s in phase1_summary])
    p2_reward = sum([s['Total Reward'] for s in phase2_summary])

    print(f"\nPhase 1 (3 models, 2024 training data):")
    print(f"  Average Matches: {p1_avg:.2f}")
    print(f"  Total Wins: {p1_wins}/24 predictions")
    print(f"  Total Rewards: ${p1_reward:,}")

    print(f"\nPhase 2 (2 models, 2023 training data + Optuna):")
    print(f"  Average Matches: {p2_avg:.2f}")
    print(f"  Total Wins: {p2_wins}/16 predictions")
    print(f"  Total Rewards: ${p2_reward:,}")

    improvement = ((p2_avg / p1_avg) - 1) * 100 if p1_avg > 0 else 0
    print(f"\nImprovement: {improvement:+.1f}%")

    if p2_avg > p1_avg:
        print("✅ Phase 2 models perform BETTER on average")
    elif p2_avg < p1_avg:
        print("⚠️  Phase 1 models perform BETTER on average")
    else:
        print("🤝 Phase 1 and Phase 2 perform EQUALLY")

    # 9. Detailed comparison table
    print("\n" + "=" * 90)
    print("DETAILED COMPARISON BY DRAW")
    print("=" * 90)

    print(f"\n{'Date':<12s} │ P1_RF │ P1_LGB │ P1_ENS │ P2_RF │ P2_LGB │ Best Model(s)")
    print("─" * 90)

    for idx, draw_date in enumerate(jan_df['draw_date']):
        p1_rf = results['P1_RF']['matches'][idx]
        p1_lgb = results['P1_LightGBM']['matches'][idx]
        p1_ens = results['P1_Ensemble']['matches'][idx]
        p2_rf = results['P2_RF_Opt']['matches'][idx]
        p2_lgb = results['P2_LGB_Opt']['matches'][idx]

        best_match = max(p1_rf, p1_lgb, p1_ens, p2_rf, p2_lgb)

        best_models = []
        if p1_rf == best_match: best_models.append('P1_RF')
        if p1_lgb == best_match: best_models.append('P1_LGB')
        if p1_ens == best_match: best_models.append('P1_ENS')
        if p2_rf == best_match: best_models.append('P2_RF')
        if p2_lgb == best_match: best_models.append('P2_LGB')

        best_str = ', '.join(best_models)

        print(f"{draw_date:<12s} │ {p1_rf:5d} │ {p1_lgb:6d} │ {p1_ens:6d} │ {p2_rf:5d} │ {p2_lgb:6d} │ {best_str}")

    print("\n" + "=" * 90)
    print("📝 CONCLUSIONS")
    print("=" * 90)

    print("\n1. Training Data Impact:")
    print("   Phase 1: Trained on 90 draws (2024 Q1-Q3)")
    print("   Phase 2: Trained on 91 draws (2023 full year)")

    print("\n2. Optimization:")
    print("   Phase 1: Default hyperparameters")
    print("   Phase 2: Optuna optimization (attempted)")

    print("\n3. Best Overall Model:")
    winner = all_summary[0]
    print(f"   {winner['Model']} with {winner['Avg Matches']:.2f} avg matches")

    print("\n" + "=" * 90)
    print("⚠️  DISCLAIMER: Results based on mock data for educational purposes.")
    print("Real lottery outcomes are random and cannot be reliably predicted.")
    print("=" * 90)


if __name__ == "__main__":
    main()
