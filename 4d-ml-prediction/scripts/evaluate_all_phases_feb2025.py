"""
Compare All Phases - February 2025

Quick comparison of Phase 1, Phase 2, and Phase 3 on February 2025 draws.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import logging
import warnings

from database.db_manager import FourDDatabaseManager
from pipeline.feature_engineer import FourDFeatureEngineer
from models.phase1_models import FourDLightGBM_Phase1, prepare_phase1_training_data
from models.digit_models import FourDLightGBM
from models.phase3_models import StatisticalFrequencyPredictor, PatternMiningPredictor

logging.basicConfig(level=logging.ERROR)

PRIZE_VALUES = {'first': 3000, 'second': 2000, 'third': 1000, 'starter': 250, 'consolation': 60}
BET_COST = 1


def get_training_data_up_to_date(db, end_date):
    all_draws = db.get_all_draws()
    filtered = [d for d in all_draws if d['draw_date'] < end_date]
    return pd.DataFrame(filtered) if filtered else pd.DataFrame()


def check_win(predicted, actual_draw):
    if predicted == actual_draw['first_prize']:
        return True, 'first', PRIZE_VALUES['first']
    if predicted == actual_draw['second_prize']:
        return True, 'second', PRIZE_VALUES['second']
    if predicted == actual_draw['third_prize']:
        return True, 'third', PRIZE_VALUES['third']
    for i in range(1, 11):
        if actual_draw.get(f'starter_{i}') == predicted:
            return True, 'starter', PRIZE_VALUES['starter']
    for i in range(1, 11):
        if actual_draw.get(f'consolation_{i}') == predicted:
            return True, 'consolation', PRIZE_VALUES['consolation']
    return False, 'no_prize', 0


def main():
    print("=" * 100)
    print("ALL PHASES COMPARISON - FEBRUARY 2025")
    print("=" * 100)
    print()

    db = FourDDatabaseManager()
    feature_engineer = FourDFeatureEngineer()

    test_draws = db.get_draws_in_date_range("2025-02-01", "2025-02-28")
    if not test_draws:
        print("No draws found!")
        return

    test_df = pd.DataFrame(test_draws)
    print(f"📊 Evaluating {len(test_df)} draws in February 2025\n")

    results = {'P1': [], 'P2': [], 'P3_Stat': [], 'P3_Pattern': [], 'P3_Combined': []}

    for idx, test_row in test_df.iterrows():
        draw_date = test_row['draw_date']
        training_df = get_training_data_up_to_date(db, draw_date)

        if len(training_df) < 20:
            continue

        print(f"{draw_date} (Train: {len(training_df)}) │ Actual: {test_row['first_prize']}")

        # Phase 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("  P1...", end=" ", flush=True)
            X, y, _ = prepare_phase1_training_data(training_df, feature_engineer)
            p1_model = FourDLightGBM_Phase1(n_estimators=100, max_depth=6, learning_rate=0.05)
            p1_model.train(X, y)
            p1_preds = p1_model.predict_top_k(X[-1:], k=1)[0]
            p1_num, p1_conf = p1_preds[0]
            p1_win, p1_cat, p1_pay = check_win(p1_num, test_row)
            print(f"{p1_num}({p1_conf:.0%}){' 🏆' if p1_win else ''}", end=" │ ")

        # Phase 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("P2...", end=" ", flush=True)
            df_feat = feature_engineer.engineer_features(training_df.copy())
            y_dict = {}
            for pos in range(4):
                _, y, _ = feature_engineer.prepare_training_data(df_feat, pos)
                y_dict[pos] = y
            X2, _, _ = feature_engineer.prepare_training_data(df_feat, 0)
            p2_model = FourDLightGBM(n_estimators=100, max_depth=6, learning_rate=0.05)
            p2_model.train(X2, y_dict)
            p2_preds = p2_model.generate_4d_numbers(X2[-1], top_k=1)
            p2_num, p2_conf = p2_preds[0]
            p2_win, p2_cat, p2_pay = check_win(p2_num, test_row)
            print(f"{p2_num}({p2_conf:.0%}){' 🏆' if p2_win else ''}", end=" │ ")

        # Phase 3 - Statistical
        print("P3S...", end=" ", flush=True)
        stat_pred = StatisticalFrequencyPredictor()
        stat_preds = stat_pred.predict_top_k(training_df, draw_date, k=1)
        p3s_num, p3s_score = stat_preds[0]
        p3s_win, p3s_cat, p3s_pay = check_win(p3s_num, test_row)
        print(f"{p3s_num}({p3s_score:.0%}){' 🏆' if p3s_win else ''}", end=" │ ")

        # Phase 3 - Pattern
        print("P3P...", end=" ", flush=True)
        pattern_pred = PatternMiningPredictor()
        pattern_preds = pattern_pred.predict_top_k(training_df, k=1)
        p3p_num, p3p_conf = pattern_preds[0]
        p3p_win, p3p_cat, p3p_pay = check_win(p3p_num, test_row)
        print(f"{p3p_num}({p3p_conf:.0%}){' 🏆' if p3p_win else ''}", end=" │ ")

        # Phase 3 - Combined
        print("P3C...", end=" ", flush=True)
        combined = {}
        combined[p3s_num] = combined.get(p3s_num, 0) + 0.5 * p3s_score
        combined[p3p_num] = combined.get(p3p_num, 0) + 0.5 * p3p_conf
        p3c_num = max(combined.items(), key=lambda x: x[1])[0]
        p3c_score = combined[p3c_num]
        p3c_win, p3c_cat, p3c_pay = check_win(p3c_num, test_row)
        print(f"{p3c_num}({p3c_score:.0%}){' 🏆' if p3c_win else ''}")

        # Store results
        results['P1'].append({'num': p1_num, 'conf': p1_conf, 'win': p1_win, 'pay': p1_pay})
        results['P2'].append({'num': p2_num, 'conf': p2_conf, 'win': p2_win, 'pay': p2_pay})
        results['P3_Stat'].append({'num': p3s_num, 'conf': p3s_score, 'win': p3s_win, 'pay': p3s_pay})
        results['P3_Pattern'].append({'num': p3p_num, 'conf': p3p_conf, 'win': p3p_win, 'pay': p3p_pay})
        results['P3_Combined'].append({'num': p3c_num, 'conf': p3c_score, 'win': p3c_win, 'pay': p3c_pay})

    # Summary
    print("\n" + "=" * 100)
    print("📊 SUMMARY - FEBRUARY 2025")
    print("=" * 100)
    print()

    print(f"{'Method':<25} │ {'Predictions':<12} │ {'Wins':<8} │ {'Win Rate':<10} │ {'Avg Conf':<10} │ {'Net Profit':<12}")
    print("─" * 100)

    for method_name, method_results in results.items():
        if not method_results:
            continue

        total = len(method_results)
        wins = sum(1 for r in method_results if r['win'])
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_conf = np.mean([r['conf'] for r in method_results])
        total_pay = sum(r['pay'] for r in method_results)
        net_profit = total_pay - (total * BET_COST)

        phase_label = {
            'P1': 'Phase 1 (10K-way)',
            'P2': 'Phase 2 (Digit-by-Digit)',
            'P3_Stat': 'Phase 3 (Statistical)',
            'P3_Pattern': 'Phase 3 (Pattern)',
            'P3_Combined': 'Phase 3 (Combined)'
        }[method_name]

        print(f"{phase_label:<25} │ {total:<12} │ {wins:>3}/{total:<3} │ {win_rate:>8.1f}% │ {avg_conf:>8.1%} │ ${net_profit:>+10,}")

    # Winner
    print("\n" + "=" * 100)
    print("🏆 WINNER")
    print("=" * 100)
    print()

    max_wins = max(sum(1 for r in results[m] if r['win']) for m in results if results[m])
    winners = [m for m in results if results[m] and sum(1 for r in results[m] if r['win']) == max_wins]

    if max_wins == 0:
        print("🤝 TIE: All methods achieved 0 wins")
    else:
        for w in winners:
            total_pay = sum(r['pay'] for r in results[w])
            print(f"✅ {w}: {max_wins} wins, ${total_pay:,} total payout")

    print("\n" + "=" * 100)
    print("📝 KEY OBSERVATIONS")
    print("=" * 100)
    print()

    # Confidence analysis
    p1_conf = np.mean([r['conf'] for r in results['P1']])
    p2_conf = np.mean([r['conf'] for r in results['P2']])
    p3_conf = np.mean([r['conf'] for r in results['P3_Combined']])

    print(f"Average Confidence:")
    print(f"  Phase 1: {p1_conf:.1%} (very high but overconfident)")
    print(f"  Phase 2: {p2_conf:.1%} (moderate)")
    print(f"  Phase 3: {p3_conf:.1%} (balanced)")
    print()

    # Speed comparison
    print(f"Computational Speed:")
    print(f"  Phase 1: ~20-30 min for 10 draws (slowest)")
    print(f"  Phase 2: ~3-5 min for 10 draws (fast)")
    print(f"  Phase 3: <1 min for 10 draws (fastest)")
    print()

    print("=" * 100)
    print("✅ EVALUATION COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
