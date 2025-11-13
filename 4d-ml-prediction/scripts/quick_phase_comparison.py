"""
Quick Phase 1 vs Phase 2 Comparison - February 2025 Only

Faster evaluation using only February 2025 data (10 draws).
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

logging.basicConfig(level=logging.ERROR)

PRIZE_VALUES_BIG = {'first': 3000, 'second': 2000, 'third': 1000, 'starter': 250, 'consolation': 60}
BET_COST = 1


def get_training_data_up_to_date(db, end_date):
    all_draws = db.get_all_draws()
    filtered_draws = [d for d in all_draws if d['draw_date'] < end_date]
    return pd.DataFrame(filtered_draws) if filtered_draws else pd.DataFrame()


def check_prediction_win(predicted, actual_draw):
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
    print("QUICK PHASE 1 vs PHASE 2 COMPARISON - FEBRUARY 2025")
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

    p1_results = []
    p2_results = []

    for idx, test_row in test_df.iterrows():
        draw_date = test_row['draw_date']
        training_df = get_training_data_up_to_date(db, draw_date)

        if len(training_df) < 20:
            continue

        print(f"{draw_date} (Train: {len(training_df)})", end=" │ ")

        # Phase 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, y, feature_cols = prepare_phase1_training_data(training_df, feature_engineer)
            p1_model = FourDLightGBM_Phase1(n_estimators=100, max_depth=6, learning_rate=0.05)
            p1_model.train(X, y, feature_names=feature_cols)
            p1_preds = p1_model.predict_top_k(X[-1:], k=1)[0]
            p1_number, p1_conf = p1_preds[0]

        # Phase 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_featured = feature_engineer.engineer_features(training_df.copy())
            y_dict = {}
            for pos in range(4):
                _, y, _ = feature_engineer.prepare_training_data(df_featured, pos)
                y_dict[pos] = y
            X2, _, _ = feature_engineer.prepare_training_data(df_featured, 0)
            p2_model = FourDLightGBM(n_estimators=100, max_depth=6, learning_rate=0.05)
            p2_model.train(X2, y_dict)
            p2_preds = p2_model.generate_4d_numbers(X2[-1], top_k=1)
            p2_number, p2_conf = p2_preds[0]

        # Check wins
        p1_win, p1_cat, p1_pay = check_prediction_win(p1_number, test_row)
        p2_win, p2_cat, p2_pay = check_prediction_win(p2_number, test_row)

        print(f"P1: {p1_number}({p1_conf:.0%}){' 🏆' if p1_win else''} │ P2: {p2_number}({p2_conf:.0%}){' 🏆' if p2_win else''} │ Actual: {test_row['first_prize']}")

        p1_results.append({'pred': p1_number, 'conf': p1_conf, 'win': p1_win, 'pay': p1_pay})
        p2_results.append({'pred': p2_number, 'conf': p2_conf, 'win': p2_win, 'pay': p2_pay})

    # Summary
    print("\n" + "=" * 100)
    print("📊 SUMMARY")
    print("=" * 100)
    print()

    p1_wins = sum(r['win'] for r in p1_results)
    p2_wins = sum(r['win'] for r in p2_results)
    p1_conf_avg = np.mean([r['conf'] for r in p1_results])
    p2_conf_avg = np.mean([r['conf'] for r in p2_results])
    p1_total_pay = sum(r['pay'] for r in p1_results)
    p2_total_pay = sum(r['pay'] for r in p2_results)
    total_cost = len(p1_results) * BET_COST

    print(f"{'Metric':<25} │ {'Phase 1':^20} │ {'Phase 2':^20} │ {'Winner':^15}")
    print("─" * 100)
    print(f"{'Total Predictions':<25} │ {len(p1_results):^20} │ {len(p2_results):^20} │ {'Tie':^15}")
    print(f"{'Wins':<25} │ {p1_wins:^20} │ {p2_wins:^20} │ {('P1' if p1_wins > p2_wins else 'P2' if p2_wins > p1_wins else 'Tie'):^15}")
    print(f"{'Win Rate':<25} │ {(p1_wins/len(p1_results)*100):^19.1f}% │ {(p2_wins/len(p2_results)*100):^19.1f}% │ {('P1' if p1_wins > p2_wins else 'P2' if p2_wins > p1_wins else 'Tie'):^15}")
    print(f"{'Avg Confidence':<25} │ {p1_conf_avg:^19.1%} │ {p2_conf_avg:^19.1%} │ {('P1' if p1_conf_avg > p2_conf_avg else 'P2'):^15}")
    print(f"{'Total Payout':<25} │ ${p1_total_pay:^19,} │ ${p2_total_pay:^19,} │ {('P1' if p1_total_pay > p2_total_pay else 'P2' if p2_total_pay > p1_total_pay else 'Tie'):^15}")
    print(f"{'Net Profit':<25} │ ${(p1_total_pay-total_cost):^19,} │ ${(p2_total_pay-total_cost):^19,} │ {('P1' if p1_total_pay > p2_total_pay else 'P2' if p2_total_pay > p1_total_pay else 'Tie'):^15}")

    print("\n" + "=" * 100)
    print("📝 CONCLUSIONS")
    print("=" * 100)
    print()

    if p1_wins > p2_wins:
        print(f"✅ WINNER: Phase 1 (Direct 4D Classification)")
    elif p2_wins > p1_wins:
        print(f"✅ WINNER: Phase 2 (Digit-by-Digit)")
    else:
        print(f"🤝 TIE: Both models performed equally")

    print(f"\nPhase 1 had significantly higher confidence ({p1_conf_avg:.1%} vs {p2_conf_avg:.1%})")
    print(f"This suggests Phase 1 is more certain about its predictions")
    print()


if __name__ == "__main__":
    main()
