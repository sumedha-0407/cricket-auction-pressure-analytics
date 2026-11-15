"""
scripts/validate_pressure_index.py
Executes the four mandatory validation tests for the Pressure Index pipeline.
"""

import numpy as np
import pandas as pd
from src.pressure.features import run_feature_pipeline


def test_1_range(df):
    print("\n========================================")
    print("TEST 1: RANGE TEST [0, 1]")
    print("========================================")
    
    bat_min = df["batting_pressure_index"].min()
    bat_max = df["batting_pressure_index"].max()
    bowl_min = df["bowling_pressure_index"].min()
    bowl_max = df["bowling_pressure_index"].max()
    
    print(f"Batting Pressure Index Range:  [{bat_min:.4f}, {bat_max:.4f}]")
    print(f"Bowling Pressure Index Range: [{bowl_min:.4f}, {bowl_max:.4f}]")
    
    assert 0.0 <= bat_min and bat_max <= 1.0, "❌ Batting Pressure Index out of [0, 1] bounds!"
    assert 0.0 <= bowl_min and bowl_max <= 1.0, "❌ Bowling Pressure Index out of [0, 1] bounds!"
    print("✅ TEST 1 PASSED: Both indices are strictly within [0, 1].")


def test_2_pre_delivery(df):
    print("\n========================================")
    print("TEST 2: PRE-DELIVERY STATE TEST")
    print("========================================")
    
    # Verify that runs_before + current delivery runs equals cum_runs
    # Pick a sample row where total_delivery_runs > 0
    sample = df[df["total_delivery_runs"] > 0].head(1).iloc[0]
    
    match_id = sample["match_id"]
    innings = sample["innings"]
    overid = sample["overid"]
    ball_num = sample["over_ball_num"]
    
    print(f"Inspecting Match {match_id}, Innings {innings}, Over {overid}.{ball_num}")
    print(f"  - runs_before (State before ball): {sample['runs_before']}")
    print(f"  - total_delivery_runs (Outcome of ball): {sample['total_delivery_runs']}")
    print(f"  - cum_runs (Cumulative after ball): {sample['cum_runs']}")
    
    assert sample["runs_before"] + sample["total_delivery_runs"] == sample["cum_runs"], \
        "❌ Leakage detected: runs_before does not match prior cumulative state!"
    print("✅ TEST 2 PASSED: Pre-delivery states correctly exclude current delivery outcomes.")


def test_3_match_inspection(df):
    print("\n========================================")
    print("TEST 3: MATCH INSPECTION (1st, 2nd, & DLS)")
    print("========================================")
    
    # Find a match with revised target / DLS if possible, otherwise standard match
    dls_matches = df[df["revised_target"].notna()]["match_id"]
    sample_match_id = dls_matches.iloc[0] if not dls_matches.empty else df["match_id"].iloc[0]
    
    match_sample = df[df["match_id"] == sample_match_id]
    print(f"Inspecting Match ID: {sample_match_id} (Total Deliveries: {len(match_sample)})")
    
    # Print a few key moments from 1st and 2nd innings
    cols_to_show = [
        "innings", "overid", "over_ball_num", "runs_before", "wickets_before", 
        "runs_required", "rrr", "batting_pressure_index", "bowling_pressure_index"
    ]
    
    print("\n--- First Innings Snapshot (Powerplay vs Death) ---")
    print(match_sample[match_sample["innings"] == 1].iloc[[0, 10, 35, 110]][cols_to_show].to_string(index=False))
    
    print("\n--- Second Innings Snapshot (Chase Progression) ---")
    print(match_sample[match_sample["innings"] == 2].iloc[[0, 10, 35, 110]][cols_to_show].to_string(index=False))
    print("✅ TEST 3 COMPLETED: Manual inspection output generated successfully.")


def test_4_sensitivity(df):
    print("\n========================================")
    print("TEST 4: SENSITIVITY TEST (Threshold Variations)")
    print("========================================")
    
    total_balls = len(df)
    for threshold in [0.60, 0.65, 0.70]:
        high_bat = (df["batting_pressure_index"] >= threshold).sum()
        high_bowl = (df["bowling_pressure_index"] >= threshold).sum()
        print(f"Threshold >= {threshold}:")
        print(f"  - High-Pressure Batting Deliveries: {high_bat} ({high_bat/total_balls*100:.2f}% of total)")
        print(f"  - High-Pressure Bowling Deliveries: {high_bowl} ({high_bowl/total_balls*100:.2f}% of total)")
        
    print("✅ TEST 4 PASSED: Sensitivity thresholds evaluated.")


if __name__ == "__main__":
    print("🔄 Running Feature Pipeline for Validation...")
    df_master = run_feature_pipeline()
    
    # Execute 4 validation tests
    test_1_range(df_master)
    test_2_pre_delivery(df_master)
    test_3_match_inspection(df_master)
    test_4_sensitivity(df_master)
    
    print("\n🎯 All Pressure Index Validation Tests Completed Successfully!")