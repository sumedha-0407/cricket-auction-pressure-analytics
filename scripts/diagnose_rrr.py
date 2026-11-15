"""
scripts/diagnose_rrr.py
Isolates RRR anomalies, ball-remaining logic, and pressure distributions.
"""

from pathlib import Path
import pandas as pd
from src.pressure.features import load_raw_data, run_feature_pipeline

DATA_DIR = Path("data/raw")

def diagnose():
    print("🔄 Loading raw matches and running feature pipeline...")
    competitions, matches, balls = load_raw_data(DATA_DIR)
    df_features = run_feature_pipeline()
    
    match_id = 13612
    
    print(f"\n========================================")
    print(f"1. MATCH METADATA INSPECTION FOR MATCH {match_id}")
    print("========================================")
    match_meta = matches.loc[matches["match_id"] == match_id, [
        "match_id", "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"
    ]]
    print(match_meta.to_string(index=False))
    
    print(f"\n========================================")
    print(f"2. FEATURE TABLE SNAPSHOT FOR MATCH {match_id} (2ND INNINGS)")
    print("========================================")
    cols = [
        "match_id", "innings", "overid", "over_ball_num", "target", 
        "runs_before", "runs_required", "legal_balls_before", "balls_remaining", "rrr"
    ]
    snapshot = df_features[
        (df_features["match_id"] == match_id) & 
        (df_features["innings"] == 2)
    ][cols].head(10)
    print(snapshot.to_string(index=False))
    
    print(f"\n========================================")
    print(f"3. PRESSURE INDEX DISTRIBUTION INSPECTION")
    print("========================================")
    print("--- Batting Pressure Index Quantiles ---")
    print(df_features["batting_pressure_index"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
    
    print("\n--- Bowling Pressure Index Quantiles ---")
    print(df_features["bowling_pressure_index"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

if __name__ == "__main__":
    diagnose()