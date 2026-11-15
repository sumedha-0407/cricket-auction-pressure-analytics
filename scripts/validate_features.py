"""
scripts/validate_features.py
Validates dataset column distributions, missing values, and run definitions.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

def validate_dataset():
    print("🔄 Loading datasets for validation...")
    competitions = pd.read_csv(DATA_DIR / "competitions.csv")
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    balls = pd.read_csv(DATA_DIR / "ball_by_ball.csv")
    
    print("\n--- EXTRAS TYPE VALUES ---")
    print(balls["extras_type"].value_counts(dropna=False))
    
    print("\n--- RUNS COLUMNS ---")
    print(balls[["runs", "extras1_runs", "extras_2_runs"]].describe())
    
    print("\n--- INNINGS DISTRIBUTION ---")
    print(balls["innings"].value_counts())
    
    print("\n--- REVISED OVERS ---")
    print(matches[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].describe(include="all"))
    
    print("\n--- WICKET VALUES ---")
    print(balls["is_wicket"].value_counts(dropna=False))
    
    print("\n--- DUPLICATE DELIVERY CHECK ---")
    print(balls.duplicated(["match_id", "innings", "overid", "over_ball_num"]).sum())

if __name__ == "__main__":
    validate_dataset()