"""
scripts/inspect_dataset_schema.py
Inspects extras encoding, wicket types, and delivery columns for models.py refinement.
"""

from pathlib import Path
import pandas as pd
from src.pressure.features import run_feature_pipeline

def inspect_schema():
    print("🔄 Loading pipeline features...")
    df = run_feature_pipeline()
    
    print("\n--- EXTRAS COLUMNS ---")
    print([c for c in df.columns if "extra" in c.lower()])
    
    print("\n--- WICKET COLUMNS ---")
    print([c for c in df.columns if "wicket" in c.lower()])
    
    print("\n--- BALL COLUMNS ---")
    print([c for c in df.columns if "ball" in c.lower()])
    
    print("\n--- SAMPLE DELIVERY DATA ---")
    cols = [
        c for c in [
            "match_id", "innings", "overid", "over_ball_num",
            "runs", "total_delivery_runs", "extras_type", "extras1_runs",
            "extras_2_runs", "is_wicket", "wicket_type", "bowler_name"
        ] if c in df.columns
    ]
    print(df[cols].head(15).to_string(index=False))

if __name__ == "__main__":
    inspect_schema()