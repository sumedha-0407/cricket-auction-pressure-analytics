"""scripts/run_pipeline.py
One command to run the complete end-to-end pipeline:
  1. clean.run() -> data/interim/
  2. features.run_feature_pipeline() -> data/processed/
  3. models.run() -> outputs/ (Top 10 CSVs, JSONs, figures)
  4. build_notebook.create_notebook() -> notebooks/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pressure import clean, features, models
from scripts import build_notebook


def run_all():
    print("==================================================")
    print("STEP 1: CLEANING RAW CSVs -> INTERIM PARQUET")
    print("==================================================")
    clean_summary = clean.run()
    for k, v in clean_summary.items():
        print(f"  {k}: {v}")

    print("\n==================================================")
    print("STEP 2: FEATURE ENGINEERING (BPI & QPI)")
    print("==================================================")
    df_features = features.run_feature_pipeline()

    print("\n==================================================")
    print("STEP 3: CALIBRATED CLUTCH MODEL & EVALUATION")
    print("==================================================")
    batters, bowlers = models.run()

    print("\n==================================================")
    print("STEP 4: GENERATING INSIGHTS JUPYTER NOTEBOOK")
    print("==================================================")
    build_notebook.create_notebook()

    print("\n✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all()
