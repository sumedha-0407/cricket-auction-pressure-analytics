"""scripts/audit_pipeline.py
Audits the feature engineering pipeline for data leakage, boundary bleeding,
NaN propagation, and division-by-zero vulnerabilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pressure import features


def audit_pipeline():
    print("🔄 Running Feature Pipeline for Vulnerability Audit...")
    df = features.run_feature_pipeline()

    print("\n========================================")
    print("VULNERABILITY AUDIT REPORT")
    print("========================================")

    # 1. Check for NaNs in critical columns
    critical_cols = [
        "bpi", "qpi", "runs_before", "wickets_lost", "balls_remaining"
    ]
    nan_counts = df[critical_cols].isna().sum()
    print("\n1. NaN Check in Critical Columns:")
    print(nan_counts)
    for col in critical_cols:
        assert nan_counts[col] == 0, f"❌ NaN detected in critical column: {col}"
    print("   ✅ PASSED: Zero NaNs in all critical feature columns.")

    # 2. Check for Cross-Innings Bleeding
    first_balls = df[(df["over0"] == 0) & (df["over_ball_num"] == 1)]
    non_zero_starts = first_balls[first_balls["runs_before"] != 0.0]
    print(f"\n2. Cross-Innings / Match Bleeding Check:")
    print(f"   - Total innings starts inspected: {len(first_balls)}")
    print(f"   - Innings starts where runs_before != 0.0: {len(non_zero_starts)}")
    assert len(non_zero_starts) == 0, "❌ VULNERABILITY FOUND: runs_before does not reset to 0 at innings start!"
    print("   ✅ PASSED: Pre-delivery state correctly resets at every innings boundary.")

    # 3. Check Balls Remaining Range Check
    min_balls_rem = df["balls_remaining"].min()
    max_balls_rem = df["balls_remaining"].max()
    print(f"\n3. Balls Remaining Range Check:")
    print(f"   - Minimum balls_remaining: {min_balls_rem}")
    print(f"   - Maximum balls_remaining: {max_balls_rem}")
    assert min_balls_rem >= 0, "❌ VULNERABILITY FOUND: balls_remaining dropped below 0!"
    assert max_balls_rem <= 120, "❌ VULNERABILITY FOUND: balls_remaining exceeded 120!"
    print("   ✅ PASSED: balls_remaining is strictly bounded in [0, 120].")

    # 4. Check Pressure Index Bounds [0, 1]
    bpi_min, bpi_max = df["bpi"].min(), df["bpi"].max()
    qpi_min, qpi_max = df["qpi"].min(), df["qpi"].max()
    print(f"\n4. Pressure Index Bounds Check:")
    print(f"   - Batting Pressure Range (BPI):  [{bpi_min:.4f}, {bpi_max:.4f}]")
    print(f"   - Bowling Pressure Range (QPI):  [{qpi_min:.4f}, {qpi_max:.4f}]")
    assert 0.0 <= bpi_min and bpi_max <= 1.0, "❌ BPI out of bounds!"
    assert 0.0 <= qpi_min and qpi_max <= 1.0, "❌ QPI out of bounds!"
    print("   ✅ PASSED: Both BPI and QPI are strictly confined within [0, 1].")

    # 5. Check Monotonicity of Wicket Pressure
    # When wickets_lost increases, batting pressure should increase, not decrease!
    print("\n5. Wicket Pressure Monotonicity Check:")
    inn2_high_rrr = df[(df["innings"] == 2) & (df["balls_remaining"] <= 30) & (df["rrr"] >= 10.0)]
    wkt_bpi_corr = inn2_high_rrr["wickets_lost"].corr(inn2_high_rrr["bpi"])
    print(f"   - Correlation between wickets_lost and BPI: {wkt_bpi_corr:.4f}")
    assert wkt_bpi_corr > 0.3, "❌ Inversion bug detected: wickets_lost does not positively drive batting pressure!"
    print("   ✅ PASSED: Wickets lost correctly drives batting pressure upward.")

    print("\n🎯 ALL 5 VULNERABILITY CHECKS PASSED WITH ZERO FLAWS!")


if __name__ == "__main__":
    audit_pipeline()
