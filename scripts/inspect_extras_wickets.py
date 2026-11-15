
"""
scripts/inspect_extras_wickets.py

Purpose:
Inspect extras distribution, bowling runs conceded,
and wicket dismissal types in the master feature dataset.

SupeRR Selector Season 2
"""

import pandas as pd
from src.pressure.features import run_feature_pipeline


def inspect_data():

    print("🔄 Loading pipeline features...")
    df = run_feature_pipeline()

    # --------------------------------------------------
    # 1. EXTRAS DISTRIBUTION
    # --------------------------------------------------

    print("\n========== EXTRAS DISTRIBUTION ==========")

    if "extras_type" in df.columns:
        print(df["extras_type"].value_counts(dropna=False))
    else:
        print("extras_type column not found.")

    # --------------------------------------------------
    # 2. DELIVERIES WITH EXTRAS
    # --------------------------------------------------

    print("\n========== DELIVERIES WITH EXTRAS ==========")

    extra_cols = [
        c for c in ["extras1_runs", "extras_2_runs"]
        if c in df.columns
    ]

    if extra_cols:

        condition = pd.Series(False, index=df.index)

        for col in extra_cols:
            condition |= df[col].fillna(0) > 0

        extras = df[condition]

        show_cols = [
            c for c in [
                "match_id",
                "innings",
                "runs",
                "total_delivery_runs",
                "extras_type",
                "extras1_runs",
                "extras_2_runs"
            ]
            if c in df.columns
        ]

        if not extras.empty:
            print(extras[show_cols].head(25).to_string(index=False))
        else:
            print("No deliveries with extras found.")

    else:
        print("Extra run columns not found.")

    # --------------------------------------------------
    # 3. ALL WICKET DISMISSAL TYPES
    # --------------------------------------------------

    print("\n========== ALL DISMISSAL TYPES ==========")

    if "is_wicket" in df.columns and "dismissal_type" in df.columns:

        wickets = df[df["is_wicket"] == True]

        print(
            wickets["dismissal_type"]
            .value_counts(dropna=False)
        )

    else:
        print("Required wicket columns not found.")

    # --------------------------------------------------
    # 4. WICKET INFORMATION
    # --------------------------------------------------

    print("\n========== WICKET INFORMATION ==========")

    wicket_cols = [
        c for c in df.columns
        if "wicket" in c.lower() or "dismiss" in c.lower()
    ]

    print("Available columns:", wicket_cols)

    if "is_wicket" in df.columns and wicket_cols:

        wickets = df[df["is_wicket"] == True]

        print(
            wickets[wicket_cols]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------
    # 5. OTHER EXTRAS AND PENALTIES
    # --------------------------------------------------

    print("\n========== OTHER EXTRAS AND PENALTIES ==========")

    if "extras_type" in df.columns:

        unusual_extras = df[
            df["extras_type"].isin(["other", "Penalty"])
        ]

        if not unusual_extras.empty:

            show_cols = [
                c for c in [
                    "match_id",
                    "innings",
                    "runs",
                    "total_delivery_runs",
                    "extras_type",
                    "extras1_runs",
                    "extras_2_runs"
                ]
                if c in df.columns
            ]

            print(
                unusual_extras[show_cols]
                .to_string(index=False)
            )

        else:
            print("No other or penalty extras found.")

    # --------------------------------------------------
    # 6. OVERSTEP / POSSIBLE NO-BALL INSPECTION
    # --------------------------------------------------

    print("\n========== OVERSTEP EXAMPLES ==========")

    if "extras_type" in df.columns:

        oversteps = df[df["extras_type"] == "Overstep"]

        if not oversteps.empty:

            show_cols = [
                c for c in [
                    "match_id",
                    "innings",
                    "runs",
                    "total_delivery_runs",
                    "extras_type",
                    "extras1_runs",
                    "extras_2_runs"
                ]
                if c in df.columns
            ]

            print(
                oversteps[show_cols]
                .head(15)
                .to_string(index=False)
            )

        else:
            print("No Overstep deliveries found.")

    # --------------------------------------------------
    # 7. SUMMARY
    # --------------------------------------------------

    print("\n========== INSPECTION SUMMARY ==========")

    print("Total deliveries:", len(df))

    if "is_wicket" in df.columns:
        print("Total wicket deliveries:", int(df["is_wicket"].sum()))

    if "extras_type" in df.columns:
        print(
            "Total deliveries with recorded extras:",
            int(df["extras_type"].notna().sum())
        )

    print("\n✅ Inspection completed successfully.")


if __name__ == "__main__":
    inspect_data()
