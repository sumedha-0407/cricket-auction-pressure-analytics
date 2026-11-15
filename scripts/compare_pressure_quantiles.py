
import pandas as pd

from src.pressure.features import run_feature_pipeline

df = run_feature_pipeline()

columns = {
    "Batting": "batting_pressure_index",
    "Bowling": "bowling_pressure_index"
}

percentiles = [0.50, 0.75, 0.90, 0.95, 0.99]

print("\n==========================================")
print("PRESSURE INDEX QUANTILE COMPARISON")
print("==========================================")

for name, column in columns.items():

    values = df[column].dropna()

    print(f"\n--- {name.upper()} PRESSURE ---")
    print(f"Total valid deliveries: {len(values)}")

    print(
        values.quantile(percentiles).rename(
            index={
                0.50: "50% (Median)",
                0.75: "75%",
                0.90: "90%",
                0.95: "95%",
                0.99: "99%"
            }
        ).to_string()
    )


# Compare fixed and quantile-equivalent thresholds
print("\n==========================================")
print("HIGH-PRESSURE THRESHOLD COMPARISON")
print("==========================================")

for name, column in columns.items():

    values = df[column].dropna()

    print(f"\n--- {name.upper()} ---")

    for percentile in [0.90, 0.95]:
        threshold = values.quantile(percentile)

        count = (values >= threshold).sum()
        percentage = count / len(values) * 100

        print(
            f"Top {(1-percentile)*100:.0f}% threshold: "
            f"{threshold:.4f} | "
            f"Actual selected: {percentage:.2f}%"
        )

    fixed_count = (values >= 0.65).sum()

    print(
        f"Fixed threshold 0.65: "
        f"{fixed_count / len(values) * 100:.2f}%"
    )
