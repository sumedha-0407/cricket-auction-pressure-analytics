"""src/pressure/models.py

Calibrated Clutch Evaluation, Residual Lift Modeling, Empirical Bayes Shrinkage,
and Top-10 Selection Pipeline for SupeRR Selector Season 2.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from . import config as C
from . import features


def evaluate_batters(df: pd.DataFrame, min_hp_balls: int = 100, k_shrink: float = 45.0) -> pd.DataFrame:
    """Evaluates batters using league-standardized metrics, ANCOVA residual lift

    (solving the Elite Penalty / Floor Effect), Empirical Bayes shrinkage,
    and match-to-match consistency weighting.
    """
    tau_b = df.groupby("league")["bpi"].transform(lambda x: x.quantile(0.75))
    df["is_hp_bat"] = df["bpi"] >= tau_b

    # Extract delivery records per player per league
    records = []
    for (pid, pname, league), g in df.groupby(["striker_id", "striker_name", "league"]):
        hp = g[g["is_hp_bat"] & g["batter_faced"]]
        base = g[~g["is_hp_bat"] & g["batter_faced"]]

        hp_balls = hp["batter_faced"].sum()
        norm_balls = base["batter_faced"].sum()

        if hp_balls >= 10:
            hp_sr = (hp["bat_runs"].sum() / hp_balls) * 100.0
            hp_bnd = (hp["is_boundary"].sum() / hp_balls) * 100.0
            hp_out = (hp["is_wicket"].sum() / hp_balls) * 100.0

            norm_sr = (base["bat_runs"].sum() / max(norm_balls, 1)) * 100.0 if norm_balls > 0 else hp_sr
            norm_bnd = (base["is_boundary"].sum() / max(norm_balls, 1)) * 100.0 if norm_balls > 0 else hp_bnd
            norm_out = (base["is_wicket"].sum() / max(norm_balls, 1)) * 100.0 if norm_balls > 0 else hp_out

            records.append({
                "striker_id": pid, "striker_name": pname, "league": league,
                "hp_balls": hp_balls, "norm_balls": norm_balls,
                "hp_sr": hp_sr, "norm_sr": norm_sr,
                "hp_bnd": hp_bnd, "norm_bnd": norm_bnd,
                "hp_out": hp_out, "norm_out": norm_out
            })

    b_df = pd.DataFrame(records)

    # Standardize WITHIN each league
    for col in ["hp_sr", "norm_sr", "hp_bnd", "norm_bnd"]:
        b_df["z_" + col] = b_df.groupby("league")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    for col in ["hp_out", "norm_out"]:
        b_df["z_" + col] = b_df.groupby("league")[col].transform(lambda x: -(x - x.mean()) / (x.std() + 1e-6))

    b_df["z_hp"] = 0.50 * b_df["z_hp_sr"] + 0.30 * b_df["z_hp_bnd"] + 0.20 * b_df["z_hp_out"]
    b_df["z_base"] = 0.50 * b_df["z_norm_sr"] + 0.30 * b_df["z_norm_bnd"] + 0.20 * b_df["z_norm_out"]

    # Aggregate across leagues (weighted by volume)
    agg = b_df.groupby(["striker_id", "striker_name"]).apply(lambda g: pd.Series({
        "tot_hp_balls": int(g["hp_balls"].sum()),
        "tot_norm_balls": int(g["norm_balls"].sum()),
        "z_hp": np.average(g["z_hp"], weights=g["hp_balls"]),
        "z_base": np.average(g["z_base"], weights=np.maximum(g["norm_balls"], 1)),
        "avg_hp_sr": np.average(g["hp_sr"], weights=g["hp_balls"]),
        "avg_norm_sr": np.average(g["norm_sr"], weights=np.maximum(g["norm_balls"], 1)),
        "avg_hp_bnd": np.average(g["hp_bnd"], weights=g["hp_balls"]),
        "avg_hp_out": np.average(g["hp_out"], weights=g["hp_balls"])
    })).reset_index()

    # Filter to qualified volume floor (e.g. min 100 pressure balls)
    agg = agg[agg["tot_hp_balls"] >= min_hp_balls].copy()

    # ANCOVA Residual Lift (regression of pressure z-score against baseline z-score)
    slope, intercept, _, _, _ = stats.linregress(agg["z_base"], agg["z_hp"])
    agg["expected_z_hp"] = slope * agg["z_base"] + intercept
    agg["residual_lift"] = agg["z_hp"] - agg["expected_z_hp"]

    # Raw Clutch: 50% absolute high-pressure mastery + 50% residual lift
    agg["raw_clutch"] = 0.50 * agg["z_hp"] + 0.50 * agg["residual_lift"]

    # Match-to-Match Consistency Factor
    m_bat = df[df["is_hp_bat"] & df["batter_faced"]].groupby(["striker_id", "match_id"]).agg(
        m_runs=("bat_runs", "sum"),
        m_balls=("batter_faced", "sum")
    ).reset_index()
    m_bat["m_sr"] = (m_bat["m_runs"] / m_bat["m_balls"]) * 100.0
    m_bat["is_impact"] = (m_bat["m_sr"] >= 135.0) | (m_bat["m_runs"] >= 20)

    c_df = m_bat.groupby("striker_id").agg(
        tot_matches=("match_id", "count"),
        impact_matches=("is_impact", "sum")
    ).reset_index()
    c_df["impact_rate"] = c_df["impact_matches"] / c_df["tot_matches"]
    c_df["consistency_factor"] = 0.80 + 0.40 * c_df["impact_rate"]

    agg = agg.merge(c_df[["striker_id", "tot_matches", "impact_rate", "consistency_factor"]], on="striker_id", how="left")

    # Empirical Bayes Shrinkage
    agg["shrink_weight"] = agg["tot_hp_balls"] / (agg["tot_hp_balls"] + k_shrink)
    agg["clutch_score"] = agg["shrink_weight"] * agg["raw_clutch"] * agg["consistency_factor"].fillna(1.0)

    res = agg.sort_values("clutch_score", ascending=False).reset_index(drop=True)
    res["rank"] = res.index + 1
    return res


def evaluate_bowlers(df: pd.DataFrame, min_hp_balls: int = 100, k_shrink: float = 45.0) -> pd.DataFrame:
    """Evaluates bowlers using league-standardized metrics, ANCOVA residual lift,

    Empirical Bayes shrinkage, and match-to-match consistency weighting.
    """
    tau_q = df.groupby("league")["qpi"].transform(lambda x: x.quantile(0.75))
    df["is_hp_bowl"] = df["qpi"] >= tau_q

    records = []
    for (pid, pname, league), g in df.groupby(["bowler_id", "bowler_name", "league"]):
        hp = g[g["is_hp_bowl"] & g["is_legal"]]
        base = g[~g["is_hp_bowl"] & g["is_legal"]]

        hp_balls = hp["is_legal"].sum()
        norm_balls = base["is_legal"].sum()

        if hp_balls >= 10:
            hp_econ = (hp["bowler_runs"].sum() / hp_balls) * 6.0
            hp_dot = (hp["is_dot_bowler"].sum() / hp_balls) * 100.0
            hp_wkt = (hp["bowler_wicket"].sum() / hp_balls) * 100.0

            norm_econ = (base["bowler_runs"].sum() / max(norm_balls, 1)) * 6.0 if norm_balls > 0 else hp_econ
            norm_dot = (base["is_dot_bowler"].sum() / max(norm_balls, 1)) * 100.0 if norm_balls > 0 else hp_dot
            norm_wkt = (base["bowler_wicket"].sum() / max(norm_balls, 1)) * 100.0 if norm_balls > 0 else hp_wkt

            records.append({
                "bowler_id": pid, "bowler_name": pname, "league": league,
                "hp_balls": hp_balls, "norm_balls": norm_balls,
                "hp_econ": hp_econ, "norm_econ": norm_econ,
                "hp_dot": hp_dot, "norm_dot": norm_dot,
                "hp_wkt": hp_wkt, "norm_wkt": norm_wkt
            })

    b_df = pd.DataFrame(records)

    # Standardize WITHIN each league
    for col in ["hp_econ", "norm_econ"]:
        b_df["z_" + col] = b_df.groupby("league")[col].transform(lambda x: -(x - x.mean()) / (x.std() + 1e-6))
    for col in ["hp_dot", "norm_dot", "hp_wkt", "norm_wkt"]:
        b_df["z_" + col] = b_df.groupby("league")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    b_df["z_hp"] = 0.45 * b_df["z_hp_econ"] + 0.30 * b_df["z_hp_dot"] + 0.25 * b_df["z_hp_wkt"]
    b_df["z_base"] = 0.45 * b_df["z_norm_econ"] + 0.30 * b_df["z_norm_dot"] + 0.25 * b_df["z_norm_wkt"]

    agg = b_df.groupby(["bowler_id", "bowler_name"]).apply(lambda g: pd.Series({
        "tot_hp_balls": int(g["hp_balls"].sum()),
        "tot_norm_balls": int(g["norm_balls"].sum()),
        "z_hp": np.average(g["z_hp"], weights=g["hp_balls"]),
        "z_base": np.average(g["z_base"], weights=np.maximum(g["norm_balls"], 1)),
        "avg_hp_econ": np.average(g["hp_econ"], weights=g["hp_balls"]),
        "avg_norm_econ": np.average(g["norm_econ"], weights=np.maximum(g["norm_balls"], 1)),
        "avg_hp_dot": np.average(g["hp_dot"], weights=g["hp_balls"]),
        "avg_hp_wkt": np.average(g["hp_wkt"], weights=g["hp_balls"])
    })).reset_index()

    agg = agg[agg["tot_hp_balls"] >= min_hp_balls].copy()

    slope, intercept, _, _, _ = stats.linregress(agg["z_base"], agg["z_hp"])
    agg["expected_z_hp"] = slope * agg["z_base"] + intercept
    agg["residual_lift"] = agg["z_hp"] - agg["expected_z_hp"]

    agg["raw_clutch"] = 0.50 * agg["z_hp"] + 0.50 * agg["residual_lift"]

    # Match-to-Match Consistency Factor
    m_bowl = df[df["is_hp_bowl"] & df["is_legal"]].groupby(["bowler_id", "match_id"]).agg(
        m_runs=("bowler_runs", "sum"),
        m_balls=("is_legal", "sum"),
        m_wkts=("bowler_wicket", "sum")
    ).reset_index()
    m_bowl["m_econ"] = (m_bowl["m_runs"] / m_bowl["m_balls"]) * 6.0
    m_bowl["is_impact"] = (m_bowl["m_econ"] <= 8.5) | (m_bowl["m_wkts"] >= 1)

    c_df = m_bowl.groupby("bowler_id").agg(
        tot_matches=("match_id", "count"),
        impact_matches=("is_impact", "sum")
    ).reset_index()
    c_df["impact_rate"] = c_df["impact_matches"] / c_df["tot_matches"]
    c_df["consistency_factor"] = 0.80 + 0.40 * c_df["impact_rate"]

    agg = agg.merge(c_df[["bowler_id", "tot_matches", "impact_rate", "consistency_factor"]], on="bowler_id", how="left")

    agg["shrink_weight"] = agg["tot_hp_balls"] / (agg["tot_hp_balls"] + k_shrink)
    agg["clutch_score"] = agg["shrink_weight"] * agg["raw_clutch"] * agg["consistency_factor"].fillna(1.0)

    res = agg.sort_values("clutch_score", ascending=False).reset_index(drop=True)
    res["rank"] = res.index + 1
    return res


def generate_visualizations(df_deliveries: pd.DataFrame, top_batters: pd.DataFrame, top_bowlers: pd.DataFrame, fig_dir: Path):
    """Generates publication-quality charts for the methodology report and outputs."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. BPI & QPI Distribution
    plt.figure(figsize=(10, 4.5))
    plt.hist(df_deliveries["bpi"], bins=50, density=True, alpha=0.4, color="#d9534f", label="Batting Pressure Index (BPI)")
    plt.hist(df_deliveries["qpi"], bins=50, density=True, alpha=0.4, color="#0275d8", label="Bowling Pressure Index (QPI)")
    plt.axvline(df_deliveries["bpi"].quantile(0.75), color="#d9534f", linestyle="--", linewidth=1.5, label="Batting 75th %ile Threshold")
    plt.axvline(df_deliveries["qpi"].quantile(0.75), color="#0275d8", linestyle="--", linewidth=1.5, label="Bowling 75th %ile Threshold")
    plt.title("Distribution of Match Pressure Indices (BPI vs QPI)", fontsize=13, fontweight="bold")
    plt.xlabel("Pressure Index [0 - 1]")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_dir / "bpi_qpi_distribution.png", dpi=300)
    plt.close()

    # 2. Top 10 Batters Leaderboard
    plt.figure(figsize=(10, 5.5))
    top10_bat = top_batters.head(10).sort_values("clutch_score", ascending=True)
    bars = plt.barh(top10_bat["striker_name"], top10_bat["clutch_score"], color="#d9534f")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9, fontweight="bold")
    plt.title("SupeRR Selector Season 2: Top 10 Batters Under Pressure", fontsize=13, fontweight="bold")
    plt.xlabel("Calibrated Clutch Score (CCS)")
    plt.xlim(0, max(top10_bat["clutch_score"]) * 1.15)
    plt.tight_layout()
    plt.savefig(fig_dir / "top_10_batters_chart.png", dpi=300)
    plt.close()

    # 3. Top 10 Bowlers Leaderboard
    plt.figure(figsize=(10, 5.5))
    top10_bowl = top_bowlers.head(10).sort_values("clutch_score", ascending=True)
    bars = plt.barh(top10_bowl["bowler_name"], top10_bowl["clutch_score"], color="#0275d8")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9, fontweight="bold")
    plt.title("SupeRR Selector Season 2: Top 10 Bowlers Under Pressure", fontsize=13, fontweight="bold")
    plt.xlabel("Calibrated Clutch Score (CCS)")
    plt.xlim(0, max(top10_bowl["clutch_score"]) * 1.15)
    plt.tight_layout()
    plt.savefig(fig_dir / "top_10_bowlers_chart.png", dpi=300)
    plt.close()

    # 4. Batter Strike Rate Scatter: Normal vs High Pressure
    plt.figure(figsize=(8, 6))
    plt.scatter(top_batters["avg_norm_sr"], top_batters["avg_hp_sr"], c=top_batters["clutch_score"], cmap="Reds", s=top_batters["tot_hp_balls"]*0.6, alpha=0.7, edgecolors="grey")
    lims = [100, 220]
    plt.plot(lims, lims, "k--", alpha=0.5, label="Parity (No Elevation)")
    for _, r in top_batters.head(10).iterrows():
        plt.annotate(r["striker_name"], (r["avg_norm_sr"]+1, r["avg_hp_sr"]+1), fontsize=8, fontweight="bold")
    plt.title("Batters: Normal Strike Rate vs High-Pressure Strike Rate", fontsize=12, fontweight="bold")
    plt.xlabel("Normal Baseline Strike Rate")
    plt.ylabel("High-Pressure Strike Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "batter_clutch_scatter.png", dpi=300)
    plt.close()

    # 5. Bowler Economy Scatter: Normal vs High Pressure
    plt.figure(figsize=(8, 6))
    plt.scatter(top_bowlers["avg_norm_econ"], top_bowlers["avg_hp_econ"], c=top_bowlers["clutch_score"], cmap="Blues", s=top_bowlers["tot_hp_balls"]*0.5, alpha=0.7, edgecolors="grey")
    lims = [5, 12]
    plt.plot(lims, lims, "k--", alpha=0.5, label="Parity (No Elevation)")
    for _, r in top_bowlers.head(10).iterrows():
        plt.annotate(r["bowler_name"], (r["avg_norm_econ"]+0.1, r["avg_hp_econ"]+0.1), fontsize=8, fontweight="bold")
    plt.title("Bowlers: Normal Economy vs High-Pressure Economy", fontsize=12, fontweight="bold")
    plt.xlabel("Normal Baseline Economy")
    plt.ylabel("High-Pressure Economy (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bowler_clutch_scatter.png", dpi=300)
    plt.close()


def run():
    """Runs complete model training, evaluation, outputs export, and visualization generation."""
    print("Loading processed delivery features...")
    processed_path = C.PROCESSED / "deliveries_pressure.parquet"
    if not processed_path.exists():
        print("Processed deliveries not found. Generating now...")
        df_deliveries = features.run_feature_pipeline()
    else:
        df_deliveries = pd.read_parquet(processed_path)

    print("Evaluating Batters...")
    batters = evaluate_batters(df_deliveries, min_hp_balls=100, k_shrink=45.0)

    print("Evaluating Bowlers...")
    bowlers = evaluate_bowlers(df_deliveries, min_hp_balls=100, k_shrink=45.0)

    # Save CSV and JSON outputs
    C.OUT.mkdir(parents=True, exist_ok=True)
    fig_dir = C.OUT / "figures"

    # Export Top 10 Batters
    top10_bat = batters.head(10)[[
        "rank", "striker_name", "tot_hp_balls", "avg_hp_sr", "avg_norm_sr",
        "avg_hp_bnd", "residual_lift", "impact_rate", "clutch_score"
    ]].copy()
    top10_bat.to_csv(C.OUT / "top_10_batters.csv", index=False)
    with open(C.OUT / "top_10_batters.json", "w") as f:
        json.dump(top10_bat.to_dict(orient="records"), f, indent=2)

    # Export Top 10 Bowlers
    top10_bowl = bowlers.head(10)[[
        "rank", "bowler_name", "tot_hp_balls", "avg_hp_econ", "avg_norm_econ",
        "avg_hp_dot", "avg_hp_wkt", "residual_lift", "impact_rate", "clutch_score"
    ]].copy()
    top10_bowl.to_csv(C.OUT / "top_10_bowlers.csv", index=False)
    with open(C.OUT / "top_10_bowlers.json", "w") as f:
        json.dump(top10_bowl.to_dict(orient="records"), f, indent=2)

    # Export all qualified players
    batters.to_csv(C.OUT / "all_qualified_batters.csv", index=False)
    bowlers.to_csv(C.OUT / "all_qualified_bowlers.csv", index=False)

    print("Generating visualizations...")
    generate_visualizations(df_deliveries, batters, bowlers, fig_dir)

    print("\n========================================================")
    print("🏆 RAJASTHAN ROYALS SUPERR SELECTOR S2: TOP 10 BATTERS")
    print("========================================================")
    print(top10_bat.to_string(index=False))

    print("\n========================================================")
    print("🏆 RAJASTHAN ROYALS SUPERR SELECTOR S2: TOP 10 BOWLERS")
    print("========================================================")
    print(top10_bowl.to_string(index=False))

    return batters, bowlers


if __name__ == "__main__":
    run()
