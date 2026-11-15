"""src/pressure/features.py

Leak-Free Feature Engineering & Pressure Formulation for SupeRR Selector Season 2.
Computes pre-delivery match states and distinct, mathematically rigorous
Batting Pressure Index (BPI) and Bowling Pressure Index (QPI).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C
from . import clean


def compute_empirical_par_curve(df_1st: pd.DataFrame) -> dict:
    """Computes empirical average runs scored before each over (over0 in 0..19)

    separately for IPL and T20I to accurately anchor 1st innings par pacing.
    """
    par_map = {}
    for league in ["IPL", "T20I"]:
        sub = df_1st[df_1st["league"] == league]
        if len(sub) > 0:
            means = sub.groupby("over0")["runs_before"].mean().to_dict()
            par_map[league] = {o: means.get(o, o * 8.2) for o in range(20)}
        else:
            par_map[league] = {o: o * 8.2 for o in range(20)}
    return par_map


def engineer_pre_delivery_states(df: pd.DataFrame) -> pd.DataFrame:
    """Computes strict pre-delivery states using .shift(1) grouped by match_id and innings.

    Guarantees zero look-ahead bias or cross-innings bleeding.
    """
    # Sort chronologically by match, innings, and delivery sequence
    df = df.sort_values(by=["match_id", "innings", "innings_seq"]).reset_index(drop=True)

    # Group by match and innings
    grp = df.groupby(["match_id", "innings"], sort=False)

    # Cumulative stats including current delivery
    cum_runs = grp["total_runs"].cumsum()
    cum_wickets = grp["is_wicket"].cumsum()
    cum_legal = grp["is_legal"].cumsum()

    # Pre-delivery state (strictly shifted by 1)
    df["runs_before"] = grp["total_runs"].cumsum().groupby([df["match_id"], df["innings"]]).shift(1).fillna(0)
    df["wickets_before"] = grp["is_wicket"].cumsum().groupby([df["match_id"], df["innings"]]).shift(1).fillna(0)
    df["legal_balls_before"] = grp["is_legal"].cumsum().groupby([df["match_id"], df["innings"]]).shift(1).fillna(0)

    # Wickets remaining and lost
    df["wickets_lost"] = df["wickets_before"].clip(0, 10)
    df["wickets_in_hand"] = (10 - df["wickets_lost"]).clip(0, 10)

    # Balls remaining (120 legal balls in full T20 innings)
    df["balls_remaining"] = (120.0 - df["legal_balls_before"]).clip(lower=0.0)

    # Tactical Phase
    df["phase"] = np.select(
        [df["over0"] < 6, df["over0"] < 15],
        ["Powerplay", "Middle"],
        default="Death"
    )

    # Pre-delivery rolling collapse / momentum (strictly lagged via shift(1))
    shifted_wkt = grp["is_wicket"].shift(1).fillna(0)
    shifted_runs = grp["total_runs"].shift(1).fillna(0)

    df["wickets_last_18"] = shifted_wkt.groupby([df["match_id"], df["innings"]]).rolling(18, min_periods=1).sum().reset_index(drop=True).fillna(0)
    df["runs_last_12"] = shifted_runs.groupby([df["match_id"], df["innings"]]).rolling(12, min_periods=1).sum().reset_index(drop=True).fillna(0)

    return df


def compute_batting_pressure_index(df: pd.DataFrame, par_curve: dict) -> pd.DataFrame:
    """Computes Batting Pressure Index (BPI in [0, 1]).

    Reflects:
      - 2nd innings: RRR stress, wickets lost (diminishing depth), time running out, contest closeness.
      - 1st innings: Par deficit, recent collapse, death acceleration urgency, wickets lost.
    """
    df["bpi"] = 0.0

    # --- Innings 2: The Chase ---
    mask_2nd = df["innings"] == 2
    if mask_2nd.any():
        # First innings total score per match + 1
        inn1_totals = df[df["innings"] == 1].groupby("match_id")["total_runs"].sum().to_dict()
        targets = df.loc[mask_2nd, "match_id"].map(inn1_totals).fillna(160) + 1.0

        runs_req = (targets - df.loc[mask_2nd, "runs_before"]).clip(lower=0.0)
        balls_rem = df.loc[mask_2nd, "balls_remaining"].clip(lower=1.0)
        legal_bowled = df.loc[mask_2nd, "legal_balls_before"].clip(lower=1.0)

        rrr = (runs_req / balls_rem) * 6.0
        crr = (df.loc[mask_2nd, "runs_before"] / legal_bowled) * 6.0

        # Sub-indicators for batting pressure
        # 1. RRR Stress: Sigmoidal scaling centered around 9.5 RPO
        s_rrr = 1.0 / (1.0 + np.exp(-0.5 * (rrr - 9.5)))

        # 2. Wickets Lost (CORRECT: More wickets lost = Higher batting pressure!)
        # (wickets_lost / 10)^1.4
        s_wkts = (df.loc[mask_2nd, "wickets_lost"] / 10.0) ** 1.4

        # 3. Balls Remaining Scarcity
        s_balls = 1.0 - (df.loc[mask_2nd, "balls_remaining"] / 120.0)

        # 4. Contest Closeness (death overs swing)
        s_close = np.exp(-0.15 * np.abs(rrr - crr)) * (df.loc[mask_2nd, "balls_remaining"] <= 36).astype(float)

        bpi_2 = 0.40 * s_rrr + 0.30 * s_wkts + 0.15 * s_balls + 0.15 * s_close
        df.loc[mask_2nd, "bpi"] = np.clip(bpi_2, 0.0, 1.0)
        df.loc[mask_2nd, "runs_required"] = runs_req
        df.loc[mask_2nd, "rrr"] = rrr
        df.loc[mask_2nd, "crr"] = crr

    # --- Innings 1: Setting the Total ---
    mask_1st = df["innings"] == 1
    if mask_1st.any():
        league_col = df.loc[mask_1st, "league"]
        over_col = df.loc[mask_1st, "over0"]
        
        # Look up expected runs before this over
        exp_par = [par_curve.get(l, {}).get(o, o * 8.2) for l, o in zip(league_col, over_col)]
        exp_par = pd.Series(exp_par, index=df[mask_1st].index)

        par_deficit = (exp_par - df.loc[mask_1st, "runs_before"]).clip(lower=0.0)
        f_par = np.clip(par_deficit / 35.0, 0.0, 1.0)

        # Wickets lost: More wickets down = higher pressure on remaining batters
        f_wkts = (df.loc[mask_1st, "wickets_lost"] / 10.0) ** 1.4

        # Recent collapse (wickets in last 18 balls)
        f_collapse = np.clip(df.loc[mask_1st, "wickets_last_18"] / 3.0, 0.0, 1.0)

        # Phase urgency
        phase_map = {"Powerplay": 0.25, "Middle": 0.45, "Death": 0.85}
        f_stage = df.loc[mask_1st, "phase"].map(phase_map).fillna(0.45)

        bpi_1 = 0.30 * f_par + 0.30 * f_wkts + 0.25 * f_collapse + 0.15 * f_stage
        df.loc[mask_1st, "bpi"] = np.clip(bpi_1, 0.0, 1.0)

    return df


def compute_bowling_pressure_index(df: pd.DataFrame, par_curve: dict) -> pd.DataFrame:
    """Computes Bowling Pressure Index (QPI in [0, 1]).

    Reflects:
      - 2nd innings: Defending tight margin, high contest volatility, bowling death overs,
                     opposition threat (batting team having wickets in hand).
      - 1st innings: Stemming opposition onslaught (batting team ahead of par), Powerplay/Death restrictions,
                     countering momentum.
    """
    df["qpi"] = 0.0

    # --- Innings 2: Defending ---
    mask_2nd = df["innings"] == 2
    if mask_2nd.any():
        rrr = df.loc[mask_2nd, "rrr"]
        crr = df.loc[mask_2nd, "crr"]
        balls_rem = df.loc[mask_2nd, "balls_remaining"]
        wkts_in_hand = df.loc[mask_2nd, "wickets_in_hand"]

        # 1. Volatility in close death finish
        q_volatility = np.exp(-0.15 * np.abs(rrr - crr)) * (balls_rem <= 36).astype(float)

        # 2. Death over execution stress (overs 16-20)
        q_death = np.clip((df.loc[mask_2nd, "over0"] - 14.0) / 5.0, 0.0, 1.0)

        # 3. Opposition Threat: High wickets in hand with attainable RRR creates peak bowling anxiety
        q_threat = np.clip(1.0 - (rrr - 6.0) / 10.0, 0.0, 1.0) * (wkts_in_hand / 10.0)

        # 4. Critical Breakthrough Demand (batting team established, bowler must take a wicket)
        q_wkt_demand = (wkts_in_hand / 10.0) ** 1.3 * (1.0 - balls_rem / 120.0)

        qpi_2 = 0.35 * q_volatility + 0.25 * q_death + 0.25 * q_threat + 0.15 * q_wkt_demand
        df.loc[mask_2nd, "qpi"] = np.clip(qpi_2, 0.0, 1.0)

    # --- Innings 1: Bowling First ---
    mask_1st = df["innings"] == 1
    if mask_1st.any():
        league_col = df.loc[mask_1st, "league"]
        over_col = df.loc[mask_1st, "over0"]
        exp_par = [par_curve.get(l, {}).get(o, o * 8.2) for l, o in zip(league_col, over_col)]
        exp_par = pd.Series(exp_par, index=df[mask_1st].index)

        # 1. Onslaught Pressure: Batting team is way ahead of par with wickets in hand
        par_surplus = (df.loc[mask_1st, "runs_before"] - exp_par).clip(lower=0.0)
        q_onslaught = np.clip(par_surplus / 35.0, 0.0, 1.0) * (df.loc[mask_1st, "wickets_in_hand"] / 10.0)

        # 2. Phase Execution Pressure: Powerplay (field restrictions) and Death (slog)
        phase_risk = {"Powerplay": 0.60, "Middle": 0.30, "Death": 0.85}
        q_phase = df.loc[mask_1st, "phase"].map(phase_risk).fillna(0.35)

        # 3. Momentum Stemming: Opponent scored heavily in the last 12 balls
        q_momentum = np.clip(df.loc[mask_1st, "runs_last_12"] / 24.0, 0.0, 1.0)

        # 4. Opposition Wickets in hand factor
        q_depth = (df.loc[mask_1st, "wickets_in_hand"] / 10.0) ** 1.2

        qpi_1 = 0.35 * q_onslaught + 0.30 * q_phase + 0.20 * q_momentum + 0.15 * q_depth
        df.loc[mask_1st, "qpi"] = np.clip(qpi_1, 0.0, 1.0)

    return df


def run_feature_pipeline() -> pd.DataFrame:
    """Full feature engineering pipeline: loads interim clean tables,

    computes leak-free pre-delivery state and separate BPI/QPI pressure indices,
    and persists processed parquet table.
    """
    interim_balls = C.INTERIM / "balls.parquet"
    if not interim_balls.exists():
        print("Interim tables not found. Running clean.run()...")
        clean.run()

    print("Loading clean deliveries from interim...")
    df = pd.read_parquet(interim_balls)

    # Filter strictly to usable matches (IPL & T20I, non-DLS, non-abandoned)
    df = df[df["usable"]].copy()

    print(f"Engineering pre-delivery state for {len(df):,} usable deliveries...")
    df = engineer_pre_delivery_states(df)

    # Compute empirical par curve from 1st innings data
    par_curve = compute_empirical_par_curve(df[df["innings"] == 1])

    print("Computing Batting Pressure Index (BPI)...")
    df = compute_batting_pressure_index(df, par_curve)

    print("Computing Bowling Pressure Index (QPI)...")
    df = compute_bowling_pressure_index(df, par_curve)

    # Save to processed
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = C.PROCESSED / "deliveries_pressure.parquet"
    print(f"Saving processed deliveries to {out_path}...")
    df.to_parquet(out_path, index=False)
    print(f"Feature pipeline successfully finished. Processed shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_feature_pipeline()
