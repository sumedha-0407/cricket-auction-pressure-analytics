"""Central settings. Change numbers here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

# ---- cleaning rules (verified on the data pack) ----
BALLS_PER_INNINGS = 120
# rows that are NOT a legal delivery (wide, no-ball, 'other' = no-ball variant, penalty runs)
ILLEGAL_EXTRAS = {"wide", "Overstep", "other", "Penalty"}
NOBALL_EXTRAS = {"Overstep", "other"}
# extras that are NOT charged to the bowler
NOT_BOWLER_EXTRAS = {"byes", "legbyes", "Penalty"}
# dismissals that count as a bowler's wicket
BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}

# ---- scope ----
# league tag: IPL = comp_name starts with "IPL"; T20I = match_format T20I; everything else = OTHER
# (OTHER = Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025 -> excluded from ranking)
RANK_LEAGUES = ["IPL", "T20I"]
NO_RESULT_TEXT = ("No Result", "Match Abandoned")
