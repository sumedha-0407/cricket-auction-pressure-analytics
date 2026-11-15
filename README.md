# 🏏 Rajasthan Royals SupeRR Selector Season 2
## Identify the Most Reliable Players Under Pressure

A mathematically rigorous, leak-free, and peer-benchmarked sports analytics pipeline built for the **Rajasthan Royals SupeRR Selector Season 2** hackathon. This model evaluates T20 cricket players in high-leverage match scenarios across historical IPL and T20 International data (2022–2026), balancing situational difficulty, individual baselines, and sample reliability to produce a model that is **neither underfit nor overfit**.

---

## 🏆 Official Competition Submissions

### Top 10 Batters Under Pressure

| Rank | Player Name | HP Balls Faced | HP Strike Rate | Normal Strike Rate | Boundary % | Residual Lift | Impact Rate | Calibrated Clutch Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Naman Dhir** | 150 | **206.0** | 155.1 | 30.0% | +1.251 | 76.5% | **1.217** |
| **2** | **Jos Buttler** | 180 | **199.4** | 152.9 | 29.4% | +1.073 | 70.6% | **1.086** |
| **3** | **Suryakumar Yadav** | 475 | **177.7** | 164.0 | 27.0% | +0.785 | 78.9% | **0.992** |
| **4** | **Mark Chapman** | 213 | **175.6** | 134.5 | 23.0% | +0.884 | 75.0% | **0.958** |
| **5** | **Glenn Maxwell** | 163 | **179.1** | 156.6 | 27.6% | +0.897 | 81.3% | **0.951** |
| **6** | **Nicholas Pooran** | 461 | **180.3** | 150.2 | 25.2% | +0.765 | 71.4% | **0.934** |
| **7** | **Fakhar Zaman** | 101 | **185.1** | 124.3 | 26.7% | +1.083 | 69.2% | **0.930** |
| **8** | **Reeza Hendricks** | 123 | **179.7** | 141.5 | 25.2% | +0.969 | 71.4% | **0.911** |
| **9** | **Tim David** | 446 | **178.7** | 175.8 | 24.4% | +0.683 | 65.2% | **0.845** |
| **10** | **Matthew Wade** | 167 | **161.1** | 126.4 | 20.4% | +0.679 | 76.9% | **0.737** |

---

### Top 10 Bowlers Under Pressure

| Rank | Player Name | HP Legal Balls | HP Economy | Normal Economy | HP Dot % | HP Wicket % | Residual Lift | Impact Rate | Calibrated Clutch Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Noor Ahmad** | 244 | **8.29** | 7.52 | 40.2% | 7.79% | +0.764 | 66.7% | **0.750** |
| **2** | **Roston Chase** | 109 | **7.49** | 7.55 | 35.8% | 10.09% | +0.753 | 100.0% | **0.667** |
| **3** | **Ben Dwarshuis** | 139 | **7.90** | 9.18 | 38.8% | 9.35% | +0.728 | 83.3% | **0.648** |
| **4** | **Taskin Ahmed** | 259 | **8.04** | 7.56 | 40.9% | 8.11% | +0.605 | 80.0% | **0.643** |
| **5** | **Sunil Narine** | 256 | **7.66** | 6.49 | 36.3% | 4.30% | +0.532 | 65.5% | **0.567** |
| **6** | **Tim Southee** | 161 | **8.35** | 7.89 | 39.8% | 7.45% | +0.550 | 68.4% | **0.523** |
| **7** | **Jasprit Bumrah** | 552 | **7.61** | 6.42 | 35.1% | 4.35% | +0.395 | 75.5% | **0.523** |
| **8** | **Eshan Malinga** | 233 | **8.47** | 9.26 | 33.9% | 7.30% | +0.487 | 90.0% | **0.510** |
| **9** | **Corbin Bosch** | 101 | **9.62** | 7.74 | 42.6% | 10.89% | +0.571 | 72.7% | **0.509** |
| **10** | **Rashid Khan** | 504 | **8.35** | 7.08 | 32.9% | 7.34% | +0.418 | 80.3% | **0.507** |

---

## 🔬 Core Modeling Innovations

1. **Leak-Free Pre-Delivery State Tracking (`features.py`)**:
   - Implements strict `.shift(1)` operations grouped by `(match_id, innings)` to prevent look-ahead contamination.
   - Pre-delivery collapse window (`wickets_last_18`) is strictly lagged to prevent current ball dismissal leakage.
2. **Asymmetric Pressure Modeling (BPI vs QPI)**:
   - Batters and bowlers experience different situational pressure on the same delivery.
   - **Batting Pressure Index ($BPI$)**: Captures steep RRR chases, wickets lost (diminishing depth), ball scarcity, and 1st innings par deficits.
   - **Bowling Pressure Index ($QPI$)**: Captures defending margin volatility, death-over slog threat, and bowling to established top-order batters with wickets in hand.
3. **Elimination of the "Elite Penalty" (ANCOVA Residual Lift)**:
   - Naive difference models ($\Delta = \text{Pressure} - \text{Normal}$) heavily penalize superstars whose baselines are already dominant (e.g. Jasprit Bumrah conceding 7.6 RPO under death pressure vs 6.4 baseline is labeled negative).
   - Our **Residual Lift Model** measures performance relative to regression expectations ($\beta \approx 0.34$), properly rewarding elite performers while eliminating the floor bias of mediocre players.
4. **League-Standardized Calibration (IPL vs T20I)**:
   - Normalizes scoring environments independently within IPL and T20Is before aggregating, ensuring true like-with-like comparisons.
5. **Empirical Bayes (James-Stein) Shrinkage & Consistency**:
   - Enforces a **100 high-pressure delivery qualification floor** and shrinks low-sample metrics with prior constant $K = 45.0$.
   - Incorporates a **Match-to-Match Consistency Factor ($C_i$)** to reward players who deliver repeatedly across multiple matches rather than on single-match heroics.

---

## 📁 Repository Structure

```text
cricket-auction-pressure-analytics/
├── data/
│   ├── raw/                 # Original competition datasets (CSVs)
│   ├── interim/             # Cleaned Parquet tables (matches, balls, players)
│   └── processed/           # Feature-engineered deliveries with BPI & QPI
├── notebooks/
│   └── SupeRR_Pressure_Analytics_Insights.ipynb  # Comprehensive narrative analysis notebook
├── outputs/
│   ├── figures/             # High-resolution charts (distributions, scatter plots, bar charts)
│   ├── top_10_batters.csv   # Dropdown-ready ordered top-10 batters
│   ├── top_10_batters.json  # JSON schema for top-10 batters
│   ├── top_10_bowlers.csv   # Dropdown-ready ordered top-10 bowlers
│   ├── top_10_bowlers.json  # JSON schema for top-10 bowlers
│   ├── all_qualified_batters.csv  # Full candidate pool metrics
│   ├── all_qualified_bowlers.csv  # Full candidate pool metrics
│   └── Methodology_Report.md      # Complete competition document upload
├── scripts/
│   ├── audit_pipeline.py    # Zero-leakage & vulnerability verification
│   ├── build_notebook.py    # Generates Jupyter notebook
│   └── run_pipeline.py      # End-to-end execution runner
├── src/
│   └── pressure/
│       ├── clean.py         # Filtering super overs, DLS, illegal extras, bowler wickets
│       ├── config.py        # Central configuration settings
│       ├── features.py      # Pre-delivery state tracking, par curve, BPI & QPI
│       ├── load.py          # Relational data loader
│       └── models.py        # ANCOVA residual lift, Empirical Bayes shrinkage, rankings
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Pipeline

To execute the entire end-to-end pipeline in a single command:

```bash
python scripts/run_pipeline.py
```

To run the automated vulnerability and leakage audit:

```bash
python scripts/audit_pipeline.py
```
