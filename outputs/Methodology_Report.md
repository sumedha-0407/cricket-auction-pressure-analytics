# SupeRR Selector Season 2: Methodology Report
## Identifying the Most Reliable T20 Players Under Pressure

**Submitted for:** Rajasthan Royals — SupeRR Selector Hackathon (Round 1)  
**Author / Team:** Cricket Pressure Analytics  
**Date:** November 2026  
**Scope:** Historical IPL & T20 International (T20I) Ball-by-Ball Records (2022–2026)

---

## Executive Summary

Cricket match outcomes are disproportionately decided in high-leverage moments—chasing steep targets against escalating required run rates, recovering from sudden top-order collapses, defending razor-thin death-over margins, or executing yorkers under intense crowd scrutiny. However, traditional cricket metrics (aggregate strike rates, raw batting averages, and unweighted bowling economies) are context-blind and suffer from severe sample distortions.

This methodology presents a leak-free, mathematically rigorous, and peer-benchmarked player evaluation architecture designed to identify the **Top 10 Batters** and **Top 10 Bowlers** who consistently rise to the occasion. 

### Key Modeling Innovations
1. **Asymmetric Pressure Modeling:** Batters and bowlers experience different psychological and situational stress on the same delivery. We construct separate **Batting Pressure Index ($BPI \in [0, 1]$)** and **Bowling Pressure Index ($QPI \in [0, 1]$)** formulations.
2. **Elimination of the "Elite Penalty":** Naive clutch formulations ($\Delta = \text{Pressure} - \text{Normal}$) severely penalize world-class players whose normal baseline is already dominant (e.g. rating Jasprit Bumrah as un-clutch because his baseline economy is 6.4 and his pressure economy is 7.6). We employ an **ANCOVA Residual Lift Model** that evaluates performance relative to regression-to-the-mean expectations.
3. **League-Standardized Calibration:** IPL pitches (average scores ~180–210) and international T20I surfaces (average scores ~140–165) are standardized within their respective cohorts to ensure fair, like-with-like comparisons.
4. **Empirical Bayes (James-Stein) Shrinkage:** To eliminate small-sample noise and "one-hit wonders," player performance estimates are smoothed toward the cohort prior, anchored by a minimum sample floor ($N_{\text{pressure}} \ge 100$ legal balls).
5. **Match-to-Match Consistency Regularization:** Players are rewarded for delivering across multiple independent fixtures rather than riding on a single freak innings.

---

## 1. Pressure Definition & Match Context Dynamics

### 1.1 Why Batters and Bowlers Feel Pressure Differently
In a chase where the batting team requires 14 off 6 balls with 8 wickets down:
* **The Batter** faces acute pressure ($BPI \approx 0.92$): time is running out, every dot ball exponentially drives up the required rate, and a single mistake ends the match.
* **The Bowler**, conversely, enjoys defensive leverage ($QPI \approx 0.40$): conceding 8 or 9 runs still guarantees victory, and batting depth has evaporated.

Conversely, when the batting team needs 18 off 12 balls with 8 wickets in hand:
* **The Batter** is comfortable ($BPI \approx 0.35$): required rate is accessible, and wickets remain in abundance.
* **The Bowler** is under maximum stress ($QPI \approx 0.88$): bowling with no margin for error against set batters, where one boundary breaks open the game.

### 1.2 Mathematical Formulation of Pressure Indices

All match states are computed strictly **pre-delivery** using a grouped `.shift(1)` operation across `(match_id, innings)` to guarantee zero lookahead leakage.

#### A. Batting Pressure Index ($BPI \in [0, 1]$)
* **Second Innings (The Chase):**
  $$BPI_2 = 0.40 \cdot S_{\text{RRR}} + 0.30 \cdot S_{\text{wkts}} + 0.15 \cdot S_{\text{balls}} + 0.15 \cdot S_{\text{close}}$$
  Where:
  * $S_{\text{RRR}} = \frac{1}{1 + e^{-0.5 \cdot (\text{RRR} - 9.5)}}$ (Sigmoid scaling centered at 9.5 RPO).
  * $S_{\text{wkts}} = \left(\frac{\text{Wickets Lost}}{10}\right)^{1.4}$ (Monotonically increases as wickets fall; 0 wickets down = 0.0, 9 wickets down = 0.87).
  * $S_{\text{balls}} = 1.0 - \frac{\text{Balls Remaining}}{120.0}$.
  * $S_{\text{close}} = e^{-0.15 \cdot |\text{RRR} - \text{CRR}|} \cdot \mathbf{1}_{\{\text{Balls Remaining} \le 36\}}$.

* **First Innings (Setting the Target):**
  $$BPI_1 = 0.30 \cdot F_{\text{par}} + 0.30 \cdot F_{\text{wkts}} + 0.25 \cdot F_{\text{collapse}} + 0.15 \cdot F_{\text{stage}}$$
  Where:
  * $F_{\text{par}} = \text{clip}\left(\frac{\text{Par}(\text{over0}) - \text{Runs Before}}{35.0}, 0, 1\right)$ anchored to empirical league-phase par curves.
  * $F_{\text{wkts}} = \left(\frac{\text{Wickets Lost}}{10}\right)^{1.4}$.
  * $F_{\text{collapse}} = \text{clip}\left(\frac{\text{Wickets Last 18 Balls}}{3.0}, 0, 1\right)$ (lagged pre-delivery).
  * $F_{\text{stage}} \in \{0.25 \text{ (Powerplay)}, 0.45 \text{ (Middle)}, 0.85 \text{ (Death)}\}$.

#### B. Bowling Pressure Index ($QPI \in [0, 1]$)
* **Second Innings (Defending):**
  $$QPI_2 = 0.35 \cdot Q_{\text{volatility}} + 0.25 \cdot Q_{\text{death}} + 0.25 \cdot Q_{\text{threat}} + 0.15 \cdot Q_{\text{wkt\_demand}}$$
  Where:
  * $Q_{\text{volatility}} = e^{-0.15 \cdot |\text{RRR} - \text{CRR}|} \cdot \mathbf{1}_{\{\text{Balls Remaining} \le 36\}}$.
  * $Q_{\text{death}} = \text{clip}\left(\frac{\text{over0} - 14}{5.0}, 0, 1\right)$.
  * $Q_{\text{threat}} = \text{clip}\left(1.0 - \frac{\text{RRR} - 6.0}{10.0}, 0, 1\right) \times \left(\frac{\text{Wickets In Hand}}{10}\right)^{1.2}$.
  * $Q_{\text{wkt\_demand}} = \left(\frac{\text{Wickets In Hand}}{10}\right)^{1.3} \times \left(1.0 - \frac{\text{Balls Remaining}}{120.0}\right)$.

* **First Innings (Bowling First):**
  $$QPI_1 = 0.35 \cdot Q_{\text{onslaught}} + 0.30 \cdot Q_{\text{phase}} + 0.20 \cdot Q_{\text{momentum}} + 0.15 \cdot Q_{\text{depth}}$$
  Where $Q_{\text{onslaught}}$ penalizes teams allowing the opponent to surge ahead of par with wickets in hand, and $Q_{\text{phase}}$ accounts for field restriction vulnerabilities.

* **Threshold Definition:** Deliveries at or above the **75th percentile** within each discipline ($BPI \ge 0.418$, $QPI \ge 0.455$) are classified as **High Pressure**, while deliveries below serve as each player's **Normal Baseline**.

---

## 2. Performance Measurement & Scoring Logic

### 2.1 Discipline-Specific Metrics
* **Batters:**
  1. **Strike Rate (SR)**: $\frac{\text{Runs}}{\text{Balls Faced}} \times 100$ (Weight: 50%)
  2. **Boundary Rate (Bnd%)**: $\frac{\text{Fours} + \text{Sixes}}{\text{Balls Faced}} \times 100$ (Weight: 30%)
  3. **Dismissal Resistance (-Out%)**: $-\frac{\text{Dismissals}}{\text{Balls Faced}} \times 100$ (Weight: 20%)

* **Bowlers:**
  1. **Economy Rate (-Econ)**: $-\frac{\text{Bowler Runs}}{\text{Legal Balls}} \times 6.0$ (Weight: 45%) — *excludes byes/leg-byes*
  2. **Dot Ball Rate (Dot%)**: $\frac{\text{Dot Balls}}{\text{Legal Balls}} \times 100$ (Weight: 30%)
  3. **Wicket Strike Rate (Wkt%)**: $\frac{\text{Bowler Wickets}}{\text{Legal Balls}} \times 100$ (Weight: 25%) — *excludes run-outs*

### 2.2 Solving the Elite Penalty via ANCOVA Residual Lift
In empirical testing of naive difference models ($\Delta = Z_{\text{pressure}} - Z_{\text{baseline}}$), an acute flaw was revealed:
Because performance naturally regresses toward the mean under extreme pressure ($\beta \approx 0.34$), naive subtraction heavily penalizes world-class players whose baseline is already at the 95th percentile. Under naive subtraction, Jasprit Bumrah received a negative clutch score (-0.011) and fell to Rank #63, while below-average bowlers who merely jumped from terrible to mediocre scored high positive deltas.

To correct this, we implement **Regression-Adjusted Residual Lift**:
$$Z_{\text{pressure}, i} = \alpha + \beta \cdot Z_{\text{baseline}, i} + \epsilon_i$$
$$\text{Residual Lift}_i = Z_{\text{pressure}, i} - \left(\hat{\beta} \cdot Z_{\text{baseline}, i} + \hat{\alpha}\right)$$
* Players who perform above the regression expectation given their baseline achieve positive lift ($\text{Lift} > 0$).
* World-class baseline performers who sustain elite execution in pressure conditions are rewarded rather than penalized.

The combined **Raw Clutch Score** is:
$$\text{Raw Clutch}_i = 0.50 \cdot Z_{\text{pressure}, i} + 0.50 \cdot \text{Residual Lift}_i$$

---

## 3. Fair Comparisons, Shrinkage & Regularization

### 3.1 League Calibration (IPL vs T20I)
IPL matches feature higher run-scoring environments (average pressure economy: 10.67 RPO) compared to international T20I fixtures (average pressure economy: 9.85 RPO). Comparing raw figures directly introduces severe structural bias.
Our model **standardizes performance within each league independently** ($Z_{\text{IPL}}$ and $Z_{\text{T20I}}$) before aggregating across competitions via volume-weighted means.

### 3.2 Sample Size Threshold & Empirical Bayes Shrinkage
To guard against small-sample noise and volatile cameos:
1. **Qualification Floor:** A minimum of **100 high-pressure legal deliveries** across the 2022–2026 dataset (ensuring at least ~16 overs of high-stress experience).
2. **Empirical Bayes (James-Stein) Shrinkage:**
   $$\lambda_i = \frac{N_i}{N_i + K}, \quad \text{where } K = 45.0$$
   $$\text{Clutch}_{\text{shrunk}, i} = \lambda_i \cdot \text{Raw Clutch}_i$$
   This pulls low-sample candidates smoothly toward the cohort prior, preventing lucky cameos from topping the board.

### 3.3 Match-to-Match Consistency Factor ($C_i$)
To filter out "one-match hero innings":
For each match $m$, player impact is tagged as positive if:
* **Batter:** Match High-Pressure $\text{SR} \ge 135$ or runs scored $\ge 20$.
* **Bowler:** Match High-Pressure $\text{Economy} \le 8.5$ or wickets taken $\ge 1$.

$$\text{Impact Rate}_i = \frac{\sum_m \mathbf{1}_{\{\text{Impact}_{i, m} > 0\}}}{N_{\text{matches}, i}}$$
$$\text{Consistency Factor } C_i = 0.80 + 0.40 \cdot \text{Impact Rate}_i \quad \in [0.80, 1.20]$$

$$\mathbf{\text{Final Calibrated Clutch Score (CCS)}_i = \text{Clutch}_{\text{shrunk}, i} \times C_i}$$

---

## 4. Final Rankings & Evidence

### 4.1 Top 10 Batters Under Pressure

| Rank | Player Name | HP Balls | HP SR | Normal SR | Boundary % | Residual Lift | Impact Rate | Final CCS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Naman Dhir** | 150 | **206.00** | 155.06 | 30.00% | +1.251 | 76.5% | **1.217** |
| **2** | **Jos Buttler** | 180 | **199.44** | 152.91 | 29.44% | +1.073 | 70.6% | **1.086** |
| **3** | **Suryakumar Yadav** | 475 | **177.68** | 164.01 | 26.95% | +0.785 | 78.9% | **0.992** |
| **4** | **Mark Chapman** | 213 | **175.59** | 134.46 | 23.00% | +0.884 | 75.0% | **0.958** |
| **5** | **Glenn Maxwell** | 163 | **179.14** | 156.59 | 27.61% | +0.897 | 81.3% | **0.951** |
| **6** | **Nicholas Pooran** | 461 | **180.26** | 150.17 | 25.16% | +0.765 | 71.4% | **0.934** |
| **7** | **Fakhar Zaman** | 101 | **185.15** | 124.28 | 26.73% | +1.083 | 69.2% | **0.930** |
| **8** | **Reeza Hendricks** | 123 | **179.67** | 141.50 | 25.20% | +0.969 | 71.4% | **0.911** |
| **9** | **Tim David** | 446 | **178.70** | 175.80 | 24.44% | +0.683 | 65.2% | **0.845** |
| **10** | **Matthew Wade** | 167 | **161.08** | 126.37 | 20.36% | +0.679 | 76.9% | **0.737** |

#### Analytical Insights on Batters:
* **Jos Buttler & Suryakumar Yadav**: Both demonstrate world-class baseline numbers and continue to elevate their scoring under pressure. Buttler strikes at near 200 in pressure chases, while Suryakumar's high volume (475 balls) across 38 pressure matches confirms elite repeatability.
* **Naman Dhir**: An exceptional surge performer in modern IPL high-pressure situations, striking at 206.0 with a 30% boundary rate and an impact rate over 76%.
* **Nicholas Pooran & Tim David**: Exemplars of power finishing under immense chase leverage, facing over 440+ pressure deliveries each while maintaining ~180 strike rates.

---

### 4.2 Top 10 Bowlers Under Pressure

| Rank | Player Name | HP Balls | HP Econ | Normal Econ | HP Dot % | HP Wkt % | Residual Lift | Impact Rate | Final CCS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Noor Ahmad** | 244 | **8.29** | 7.52 | 40.16% | 7.79% | +0.764 | 66.7% | **0.750** |
| **2** | **Roston Chase** | 109 | **7.49** | 7.55 | 35.78% | 10.09% | +0.753 | 100.0% | **0.667** |
| **3** | **Ben Dwarshuis** | 139 | **7.90** | 9.18 | 38.85% | 9.35% | +0.728 | 83.3% | **0.648** |
| **4** | **Taskin Ahmed** | 259 | **8.04** | 7.56 | 40.93% | 8.11% | +0.605 | 80.0% | **0.643** |
| **5** | **Sunil Narine** | 256 | **7.66** | 6.49 | 36.33% | 4.30% | +0.532 | 65.5% | **0.567** |
| **6** | **Tim Southee** | 161 | **8.35** | 7.89 | 39.75% | 7.45% | +0.550 | 68.4% | **0.523** |
| **7** | **Jasprit Bumrah** | 552 | **7.61** | 6.42 | 35.14% | 4.35% | +0.395 | 75.5% | **0.523** |
| **8** | **Eshan Malinga** | 233 | **8.47** | 9.26 | 33.91% | 7.30% | +0.487 | 90.0% | **0.510** |
| **9** | **Corbin Bosch** | 101 | **9.62** | 7.74 | 42.57% | 10.89% | +0.571 | 72.7% | **0.509** |
| **10** | **Rashid Khan** | 504 | **8.35** | 7.08 | 32.94% | 7.34% | +0.418 | 80.3% | **0.507** |

#### Analytical Insights on Bowlers:
* **Jasprit Bumrah**: With 552 pressure deliveries (the largest sample in the dataset), Bumrah concedes a remarkable 7.61 RPO under intense death and defending situations—over 2.5 RPO below the league pressure average. The residual lift model correctly validates him as an elite top-tier weapon.
* **Noor Ahmad & Rashid Khan**: Mystery spin proves exceptionally resilient under pressure. Noor Ahmad records a 40.2% dot-ball rate while taking wickets on nearly 8% of pressure deliveries.
* **Taskin Ahmed & Ben Dwarshuis**: High-impact pacers whose wicket-taking threat (8–10% wicket strike rate) and dot-ball choking ability consistently flip close contests.

---

## 5. Model Diagnostics & Reliability Analysis

### 5.1 Prevention of Underfitting and Overfitting
* **Underfitting Prevented**: The model captures non-linear situational volatility (sigmoid RRR, power-law resource depletion, phase transitions, and venue-adjusted par) rather than crude over-based buckets.
* **Overfitting Prevented**: The combination of a strict 100-ball sample floor, Empirical Bayes shrinkage ($K = 45$), and match-level impact rate weighting completely removes low-sample anomalies and one-match flukes.

### 5.2 Sensitivity & Limitations
1. **Fielding & Dropped Catches**: Deliveries resulting in dropped catches are charged as runs to bowlers; fielding quality data is currently unrecorded in standard ball-by-ball feeds.
2. **Impact Player Rule**: In IPL 2023–2026, the Impact Player rule deepened batting line-ups, elevating league-wide par scores. While our league standardization normalizes these macro shifts, tactical substitutions may slightly affect bowler workload distribution.
3. **Reproducibility**: The complete pipeline is fully deterministic, automated, and packaged in `src/pressure/` with all data schemas documented.
