# Rajasthan Royals: SupeRR Selector Round 1 (Pressure Ratings)

## What this is

A pressure-performance rating of batters and bowlers in IPL and T20I matches from 2022 to 2026, built for the Rajasthan Royals entry to SupeRR Selector Round 1 (Season 2). A "high-pressure" ball is one on which the match outcome is most sensitive to what happens next (leverage, top 20% within format and innings). A win-probability model is fitted to the ball-by-ball data, and each batter and bowler is credited against the average player in the same match situation. The output is two ranked top-10 lists (batters, bowlers) and the evidence behind them. Only the supplied data pack is used. Two A-team tournaments (Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025) are excluded, since the brief covers IPL and T20I only.

## Requirements

- Python 3.11
- Install the pinned packages from the project root:

```
pip install -r requirements.txt
```

## Data

The supplied data pack folder `6af3166e7b922_data_pack/` must sit in the project root, next to `pipeline/`. The scripts read `ball_by_ball.csv`, `matches.csv` and `competitions.csv` from it. This README does not include or redistribute the data pack.

## Run

Run these three commands in order, from the project root. Each one overwrites its outputs.

1. `python -P pipeline/build_state.py`
   Produces `data/processed/ball_state.parquet` (one row per delivery, with the match state before the ball). Prints `OK` on success.
2. `python -P pipeline/pressure_model.py`
   Produces `data/processed/ball_pressure.parquet` and the model and validation tables in `outputs/t3/`. Prints `OK` on success.
3. `python -P pipeline/ratings.py`
   Produces the ratings, top-10 lists, robustness tables and reliability file in `outputs/ratings/`. Prints `OK` on success.

The `data/processed/` folder is created by step 1. If a built-in check in step 1 fails, the script prints `FAILED` and exits with an error.

## Outputs

- `outputs/ratings/top10_batters.csv` and `outputs/ratings/top10_bowlers.csv`: the two lists in rank order, with robustness counts.
- `outputs/ratings/batters.csv` and `outputs/ratings/bowlers.csv`: every eligible player under the main variant.
- `outputs/ratings/robustness_batters.csv` and `outputs/ratings/robustness_bowlers.csv`: each player's rank under the 12 alternative variants.
- `outputs/ratings/reliability.json`: split-half reliability, eligibility counts and settings.
- `outputs/t3/summary.json`: model comparison, calibration error and agreement with the rule-based index.
- `outputs/t3/model_comparison_by_season.csv`, `outputs/t3/calibration.csv`, `outputs/t3/pressure_vs_ordinary.csv`: the model and pressure-definition validation tables.
- `data/processed/ball_pressure.parquet`: per-ball win probability, leverage, pressure percentile and credit (generated, git-ignored).

## Method in brief

Details and the reasons for each choice are in `docs/PROJECT.md`, decisions D1 to D8.

- **Pressure (D5):** leverage is the expected absolute change in win probability for a ball, over a league outcome mix for each format and innings. The top 20% within format and innings is "high pressure". Ranking across both innings was tried and rejected.
- **Chase targets (D1, D3):** back-solved from the official result, not from the rebuilt first innings. Chases where the official margin and the ball data disagree by more than 8 runs are flagged and kept out of model fitting.
- **Win probability (D6):** plain logistic regression on Duckworth-Lewis-style features, fitted leave-one-season-out so each season is scored by a model that did not see it.
- **Baseline (D8):** each credit is measured against the average player in the same format, innings, phase and wickets-down band.
- **Rating (D7):** stake-weighted impact on all balls, with empirical-Bayes (DerSimonian-Laird) shrinkage toward the eligible mean. Clutch uplift does not repeat across matches in the split-half test, so the shrinkage gives it little or no weight.
- **Ranking (D7):** the 20th percentile of 1,000 match-level bootstraps, not the point estimate.

## Repository map

- `docs/`: `PROJECT.md`, the brief, requirements, decisions, assumptions and status. Start here for detail.
- `pipeline/`: the three reproduction scripts, run in the order above.
- `spikes/`: exploratory checks made before the pipeline was built (for example the score-rebuild check referred to in D1). They are not needed to reproduce the results.
- `outputs/`: generated results (`outputs/t3/`, `outputs/ratings/`).
- `data/processed/`: generated intermediate files. Git-ignored (listed in `.gitignore`).
- `6af3166e7b922_data_pack/`: the supplied input data (see Data above).
- `requirements.txt`: pinned Python packages.
