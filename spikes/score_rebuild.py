"""SPIKE S03: can innings totals (and so chase targets) be rebuilt from ball_by_ball.csv?

Extras run amounts are not recorded, so we estimate runs-per-extra-event by least squares
against official run margins ("X won by N runs": total1 - total2 = N), then measure the
residual error on those matches and sanity-check on successful chases
(official: total2 - total1 must be between 1 and 6).

Run from the project root:  python -P spikes/score_rebuild.py
"""
import sys

import numpy as np
import polars as pl

DATA = sys.argv[1] if len(sys.argv) > 1 else "6af3166e7b922_data_pack/"
TYPES = ["wide", "Overstep", "legbyes", "byes", "other", "Penalty"]

b = pl.read_csv(DATA + "ball_by_ball.csv", infer_schema_length=200000)
m = pl.read_csv(DATA + "matches.csv", infer_schema_length=10000)

print("== 'other' rows sample ==")
print(
    b.filter(pl.col("extras_type") == "other")
    .group_by(["runs", "extras_2_runs", "is_four"])
    .len()
    .sort("len", descending=True)
    .head(8)
)

inn = (
    b.filter(pl.col("innings") <= 2)
    .group_by(["match_id", "innings"])
    .agg(
        pl.col("runs").sum().alias("bat"),
        pl.col("extras_2_runs").sum().alias("x2"),
        pl.col("is_wicket").sum().alias("wk"),
        pl.col("batting_team_id").first().alias("team"),
        # cast: polars sums booleans to unsigned ints, which wrap on subtraction
        *[(pl.col("extras_type") == t).sum().cast(pl.Int64).alias(t) for t in TYPES],
    )
)
w = inn.pivot(on="innings", index="match_id", values=["bat", "x2", "wk", "team", *TYPES])
r = (
    w.join(m.select(["match_id", "result", "revised_target", "win_team_id"]), on="match_id")
    .filter(pl.col("revised_target").is_null() & pl.col("bat_2").is_not_null())
    .with_columns(pl.col("result").str.to_lowercase().alias("rl"))
    .with_columns(
        pl.col("rl").str.extract(r"by (\d+) run", 1).cast(pl.Int32).alias("by_runs"),
        pl.col("rl").str.extract(r"by (\d+) (?:wicket|wkt)", 1).cast(pl.Int32).alias("by_wk"),
    )
)

runs_m = r.filter(pl.col("by_runs").is_not_null())
chase_m = r.filter(pl.col("by_wk").is_not_null())
# Odd/even match_id split so the coefficients are not scored on the matches they were fit on.
fit = runs_m.filter(pl.col("match_id") % 2 == 0)
test = runs_m.filter(pl.col("match_id") % 2 == 1)


def design(df):
    X = np.column_stack([(df[f"{t}_1"] - df[f"{t}_2"]).to_numpy() for t in TYPES]).astype(float)
    y = (df["by_runs"] - (df["bat_1"] - df["bat_2"])).to_numpy().astype(float)
    return X, y


X, y = design(fit)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print("\n== fitted runs per extras event (fit half, n=%d) ==" % fit.height)
for t, c in zip(TYPES, coef):
    print(f"  {t:9s} {c:5.2f}")


def report(name, err):
    a = np.abs(err)
    print(
        f"{name:34s} n={len(err):3d} mean={err.mean():+.2f} MAE={a.mean():.2f} "
        f"exact={np.mean(a < 0.5):.0%} within1={np.mean(a <= 1.5):.0%} "
        f"within3={np.mean(a <= 3.5):.0%} max={a.max():.0f}"
    )


Xt, yt = design(test)
print("\n== error in (total1 - total2) vs official run margin, held-out half ==")
report("bat runs only", -yt)
naive = np.array([1, 1, 1, 1, 1, 5.0])
report("naive: 1 run per extra", Xt @ naive - yt)
report("fitted coefficients", Xt @ coef - yt)

print("\n== successful chases: rebuilt (total2 - total1), official range is 1..6 ==")
lead = (chase_m["bat_2"] - chase_m["bat_1"]).to_numpy() + np.column_stack(
    [(chase_m[f"{t}_2"] - chase_m[f"{t}_1"]).to_numpy() for t in TYPES]
) @ coef
print(f"  n={len(lead)}  in [0.5, 6.5]: {np.mean((lead >= 0.5) & (lead <= 6.5)):.0%}  "
      f"<= 0 (impossible): {np.mean(lead < 0.5):.0%}  > 6.5: {np.mean(lead > 6.5):.0%}")
print("  percentiles 5/25/50/75/95:", np.percentile(lead, [5, 25, 50, 75, 95]).round(1))

big = runs_m.with_columns(
    pl.Series("err", design(runs_m)[0] @ coef - design(runs_m)[1]).round(1)
).filter(pl.col("err").abs() > 6)
print(f"\n== run-margin matches off by more than 6 runs: {big.height} of {runs_m.height} ==")
print(big.select(["match_id", "bat_1", "bat_2", "result", "err"]).head(12))

print("\n== consistency of innings order with the winner ==")
print("  'won by runs' but winner did not bat first:",
      runs_m.filter(pl.col("team_1") != pl.col("win_team_id")).height, "of", runs_m.height)
print("  'won by wickets' but winner did not bat second:",
      chase_m.filter(pl.col("team_2") != pl.col("win_team_id")).height, "of", chase_m.height)
print(runs_m.filter(pl.col("team_1") != pl.col("win_team_id"))
      .select(["match_id", "bat_1", "bat_2", "team_1", "team_2", "win_team_id", "result"]).head(10))
