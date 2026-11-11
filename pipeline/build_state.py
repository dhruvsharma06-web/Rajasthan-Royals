"""Step 2: build one row per delivery with the match state *before* the ball.

Output: data/processed/ball_state.parquet
Run from the project root:  python -P pipeline/build_state.py

Extras run amounts are not in the source data. Team runs per ball are estimated as
bat runs + 1 per extras event (+ extras_2_runs, + 5 for a penalty). Chase targets are NOT
taken from the rebuilt first-innings total; they are back-solved from the official result:

  won by N runs     -> needed_before(k) = N + 1 + team runs from ball k to the end
  won by W wickets  -> needed_before(k) = team runs from ball k to the end - overshoot
  tie               -> needed_before(k) = 1 + team runs from ball k to the end
  otherwise         -> rebuilt first-innings total + 1 - score before ball k   (target_src = "rebuilt")

so the error in runs-needed shrinks to ~0 toward the end of the chase, where pressure is highest.
"""
import os
import sys

import polars as pl

DATA = "6af3166e7b922_data_pack/"
OUT = "data/processed/ball_state.parquet"
ILLEGAL = ["wide", "Overstep", "other"]  # deliveries that do not count toward the over

b = pl.read_csv(DATA + "ball_by_ball.csv", infer_schema_length=200000)
m = pl.read_csv(DATA + "matches.csv", infer_schema_length=10000)
c = pl.read_csv(DATA + "competitions.csv")

# ---------------------------------------------------------------- match table
mt = (
    m.join(c.select(["comp_id", "comp_name", "match_format", "season"]), on="comp_id", how="left")
    .with_columns(pl.col("result").str.to_lowercase().alias("rl"))
    .with_columns(
        # result strings are free text ("won by 7 runs", "Won By 13Runs", "Won 3 Wkts", ...)
        pl.col("rl").str.contains(r"\btie|tied|super over|eliminator").alias("is_tie"),
        (
            pl.col("revised_target").is_not_null()
            | pl.col("first_inn_revised_overs").is_not_null()
            | pl.col("second_inn_revised_overs").is_not_null()
            | pl.col("rl").str.contains(r"d/l|dls")
        ).alias("is_dls"),
        # match_format "T20" is IPL plus two A-team tournaments (Emerging Teams Asia Cup 2024,
        # Asia Cup Rising Stars 2025). The brief covers IPL and T20I only, so those are labelled apart.
        pl.when(pl.col("match_format") == "T20I").then(pl.lit("T20I"))
        .when(pl.col("comp_name").str.starts_with("IPL")).then(pl.lit("IPL"))
        .otherwise(pl.lit("OTHER")).alias("fmt"),
    )
    .with_columns(
        pl.when(pl.col("is_tie")).then(None)
        .otherwise(pl.col("rl").str.extract(r"(\d+)\s*runs?\b", 1).cast(pl.Int32)).alias("by_runs"),
        pl.when(pl.col("is_tie")).then(None)
        .otherwise(pl.col("rl").str.extract(r"(\d+)\s*(?:wicket|wkt)", 1).cast(pl.Int32)).alias("by_wk"),
    )
)

# ---------------------------------------------------------------- per-ball basics
s = (
    b.filter(pl.col("innings") <= 2)  # innings 3/4 are super overs: excluded
    .sort(["match_id", "innings", "innings_ball_num"])
    .with_columns(
        pl.col("extras_type").is_in(ILLEGAL).not_().alias("is_legal"),
        (
            pl.col("runs")
            + (pl.col("extras_type") != "None").cast(pl.Int32)
            + pl.col("extras_2_runs")
            + (pl.col("extras_type") == "Penalty").cast(pl.Int32) * 4
        ).alias("team_runs"),
        # runs charged to the bowler: bat runs + 1 for a wide / no-ball (byes and leg-byes are not)
        (pl.col("runs") + pl.col("extras_type").is_in(["wide", "Overstep", "other"]).cast(pl.Int32)).alias(
            "bowler_runs"
        ),
        (pl.col("extras_type") != "wide").alias("faced"),  # counts as a ball faced by the striker
        (
            pl.col("is_wicket")
            & pl.col("dismissal_type").is_in(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"])
        ).alias("bowler_wicket"),
        (pl.col("is_wicket") & (pl.col("dismissed_player_id") == pl.col("striker_id"))).alias("striker_out"),
    )
)

grp = ["match_id", "innings"]
s = s.with_columns(
    (pl.col("team_runs").cum_sum().over(grp) - pl.col("team_runs")).alias("score_before"),
    (pl.col("is_wicket").cast(pl.Int32).cum_sum().over(grp) - pl.col("is_wicket").cast(pl.Int32)).alias(
        "wkts_before"
    ),
    (pl.col("is_legal").cast(pl.Int32).cum_sum().over(grp) - pl.col("is_legal").cast(pl.Int32)).alias(
        "legal_before"
    ),
    pl.col("team_runs").sum().over(grp).alias("inn_total_est"),
    pl.col("team_runs").cum_sum(reverse=True).over(grp).alias("runs_from_here"),
    pl.col("team_runs").last().over(grp).alias("last_ball_runs"),
)

# recent form of the innings: team runs and wickets in the previous 12 deliveries
s = s.with_columns(
    # window of 13 including the current ball, minus the current ball = the previous 12
    (pl.col("team_runs").rolling_sum(13, min_samples=1).over(grp) - pl.col("team_runs")).alias("runs_last12"),
    (
        pl.col("is_wicket").cast(pl.Int32).rolling_sum(13, min_samples=1).over(grp)
        - pl.col("is_wicket").cast(pl.Int32)
    ).alias("wkts_last12"),
)

inn1 = s.filter(pl.col("innings") == 1).group_by("match_id").agg(pl.col("inn_total_est").first().alias("inn1_est"))
s = s.join(inn1, on="match_id", how="left").join(
    mt.select(
        ["match_id", "fmt", "season", "match_date", "stage", "knock_out", "venue_id", "win_team_id",
         "by_runs", "by_wk", "is_tie", "is_dls", "revised_target",
         "first_inn_revised_overs", "second_inn_revised_overs"]
    ),
    on="match_id",
    how="left",
)

# ---------------------------------------------------------------- overs available and chase target
max_balls = (
    pl.when(pl.col("innings") == 1)
    .then(pl.col("first_inn_revised_overs"))
    .otherwise(pl.col("second_inn_revised_overs"))
    .cast(pl.Float64, strict=False)
)
s = s.with_columns(
    # the *_revised_overs columns hold BALLS (e.g. 114 = 19 overs), despite the name
    pl.when(max_balls.is_not_null() & (max_balls > 0)).then(max_balls.round().cast(pl.Int32)).otherwise(120)
    .alias("max_balls")
).with_columns((pl.col("max_balls") - pl.col("legal_before")).alias("balls_left"))

rebuilt_needed = pl.col("inn1_est") + 1 - pl.col("score_before")
overshoot = (pl.col("inn_total_est") - pl.col("inn1_est") - 1).clip(0, (pl.col("last_ball_runs") - 1).clip(0, 5))
chase = pl.col("innings") == 2
clean = chase & pl.col("is_dls").not_()
s = s.with_columns(
    pl.when(clean & pl.col("by_runs").is_not_null())
    .then(pl.lit("margin_runs"))
    .when(clean & pl.col("by_wk").is_not_null())
    .then(pl.lit("margin_wkts"))
    # a real tie has level scores; "Match Tied" after a rain-cut chase does not
    .when(clean & pl.col("is_tie") & ((pl.col("inn_total_est") - pl.col("inn1_est")).abs() <= 8))
    .then(pl.lit("tie"))
    .when(chase & pl.col("revised_target").is_not_null())
    .then(pl.lit("revised"))
    .when(chase)
    .then(pl.lit("rebuilt"))
    .otherwise(None)
    .alias("target_src")
).with_columns(
    pl.when(pl.col("target_src") == "margin_runs")
    .then(pl.col("by_runs") + 1 + pl.col("runs_from_here"))
    .when(pl.col("target_src") == "margin_wkts")
    .then(pl.col("runs_from_here") - overshoot)
    .when(pl.col("target_src") == "tie")
    .then(1 + pl.col("runs_from_here"))
    .when(pl.col("target_src") == "revised")
    .then(pl.col("revised_target").cast(pl.Int32, strict=False) - pl.col("score_before"))
    .when(pl.col("target_src") == "rebuilt")
    .then(rebuilt_needed)
    .otherwise(None)
    .alias("needed_before")
).with_columns(
    pl.when(pl.col("win_team_id").is_not_null() & (pl.col("win_team_id") > 0))
    .then((pl.col("batting_team_id") == pl.col("win_team_id")).cast(pl.Int8))
    .otherwise(None)
    .alias("bat_team_won"),
    # official margin and ball data disagree by more than 8 runs: balls are probably missing
    (pl.col("target_src").is_in(["margin_runs", "margin_wkts"])
     & ((pl.col("needed_before") + pl.col("score_before") - pl.col("inn1_est") - 1).abs() > 8))
    .fill_null(False).alias("target_suspect"),
    pl.col("striker_out").fill_null(False),
    pl.when(pl.col("overid") <= 6).then(pl.lit("powerplay")).when(pl.col("overid") <= 15).then(pl.lit("middle"))
    .otherwise(pl.lit("death")).alias("phase"),
)

keep = [
    "match_id", "fmt", "season", "match_date", "stage", "knock_out", "venue_id", "innings", "overid",
    "innings_ball_num", "phase", "batting_team_id", "bowling_team_id", "striker_id", "striker_name",
    "non_striker_id", "bowler_id", "bowler_name", "batting_order", "batting_type", "bowling_type", "pace_or_spin",
    "runs", "is_four", "is_six", "extras_type", "team_runs", "bowler_runs", "is_legal", "faced",
    "is_wicket", "bowler_wicket", "striker_out", "dismissal_type", "dismissed_player_id",
    "score_before", "wkts_before", "legal_before", "balls_left", "max_balls", "runs_last12", "wkts_last12",
    "inn1_est", "needed_before", "target_src", "target_suspect", "is_dls", "bat_team_won",
]
out = s.select(keep)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.write_parquet(OUT)

# ---------------------------------------------------------------- checks (fail loudly)
print(f"wrote {OUT}: {out.height} rows, {out['match_id'].n_unique()} matches")
fails = []

legal = out.group_by(grp).agg(pl.col("is_legal").sum().alias("n"), pl.col("max_balls").first())
over = legal.filter(pl.col("n") > pl.col("max_balls")).height
print(f"[check] innings with more legal balls than allowed: {over} of {legal.height}")
print("        legal balls in completed-looking innings:",
      dict(legal["n"].value_counts().sort("count", descending=True).head(4).iter_rows()))

wk = out.filter(pl.col("wkts_before") > 9).height
print(f"[check] balls bowled with 10 wickets already down: {wk}")
if wk:
    fails.append("wickets > 9")

print("[info] chases flagged target_suspect:", out.filter(pl.col("target_suspect"))["match_id"].n_unique())
ch = out.filter((pl.col("innings") == 2) & pl.col("target_src").is_in(["margin_runs", "margin_wkts", "tie"])
                & pl.col("target_suspect").not_())
bad = ch.filter(pl.col("needed_before") < 1).height
print(f"[check] chase balls bowled when runs needed < 1 (back-solved targets): {bad} of {ch.height}")
if bad:
    fails.append("needed < 1 mid-chase")

# independent cross-check: back-solved target vs rebuilt first-innings total + 1, at ball 1 of the chase
first = ch.filter(pl.col("innings_ball_num") == pl.col("innings_ball_num").min().over("match_id")).with_columns(
    (pl.col("needed_before") - (pl.col("inn1_est") + 1)).alias("d")
)
d = first["d"].abs()
print(f"[check] back-solved target vs rebuilt inn1+1 over {first.height} chases: "
      f"MAE={d.mean():.2f} within2={(d <= 2).mean():.0%} within5={(d <= 5).mean():.0%} max={d.max()}")
print("        worst:", first.sort(pl.col("d").abs(), descending=True).select(["match_id", "target_src", "d"]).head(6).rows())

print("[info] chase target source (matches):",
      dict(out.filter(pl.col("innings") == 2).group_by("target_src").agg(pl.col("match_id").n_unique()).iter_rows()))
print("[info] balls by format:", dict(out.group_by("fmt").len().iter_rows()))
print("[info] winner known for", out.filter(pl.col("bat_team_won").is_not_null())["match_id"].n_unique(), "matches")
nulls = {k: v for k, v in zip(out.columns, out.null_count().row(0)) if v and k not in
         ("dismissal_type", "dismissed_player_id", "needed_before", "target_src", "bat_team_won")}
print("[info] unexpected nulls:", nulls)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("OK")
