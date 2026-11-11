"""T3: win-probability model, per-ball leverage (pressure), and validation of the pressure definition.

Input : data/processed/ball_state.parquet   (from pipeline/build_state.py)
Output: data/processed/ball_pressure.parquet  one row per delivery in the rating set, with
          wp_before, wp_after, wpa (batting side), leverage, pressure_pct, high_pressure, rule_index
        outputs/t3/*.csv / *.json             model comparison, calibration, pressure validation
Run from the project root:  python -P pipeline/pressure_model.py

Design (see docs/PROJECT.md, rules F1-F8 and G4-G6):
- Value function V(state) = P(batting side wins). One model per innings; first-innings states are
  converted into a chase at the innings break so both innings share one currency.
- Leave-one-season-out: every season is scored by a model that never saw it (G4).
- Team strength (pre-match Elo difference) and scoring era (format-season par) are inputs (G5).
- Leverage of a ball = expected absolute change in V over the league outcome mix for that
  format/innings/phase. Pressure percentile is ranked within format x innings (G6).
"""
import json
import os

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

DATA = "6af3166e7b922_data_pack/"
IN = "data/processed/ball_state.parquet"
OUT = "data/processed/ball_pressure.parquet"
REP = "outputs/t3/"
HIGH = 0.80  # top 20% of leverage within format x innings = high pressure
SEED = 7
os.makedirs(REP, exist_ok=True)

s = pl.read_parquet(IN).with_columns(pl.col("match_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
m = pl.read_csv(DATA + "matches.csv", infer_schema_length=10000)

# ---------------------------------------------------------------- Elo team strength (pre-match, no leakage)
K = 24.0
elo: dict[int, float] = {}
pre = []
for mid, date, h, a, w, res in (
    m.sort(["match_date", "match_id"])
    .select(["match_id", "match_date", "home_team_id", "away_team_id", "win_team_id", "result"])
    .iter_rows()
):
    rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
    pre.append((mid, h, rh, a, ra))
    rl = (res or "").lower()
    if w in (h, a) and not any(k in rl for k in ("tie", "tied", "super over", "eliminator")):
        sh = 1.0 if w == h else 0.0
    elif any(k in rl for k in ("tie", "tied", "super over", "eliminator")):
        sh = 0.5
    else:
        continue  # no result / abandoned: no update
    eh = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
    elo[h], elo[a] = rh + K * (sh - eh), ra - K * (sh - eh)
pre_df = pl.DataFrame(pre, schema=["match_id", "t1", "r1", "t2", "r2"], orient="row")
team_elo = pl.concat(
    [pre_df.select("match_id", pl.col("t1").alias("team"), pl.col("r1").alias("elo")),
     pre_df.select("match_id", pl.col("t2").alias("team"), pl.col("r2").alias("elo"))]
)
s = (
    s.join(team_elo.rename({"team": "batting_team_id", "elo": "elo_bat"}), on=["match_id", "batting_team_id"], how="left")
    .join(team_elo.rename({"team": "bowling_team_id", "elo": "elo_bowl"}), on=["match_id", "bowling_team_id"], how="left")
    .with_columns((pl.col("elo_bat") - pl.col("elo_bowl")).fill_null(0.0).alias("elo_diff"))
)
pl.DataFrame({"team_id": list(elo), "final_elo": list(elo.values())}).write_csv(REP + "team_elo_final.csv")

# ---------------------------------------------------------------- scope
clean_match = (pl.col("target_src").is_in(["margin_runs", "margin_wkts", "tie"]) & ~pl.col("target_suspect")).any().over("match_id")
s = s.filter((pl.col("fmt") != "OTHER") & ~pl.col("is_dls") & clean_match)
s = s.with_columns(pl.col("target_src").drop_nulls().first().over("match_id").alias("match_src"))
s = s.with_columns(
    pl.when(pl.col("match_src") == "tie").then(0.5).otherwise(pl.col("bat_team_won").cast(pl.Float64)).alias("y"),
    (10 - pl.col("wkts_before")).alias("wih"),
    (pl.col("fmt") == "IPL").cast(pl.Int8).alias("ipl"),
)
# par = mean first-innings total of the OTHER matches in the same format and season (leave-one-match-out),
# so a match never contributes to its own par (review finding). It is a scoring-environment descriptor and
# still uses other matches of the season, including later ones; documented as a known approximation.
inn1_tot = s.filter(pl.col("innings") == 1).group_by(["match_id", "fmt", "year"]).agg(pl.col("inn1_est").first())
par = inn1_tot.with_columns(
    ((pl.col("inn1_est").sum().over(["fmt", "year"]) - pl.col("inn1_est"))
     / (pl.len().over(["fmt", "year"]) - 1)).alias("par")
).select(["match_id", "par"])
s = s.join(par, on="match_id", how="left")
s = s.with_columns(pl.col("par").fill_null(pl.col("par").mean().over(["fmt", "year"])))
YEARS = sorted(s["year"].unique().to_list())
print(f"scope: {s['match_id'].n_unique()} matches, {s.height} balls, years {YEARS}")
print("par (leave-one-match-out) mean by format/year:", sorted(s.group_by(["fmt", "year"]).agg(pl.col("par").mean().round(1)).rows()))

# ---------------------------------------------------------------- features
# R(balls_left, wickets_in_hand, par, format) = expected runs still to come in an innings (DLS-style
# resource), learned from first innings. It makes the value of a wicket depend on how much batting is left.
# Smooth Duckworth-Lewis form (a tree model gave a step-shaped table and spurious leverage):
#   R(u, w) = scale * Z0 * F(w) * (1 - exp(-b * u / F(w))),  u = balls left, w = wickets in hand, F(10) = 1
# scale = par / 170 carries the format-season scoring level.
PAR_REF = 170.0


class DLResource:
    def __init__(self, z0, b, F):
        self.z0, self.b, self.F = z0, b, F

    def predict(self, X):
        u, w, par_ = X[:, 0], np.clip(X[:, 1], 0, 10).astype(int), X[:, 2]
        Fw = self.F[w]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (par_ / PAR_REF) * self.z0 * Fw * (1 - np.exp(-self.b * u / np.where(Fw > 0, Fw, 1)))
        return np.where(Fw > 0, r, 0.0)


def fit_resource(df):
    from scipy.optimize import least_squares

    d = df.filter(pl.col("innings") == 1)
    u = d["balls_left"].to_numpy().astype(float)
    w = d["wih"].to_numpy().astype(int)
    sc = d["par"].to_numpy() / PAR_REF
    y = (d["inn1_est"] - d["score_before"]).to_numpy().astype(float)

    def unpack(th):
        F = np.zeros(11)
        F[1:10] = np.exp(th[2:11])
        F[10] = 1.0
        return np.exp(th[0]), np.exp(th[1]), F

    def resid(th):
        z0, b, F = unpack(th)
        Fw = F[w]
        return sc * z0 * Fw * (1 - np.exp(-b * u / Fw)) - y

    th0 = np.concatenate([[np.log(250.0), np.log(0.01)], np.log(np.linspace(0.08, 0.92, 9))])
    sol = least_squares(resid, th0, loss="soft_l1", f_scale=20.0)
    z0, b, F = unpack(sol.x)
    if not np.all(np.diff(F[1:]) > 0):
        print("  [warn] resource F(w) not increasing in wickets in hand:", F.round(3))
    return DLResource(z0, b, F)


def resource(R, balls, wih, par_, ipl):
    balls, wih = np.asarray(balls, float), np.asarray(wih, float)
    out = R.predict(np.column_stack([balls, wih, np.asarray(par_, float), np.asarray(ipl, float)]))
    return np.where((balls <= 0) | (wih <= 0), 0.0, np.maximum(out, 0.0))


def feats1(R, score, balls, wih, par_, elo_diff, ipl):
    score, balls = np.asarray(score, float), np.asarray(balls, float)
    diff = score + resource(R, balls, wih, par_, ipl) - np.asarray(par_, float)  # projected total vs par
    bf = balls / 120.0
    return np.column_stack([diff, diff * (1 - bf), bf, elo_diff, ipl]).astype(float)


def feats2(R, need, balls, wih, par_, elo_diff, ipl):
    need, balls, wih = (np.asarray(x, float) for x in (need, balls, wih))
    deficit = need - resource(R, balls, wih, par_, ipl)  # runs needed beyond what is expected to come
    z = deficit / np.sqrt(balls + 6.0)  # spread of remaining runs grows with balls left
    rrr = need * 6.0 / np.maximum(balls, 1.0)
    return np.column_stack([z, deficit, rrr, balls, wih, elo_diff, ipl]).astype(float)


MONO1 = [1, 1, 0, 1, 0]
MONO2 = [-1, -1, -1, 0, 1, 1, 0]


def make(kind, mono):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, monotonic_cst=mono, random_state=SEED,
        )
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    # linear extrapolation: constant extrapolation flattened the curve in hopeless chases
    # ("need 97 off 1 ball" scored 19%)
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3, extrapolation="linear"),
                         LogisticRegression(C=1.0, max_iter=3000))


class Value:
    """V(state) = P(batting side wins), with terminal states handled exactly."""

    def __init__(self, R, m1, m2):
        self.R, self.m1, self.m2 = R, m1, m2

    def v2(self, need, balls, wih, par_, elo_diff, ipl):
        need, balls, wih = (np.asarray(x, float) for x in (need, balls, wih))
        par_, elo_diff, ipl = np.asarray(par_, float), np.asarray(elo_diff, float), np.asarray(ipl)
        out = np.empty(len(need))
        won = need <= 0
        over = ~won & ((wih <= 0) | (balls <= 0))
        live = ~won & ~over
        out[won] = 1.0
        out[over] = np.where(need[over] == 1, 0.5, 0.0)
        if live.any():
            X = feats2(self.R, need[live], balls[live], wih[live], par_[live], elo_diff[live], ipl[live])
            out[live] = self.m2.predict_proba(X)[:, 1]
            # more than 6 per remaining ball is only possible through extras: near-certain loss
            imp = live & (need > 6 * balls)
            out[imp] = np.minimum(out[imp], 0.002)
        return out

    def v1(self, score, balls, wih, par_, elo_diff, ipl):
        score, balls, wih = (np.asarray(x, float) for x in (score, balls, wih))
        elo_diff, ipl, par_ = np.asarray(elo_diff, float), np.asarray(ipl), np.asarray(par_, float)
        # The innings-1 model is also used at the end of the innings (balls = 0 or all out): runs to come
        # are 0 there, so the projected total is the final total. Switching to the chase model at the break
        # made a last-ball six lose up to 0.30 win probability.
        balls = np.maximum(balls, 0.0)
        return self.m1.predict_proba(feats1(self.R, score, balls, wih, par_, elo_diff, ipl))[:, 1]

    def value(self, inn, runs, balls, wih, par_, elo_diff, ipl):
        """runs = score so far (innings 1) or runs still needed (innings 2)."""
        inn = np.asarray(inn)
        out = np.empty(len(inn))
        a, b = inn == 1, inn == 2
        out[a] = self.v1(runs[a], balls[a], wih[a], par_[a], elo_diff[a], ipl[a])
        out[b] = self.v2(runs[b], balls[b], wih[b], par_[b], elo_diff[b], ipl[b])
        return out


def train_xy(R, df, inn):
    d = df.filter((pl.col("innings") == inn) & pl.col("y").is_not_null() & (pl.col("y") != 0.5))
    if inn == 1:
        X = feats1(R, d["score_before"], d["balls_left"], d["wih"], d["par"], d["elo_diff"], d["ipl"])
    else:
        X = feats2(R, d["needed_before"], d["balls_left"], d["wih"], d["par"], d["elo_diff"], d["ipl"])
    return X, d["y"].to_numpy().astype(int)


# ---------------------------------------------------------------- leave-one-season-out fit + model choice
models = {}
RES = {y: fit_resource(s.filter(pl.col("year") != y)) for y in YEARS}
for y in YEARS:
    r = RES[y]
    print(f"  resource fold {y}: Z0={r.z0:.0f} b={r.b:.4f} F(w)={np.round(r.F[1:], 2).tolist()}")
_r = RES[YEARS[-1]]
_u = np.array([120, 60, 30, 12])
print("  expected runs to come (par 170), rows = balls left 120/60/30/12, cols = 10/7/5/3 wickets in hand:")
for uu in _u:
    print("   ", [round(float(_r.predict(np.array([[uu, ww, 170.0, 0.0]]))[0]), 1) for ww in (10, 7, 5, 3)])
comp = []
for kind in ("hgb", "spline_logit", "logit"):
    for y in YEARS:
        tr, te = s.filter(pl.col("year") != y), s.filter(pl.col("year") == y)
        for inn, mono in ((1, MONO1), (2, MONO2)):
            Xtr, ytr = train_xy(RES[y], tr, inn)
            mdl = make(kind, mono).fit(Xtr, ytr)
            models[(kind, y, inn)] = mdl
            Xte, yte = train_xy(RES[y], te, inn)
            p = np.clip(mdl.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
            base = np.full(len(yte), ytr.mean())
            comp.append({"model": kind, "year": y, "innings": inn, "n_balls": len(yte),
                         "log_loss": log_loss(yte, p), "brier": brier_score_loss(yte, p),
                         "log_loss_base": log_loss(yte, base)})
comp_df = pl.DataFrame(comp)
comp_df.write_csv(REP + "model_comparison_by_season.csv")
summ = comp_df.group_by(["model", "innings"]).agg(
    (pl.col("log_loss") * pl.col("n_balls")).sum() / pl.col("n_balls").sum(),
    (pl.col("brier") * pl.col("n_balls")).sum() / pl.col("n_balls").sum(),
    (pl.col("log_loss_base") * pl.col("n_balls")).sum() / pl.col("n_balls").sum(),
).sort(["innings", "log_loss"])
print("\nheld-out (leave-one-season-out) model comparison:")
print(summ)
CHOICE = {inn: summ.filter(pl.col("innings") == inn).sort("log_loss")["model"][0] for inn in (1, 2)}
print("chosen model per innings:", CHOICE)

# ---------------------------------------------------------------- outcome mix for leverage
OUTCOMES = [(0, 0), (1, 0), (2, 0), (4, 0), (6, 0), (0, 1)]  # (runs, wicket)
legal = s.filter(pl.col("is_legal"))
mix_rows = []
# One mix per format x innings (not per phase): pressure must reflect the match situation,
# not the phase label. A per-phase mix produced steps at overs 7 and 16.
for (fmt, inn), g in legal.group_by(["fmt", "innings"]):
    n = g.height
    w = g["is_wicket"].sum()
    r = g.filter(~pl.col("is_wicket"))["team_runs"]
    cnt = {k: int((r == k).sum()) for k in (0, 1, 2, 4, 6)}
    cnt[3] = int((r == 3).sum())
    cnt[5] = int((r >= 5).sum())
    probs = [(cnt[0]) / n, (cnt[1]) / n, (cnt[2] + cnt[3]) / n, (cnt[4] + cnt[5]) / n, cnt[6] / n, w / n]
    mix_rows.append({"fmt": fmt, "innings": inn, **{f"p{i}": p for i, p in enumerate(probs)}})
mix = pl.DataFrame(mix_rows)
mix.write_csv(REP + "outcome_mix.csv")
s = s.join(mix, on=["fmt", "innings"], how="left")

# ---------------------------------------------------------------- score every ball with its held-out model
parts = []
for y in YEARS:
    d = s.filter(pl.col("year") == y)
    V = Value(RES[y], models[(CHOICE[1], y, 1)], models[(CHOICE[2], y, 2)])
    inn = d["innings"].to_numpy()
    runs = np.where(inn == 1, d["score_before"].to_numpy(), d["needed_before"].fill_null(0).to_numpy()).astype(float)
    sign = np.where(inn == 1, 1.0, -1.0)  # runs scored add to score (inn 1) / subtract from need (inn 2)
    balls, wih = d["balls_left"].to_numpy().astype(float), d["wih"].to_numpy().astype(float)
    par_, ed, ipl = d["par"].to_numpy(), d["elo_diff"].to_numpy(), d["ipl"].to_numpy()
    v0 = V.value(inn, runs, balls, wih, par_, ed, ipl)
    leg = d["is_legal"].to_numpy().astype(float)
    v1 = V.value(inn, runs + sign * d["team_runs"].to_numpy(), balls - leg,
                 wih - d["is_wicket"].to_numpy().astype(float), par_, ed, ipl)
    lev = np.zeros(len(d))
    for i, (r, wk) in enumerate(OUTCOMES):
        vo = V.value(inn, runs + sign * r, balls - 1, wih - wk, par_, ed, ipl)
        lev += d[f"p{i}"].to_numpy() * np.abs(vo - v0)
    # credit components (F5, F6): state after the ball with only some of its effects applied, no wicket
    v_bat = V.value(inn, runs + sign * d["runs"].to_numpy(), balls - leg, wih, par_, ed, ipl)
    v_bowl = V.value(inn, runs + sign * d["bowler_runs"].to_numpy(), balls - leg, wih, par_, ed, ipl)
    v_nowk = V.value(inn, runs + sign * d["team_runs"].to_numpy(), balls - leg, wih, par_, ed, ipl)
    v_six = V.value(inn, runs + sign * 6, balls - 1, wih, par_, ed, ipl)
    v_wkt = V.value(inn, runs, balls - 1, wih - 1, par_, ed, ipl)
    parts.append(d.with_columns(pl.Series("wp_before", v0), pl.Series("wp_after", v1),
                                pl.Series("v_bat", v_bat), pl.Series("v_bowl", v_bowl), pl.Series("v_nowk", v_nowk),
                                pl.Series("leverage_raw", lev), pl.Series("swing_six_vs_wkt", v_six - v_wkt)))
p = pl.concat(parts).with_columns((pl.col("wp_after") - pl.col("wp_before")).alias("wpa"))
# Credit split (batting-side win probability units; bowler credit is from the bowling side):
#   striker  : effect of the bat runs (0 on a wide: ball not faced, nothing changes)
#   dismissed: effect of the wicket itself (retired out is a team decision: charged to nobody)
#   bowler   : -(effect of runs charged to him + ball consumed) + wicket effect if he took the wicket
wk_part = pl.col("wp_after") - pl.col("v_nowk")
p = p.with_columns(
    pl.when(pl.col("faced")).then(pl.col("v_bat") - pl.col("wp_before")).otherwise(0.0).alias("cr_striker_runs"),
    pl.when(pl.col("is_wicket") & ~pl.col("dismissal_type").fill_null("").str.contains("Retire"))
    .then(wk_part).otherwise(0.0).alias("cr_wicket"),
    (-(pl.col("v_bowl") - pl.col("wp_before"))
     - pl.when(pl.col("bowler_wicket")).then(wk_part).otherwise(0.0)).alias("cr_bowler"),
)
p = p.with_columns((pl.col("leverage_raw") / pl.col("leverage_raw").mean()).alias("leverage"))
p = p.with_columns(
    # main definition (D5): stakes ranked within format x innings. Ranked across both innings, 98.7% of
    # high-pressure balls fell in chases and first-innings players could not be assessed.
    (pl.col("leverage").rank("average").over(["fmt", "innings"]) / pl.len().over(["fmt", "innings"])).alias(
        "pressure_pct"),
    # sensitivity: absolute stakes ranked within format across both innings
    (pl.col("leverage").rank("average").over("fmt") / pl.len().over("fmt")).alias("pressure_pct_pooled"),
).with_columns((pl.col("pressure_pct") >= HIGH).alias("high_pressure"))

# ---------------------------------------------------------------- rule-based pressure index (transparent cross-check)
crr = pl.col("score_before") * 6 / pl.max_horizontal(pl.lit(1), pl.col("legal_before"))
rrr = pl.col("needed_before") * 6 / pl.max_horizontal(pl.lit(1), pl.col("balls_left"))
p = p.with_columns(
    pl.when(pl.col("innings") == 2)
    .then(
        (rrr - crr).clip(-6, 12) / 6            # chase falling behind
        + pl.col("wkts_before") / 5             # wickets lost
        + (1 - pl.col("balls_left") / 120) * 2  # how late
        + (pl.col("wkts_last12") / 2)           # recent collapse
        - (pl.col("needed_before") > pl.col("balls_left") * 2.5).cast(pl.Float64) * 3  # already lost
    )
    .otherwise(
        pl.col("wkts_before") / 5 + (1 - pl.col("balls_left") / 120) * 2 + pl.col("wkts_last12") / 2
        + ((pl.col("par") * pl.col("legal_before") / 120) - pl.col("score_before")).clip(-30, 30) / 30
    )
    .alias("rule_index")
)

# ---------------------------------------------------------------- sanity checks on the credit split
bad_six = p.filter(pl.col("is_six") & (pl.col("cr_striker_runs") < -0.002))
bad_wkt = p.filter(pl.col("is_wicket") & (pl.col("cr_wicket") > 0.002)
                   & ~pl.col("dismissal_type").fill_null("").str.contains("Retire"))
print(f"\n[check] sixes that cost the batter >0.2% win probability: {bad_six.height} of {p['is_six'].sum()}")
print(f"[check] wickets that gained the batting side >0.2% win probability: {bad_wkt.height} of {p['is_wicket'].sum()}")
for name, df in (("six", bad_six), ("wicket", bad_wkt)):
    if df.height:
        print(f"        worst {name}:", df.select(["innings", "overid", "score_before", "needed_before", "balls_left",
                                                    "wkts_before", "wp_before", "wp_after"]).head(3).rows())

keep = [c for c in s.columns if not (c.startswith("p") and c[1:].isdigit())] + [
    "wp_before", "wp_after", "wpa", "cr_striker_runs", "cr_wicket", "cr_bowler", "elo_diff", "leverage_raw", "leverage", "swing_six_vs_wkt", "pressure_pct",
    "pressure_pct_pooled", "high_pressure", "rule_index"]
p = p.select([c for c in dict.fromkeys(keep) if c in p.columns])
p.write_parquet(OUT)
print(f"\nwrote {OUT}: {p.height} rows")

# ---------------------------------------------------------------- calibration (held-out)
cal = (
    p.filter(pl.col("y").is_not_null() & (pl.col("y") != 0.5))
    .with_columns((pl.col("wp_before") * 10).floor().clip(0, 9).alias("bin"))
    .group_by(["innings", "bin"])
    .agg(pl.len().alias("n"), pl.col("wp_before").mean().alias("predicted"), pl.col("y").mean().alias("observed"))
    .sort(["innings", "bin"])
)
cal.write_csv(REP + "calibration.csv")
ece = cal.group_by("innings").agg(((pl.col("predicted") - pl.col("observed")).abs() * pl.col("n")).sum() / pl.col("n").sum())
print("\ncalibration (held-out), expected calibration error by innings:", ece.sort("innings").rows())

# ---------------------------------------------------------------- validation of the pressure definition
lb = p.filter(pl.col("is_legal"))
val = (
    lb.group_by(["fmt", "innings", "high_pressure"])
    .agg(
        pl.len().alias("balls"),
        pl.col("wpa").abs().mean().alias("mean_abs_wp_swing"),
        (pl.col("runs") == 0).mean().alias("dot_rate"),
        (pl.col("is_four") | pl.col("is_six")).mean().alias("boundary_rate"),
        pl.col("is_wicket").mean().alias("wicket_rate"),
        pl.col("team_runs").mean().alias("runs_per_ball"),
        (pl.col("phase") == "death").mean().alias("share_death"),
    )
    .sort(["fmt", "innings", "high_pressure"])
)
val.write_csv(REP + "pressure_vs_ordinary.csv")
print("\n(a)+(b) high-pressure vs ordinary legal balls:")
print(val)

# match-level: is the result decided in high-pressure balls?
dec = p.group_by("high_pressure").agg(pl.col("wpa").abs().sum().alias("total_abs_wp_movement"), pl.len().alias("balls"))
tot = dec["total_abs_wp_movement"].sum()
print("\nshare of all win-probability movement that happens on high-pressure balls:",
      round(dec.filter(pl.col("high_pressure"))["total_abs_wp_movement"][0] / tot, 3),
      "while they are", round(dec.filter(pl.col("high_pressure"))["balls"][0] / p.height, 3), "of balls")

# (c) agreement with the rule-based index
agree = []
for (fmt, inn), g in p.group_by(["fmt", "innings"]):
    rho = spearmanr(g["leverage"], g["rule_index"]).statistic
    top_rule = g["rule_index"] >= g["rule_index"].quantile(HIGH)
    top_lev = g["leverage"] >= g["leverage"].quantile(HIGH)
    ov = (top_lev & top_rule).sum() / max(1, top_lev.sum())
    agree.append({"fmt": fmt, "innings": inn, "spearman": round(float(rho), 3), "top20_overlap": round(float(ov), 3)})
agree_df = pl.DataFrame(agree).sort(["fmt", "innings"])
# Why the rule index disagrees: how often each definition flags balls whose match is already decided.
dead = (pl.col("wp_before") < 0.05) | (pl.col("wp_before") > 0.95)
rule_top = pl.col("rule_index") >= pl.col("rule_index").quantile(HIGH).over(["fmt", "innings"])
decided = p.select(
    dead.filter(pl.col("high_pressure")).mean().alias("model_high_share_decided"),
    dead.filter(rule_top).mean().alias("rule_high_share_decided"),
    dead.mean().alias("all_balls_share_decided"),
)
decided.write_csv(REP + "decided_share.csv")
print("\nshare of flagged balls where the match was already decided (win prob <5% or >95%):")
print(decided)
agree_df.write_csv(REP + "rule_index_agreement.csv")
print("\n(c) model leverage vs rule-based index:")
print(agree_df)

# where pressure sits: over x wickets-lost grid (share of high-pressure balls)
grid = (
    lb.group_by(["innings", "overid", "wkts_before"]).agg(pl.col("high_pressure").mean().alias("share_high"), pl.len().alias("n"))
    .filter(pl.col("n") >= 30)
    .sort(["innings", "overid", "wkts_before"])
)
grid.write_csv(REP + "pressure_grid.csv")
for inn in (1, 2):
    (grid.filter(pl.col("innings") == inn).with_columns(pl.col("wkts_before").cast(pl.Utf8))
     .pivot(on="wkts_before", index="overid", values="share_high").sort("overid")
     .write_csv(REP + f"pressure_grid_inn{inn}_wide.csv"))
print("\nshare of high-pressure balls by phase and innings:")
print(lb.group_by(["innings", "phase"]).agg(pl.col("high_pressure").mean().round(3)).sort(["innings", "phase"]))

with open(REP + "summary.json", "w") as f:
    json.dump({
        "scope_matches": p["match_id"].n_unique(), "scope_balls": p.height, "years": YEARS,
        "model_choice": CHOICE,
        "heldout": summ.to_dicts(),
        "ece": {int(k): float(v) for k, v in ece.rows()},
        "high_pressure_threshold_pct": HIGH,
        "rule_agreement": agree_df.to_dicts(),
    }, f, indent=2)
print("\nOK")
