"""T4 + T5: player pressure ratings, reliability, robustness, final top-10 lists.

Input : data/processed/ball_pressure.parquet  (from pipeline/pressure_model.py)
Output: outputs/ratings/{batters,bowlers}.csv          every eligible player, main variant
        outputs/ratings/top10_{batters,bowlers}.csv     the submitted lists
        outputs/ratings/robustness_{batters,bowlers}.csv  rank of each player under every variant
        outputs/ratings/reliability.json               split-half reliability, eligibility, settings
Run from the project root:  python -P pipeline/ratings.py

Method (docs/PROJECT.md D5-D8, G1-G11):
- Credit per ball (win-probability units): batter = runs effect (+ wicket effect if he was the one out);
  bowler = -(runs charged to him + ball consumed) + wicket effect if he took the wicket.
- Impact = sum(credit) / sum(stake), stake = expected absolute win-probability swing of the ball (G1).
  Dimensionless: +0.10 means each ball moved the match a tenth of a typical swing in the player's favour.
- Usual = impact on ordinary balls; uplift = impact on high-pressure balls - usual.
- Rating (D9) = empirical-Bayes-shrunk impact on high-pressure balls, shrunk toward the eligible mean with
  the DerSimonian-Laird between-player variance and each player's noise from a match-level bootstrap.
  Usual level and uplift (pressure minus usual) are reported for every player. The split-half test cannot
  detect uplift as a repeatable skill but has low power: it rules out large clutch effects, not small ones.
  The all-balls stake-weighted rating is a robustness variant.
- Rank = 20th percentile of the bootstrap distribution of the rating (cautious estimate, G11).
"""
import json
import os

import numpy as np
import polars as pl

IN = "data/processed/ball_pressure.parquet"
OUT = "outputs/ratings/"
SEED = 11
B_MAIN, B_VAR = 1000, 300
Q_RANK = 0.20
os.makedirs(OUT, exist_ok=True)

ELIG = {"bat": {"hp_balls": 90, "hp_inns": 12}, "bowl": {"hp_balls": 90, "hp_inns": 12}}

base = pl.read_parquet(IN).with_columns(
    pl.col("leverage_raw").alias("stake"),
    # G2 sensitivity: pressure of the over as it started, so a batter's own slow scoring within the over
    # cannot create the pressure he is credited for
    pl.col("pressure_pct").first().over(["match_id", "innings", "overid"]).alias("pressure_pct_overstart"),
    pl.col("match_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("wkts_before").cut([1, 3, 5], labels=["0-1", "2-3", "4-5", "6+"], left_closed=False).alias("wkt_band"),
)

# D8: credit relative to the average player in the same situation. The win-probability model is calibrated
# overall but not by phase (bowlers averaged -0.043 per unit stake in chase powerplays and +0.02 in middle
# overs), so each credit is measured against the average rate in its format x innings x phase x wickets cell.
CELL = ["fmt", "year", "innings", "phase", "wkt_band"]  # per season (F4)


def baseline_adjust(df, extra_cell=()):
    cell = CELL + list(extra_cell)
    rates = df.group_by(cell).agg(
        (pl.col("cr_striker_runs").filter(pl.col("faced")).sum()
         / pl.col("stake").filter(pl.col("faced")).sum()).alias("_r_run"),
        (pl.col("cr_wicket").filter(pl.col("faced")).sum()
         / pl.col("stake").filter(pl.col("faced")).sum()).alias("_r_wkt"),
        (pl.col("cr_bowler").sum() / pl.col("stake").sum()).alias("_r_bowl"),
    )
    return df.join(rates, on=cell, how="left").with_columns(
        pl.when(pl.col("faced")).then(pl.col("cr_striker_runs") - pl.col("stake") * pl.col("_r_run"))
        .otherwise(0.0).alias("cr_striker_runs"),
        # expected wicket cost is charged against the striker's ball; the actual cost goes to whoever was out
        pl.col("cr_wicket").alias("cr_wicket"),
        pl.when(pl.col("faced")).then(-pl.col("stake") * pl.col("_r_wkt")).otherwise(0.0).alias("cr_wkt_expected"),
        (pl.col("cr_bowler") - pl.col("stake") * pl.col("_r_bowl")).alias("cr_bowler"),
    ).drop(["_r_run", "_r_wkt", "_r_bowl"])

names_bat = base.group_by("striker_id").agg(pl.col("striker_name").mode().first().alias("name"))
names_bowl = base.group_by("bowler_id").agg(pl.col("bowler_name").mode().first().alias("name"))
dup_names = set(
    pl.concat([names_bat.rename({"striker_id": "id"}), names_bowl.rename({"bowler_id": "id"})])
    .unique().group_by("name").agg(pl.col("id").n_unique().alias("n")).filter(pl.col("n") > 1)["name"]
)


def ball_rows(df, role, pcol, thr):
    """One row per (player, ball) with credit, stake, high-pressure flag."""
    hp = pl.col(pcol) >= thr
    if role == "bat":
        faced = df.filter(pl.col("faced")).select(
            pl.col("striker_id").alias("pid"), "match_id", "innings", hp.alias("hp"),
            pl.col("stake"), (pl.col("cr_striker_runs") + pl.col("cr_wkt_expected")).alias("credit"),
            pl.col("runs").alias("r"), pl.lit(1).alias("ball"),
            (pl.col("striker_out") & ~pl.col("dismissal_type").fill_null("").str.contains("Retire"))
            .cast(pl.Int32).alias("out"),
            (pl.col("is_four") | pl.col("is_six")).cast(pl.Int32).alias("bnd"),
            (pl.col("runs") == 0).cast(pl.Int32).alias("dot"),
        )
        # non-striker run-outs: wicket cost to the batter who was out, no ball faced
        ns = df.filter(pl.col("is_wicket") & (pl.col("dismissed_player_id") == pl.col("non_striker_id"))
                       & ~pl.col("dismissal_type").fill_null("").str.contains("Retire")).select(
            pl.col("non_striker_id").alias("pid"), "match_id", "innings", hp.alias("hp"),
            pl.lit(0.0).alias("stake"), pl.col("cr_wicket").alias("credit"),
            pl.lit(0).alias("r"), pl.lit(0).alias("ball"), pl.lit(1).alias("out"),
            pl.lit(0).alias("bnd"), pl.lit(0).alias("dot"),
        )
        # no `faced` filter: stumpings and run-outs off wides are charged to the striker too (review finding)
        st = df.filter(pl.col("is_wicket") & (pl.col("dismissed_player_id") == pl.col("striker_id"))
                       & ~pl.col("dismissal_type").fill_null("").str.contains("Retire")).select(
            pl.col("striker_id").alias("pid"), "match_id", "innings", hp.alias("hp"),
            pl.lit(0.0).alias("stake"), pl.col("cr_wicket").alias("credit"),
            pl.lit(0).alias("r"), pl.lit(0).alias("ball"), pl.lit(0).alias("out"),
            pl.lit(0).alias("bnd"), pl.lit(0).alias("dot"),
        )
        return pl.concat([faced, ns, st], how="vertical_relaxed")
    return df.select(
        pl.col("bowler_id").alias("pid"), "match_id", "innings", hp.alias("hp"),
        pl.col("stake"), pl.col("cr_bowler").alias("credit"),
        pl.col("bowler_runs").alias("r"), pl.col("is_legal").cast(pl.Int32).alias("ball"),
        pl.col("bowler_wicket").cast(pl.Int32).alias("out"),
        (pl.col("is_four") | pl.col("is_six")).cast(pl.Int32).alias("bnd"),
        ((pl.col("bowler_runs") == 0) & pl.col("is_legal")).cast(pl.Int32).alias("dot"),
    )


def rate(df, role, pcol="pressure_pct", thr=0.80, elig_scale=1.0, B=B_MAIN, q=Q_RANK, seed=SEED, detail=False,
         target="pressure", tau_u2=None):
    rows = ball_rows(df, role, pcol, thr)
    e = ELIG[role]
    pm = rows.group_by(["pid", "match_id"]).agg(
        pl.col("credit").filter(pl.col("hp")).sum().alias("c_hi"),
        pl.col("stake").filter(pl.col("hp")).sum().alias("s_hi"),
        pl.col("credit").filter(~pl.col("hp")).sum().alias("c_or"),
        pl.col("stake").filter(~pl.col("hp")).sum().alias("s_or"),
        pl.col("ball").filter(pl.col("hp")).sum().alias("b_hi"),
        pl.col("ball").filter(~pl.col("hp")).sum().alias("b_or"),
    )
    tot = pm.group_by("pid").agg(
        pl.col("b_hi").sum(), pl.col("b_or").sum(),
        (pl.col("b_hi") > 0).sum().alias("hp_inns"),
        pl.col("match_id").n_unique().alias("matches"),
    )
    elig = tot.filter((pl.col("b_hi") >= e["hp_balls"] * elig_scale) & (pl.col("hp_inns") >= e["hp_inns"] * elig_scale)
                      & (pl.col("b_or") >= 60 * elig_scale))
    pids = elig["pid"].to_list()
    if len(pids) < 15:
        return None
    pm = pm.filter(pl.col("pid").is_in(pids))
    mids = pm["match_id"].unique().sort().to_list()
    pi = {p: i for i, p in enumerate(pids)}
    mi = {m: i for i, m in enumerate(mids)}
    P, M = len(pids), len(mids)
    A = {k: np.zeros((M, P)) for k in ("c_hi", "s_hi", "c_or", "s_or")}
    ip = np.array([pi[x] for x in pm["pid"]])
    im = np.array([mi[x] for x in pm["match_id"]])
    for k in A:
        np.add.at(A[k], (im, ip), pm[k].to_numpy())

    def est(w):  # w: (R, M) match weights -> usual, uplift, all-balls impact (R, P)
        s = {k: w @ A[k] for k in A}
        hi = s["c_hi"] / np.maximum(s["s_hi"], 1e-9)
        us = s["c_or"] / np.maximum(s["s_or"], 1e-9)
        al = (s["c_hi"] + s["c_or"]) / np.maximum(s["s_hi"] + s["s_or"], 1e-9)
        return us, hi - us, al, hi

    us0, up0, al0, hi0 = (x[0] for x in est(np.ones((1, M))))
    rng = np.random.default_rng(seed)
    W = rng.poisson(1.0, size=(B, M)).astype(float)
    us_b, up_b, al_b, hi_b = est(W)
    out = {}
    for name, x0, xb in (("usual", us0, us_b), ("uplift", up0, up_b), ("all", al0, al_b), ("pressure", hi0, hi_b)):
        v = np.maximum(xb.var(axis=0), 1e-12)
        # DerSimonian-Laird between-player variance, precision-weighted. The plain moment estimator
        # (var(x) - mean(v)) collapsed to 0 when the field had many small samples, which shrank every
        # player to the same value and made the order arbitrary.
        w = 1.0 / v
        mu_w = (w * x0).sum() / w.sum()
        Q = (w * (x0 - mu_w) ** 2).sum()
        tau2 = max(0.0, (Q - (len(x0) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        ws = 1.0 / (v + tau2)
        mu = (ws * x0).sum() / ws.sum()
        shrink = tau2 / (tau2 + v)
        post0 = mu + shrink * (x0 - mu)
        # posterior draws: centred on the shrunk value with variance shrink * v (the earlier mu + shrink*(xb-mu)
        # gave shrink^2 * v, intervals 1.5-1.8x too narrow; found in review)
        out[name] = (post0, post0 + np.sqrt(shrink) * (xb - x0), tau2, shrink)
    w_press = np.ones(P)
    if tau_u2 is None:
        # single-component rating: target="pressure" (pressure balls only) or "all" (all balls)
        r0, rb = out[target][0], out[target][1]
    else:
        # D9: true pressure performance theta. Pressure balls measure theta with noise v_hi. Ordinary balls
        # measure theta - uplift with noise v_or, and uplift varies between players by tau_u2 (clutch spread),
        # so they count as a noisier measurement (v_or + tau_u2). Prior: theta ~ N(mu_p, tau_skill + tau_u2).
        # tau_u2 = 0 -> both kinds of ball count equally; large tau_u2 -> pressure balls only.
        v_hi = np.maximum(hi_b.var(axis=0), 1e-12)
        v_or = np.maximum(us_b.var(axis=0), 1e-12)
        mu_u = float(up0.mean())
        tau_p2 = out["all"][2] + tau_u2
        mu_p = float((hi0 / v_hi).sum() / (1 / v_hi).sum())
        a, bh, bo = (1 / tau_p2 if tau_p2 > 0 else 1e12), 1 / v_hi, 1 / (v_or + tau_u2)

        def comb(xh, xo):
            return (a * mu_p + bh * xh + bo * (xo + mu_u)) / (a + bh + bo)

        post_sd = np.sqrt(1 / (a + bh + bo))
        r0 = comb(hi0, us0)
        fb = comb(hi_b, us_b)
        rb = r0 + (post_sd / np.maximum(fb.std(axis=0), 1e-12)) * (fb - r0)
        w_press = bh / (bh + bo)  # share of the player's own-data weight carried by pressure balls
    lo = np.quantile(rb, q, axis=0)
    rank_b = (-rb).argsort(axis=1).argsort(axis=1)  # 0 = best within replicate
    p_top10 = (rank_b < 10).mean(axis=0)
    res = pl.DataFrame({
        "pid": pids, "rating": r0, "rating_lo": lo, "rating_hi": np.quantile(rb, 0.8, axis=0),
        "p_top10": p_top10, "usual_raw": us0, "uplift_raw": up0, "pressure_raw": hi0, "all_raw": al0,
        "pressure_lo": np.quantile(hi_b, 0.1, axis=0), "pressure_hi": np.quantile(hi_b, 0.9, axis=0),
        "all_shrunk": out["all"][0], "usual_shrunk": out["usual"][0], "uplift_shrunk": out["uplift"][0],
        "pressure_shrunk": out["pressure"][0], "uplift_weight": out["uplift"][3],
        "weight_on_pressure_balls": w_press,
    }).join(elig, on="pid").sort("rating_lo", descending=True).with_row_index("rank", offset=1)
    meta = {"eligible": P, "tau2_all": out["all"][2], "tau2_usual": out["usual"][2],
            "tau2_uplift": out["uplift"][2], "tau2_pressure": out["pressure"][2],
            "mean_uplift_weight": float(out["uplift"][3].mean()),
            "mean_all_weight": float(out["all"][3].mean()), "mean_uplift_raw": float(up0.mean()),
            "tau_u2": tau_u2, "mean_weight_on_pressure_balls": float(np.mean(w_press))}
    if detail:
        return res, rows.filter(pl.col("pid").is_in(pids)), meta
    return res, meta


def readable(rows, role):
    """Plain cricket numbers, pressure vs ordinary."""
    g = rows.group_by(["pid", "hp"]).agg(pl.col("r").sum(), pl.col("ball").sum(), pl.col("out").sum(),
                                         pl.col("bnd").sum(), pl.col("dot").sum())
    if role == "bat":
        g = g.with_columns((100 * pl.col("r") / pl.col("ball")).alias("sr"),
                           (pl.col("r") / pl.max_horizontal(pl.col("out"), 1)).alias("avg"),
                           (pl.col("ball") / pl.max_horizontal(pl.col("out"), 1)).alias("balls_per_out"))
        cols = ["sr", "avg", "balls_per_out"]
    else:
        g = g.with_columns((6 * pl.col("r") / pl.col("ball")).alias("econ"),
                           (pl.col("ball") / pl.max_horizontal(pl.col("out"), 1)).alias("balls_per_wkt"))
        cols = ["econ", "balls_per_wkt"]
    g = g.with_columns((pl.col("dot") / pl.col("ball")).alias("dot_pct"), (pl.col("bnd") / pl.col("ball")).alias("bnd_pct"))
    cols += ["dot_pct", "bnd_pct", "ball", "out"]
    hi = g.filter(pl.col("hp")).select(["pid"] + [pl.col(c).alias(f"{c}_hp") for c in cols])
    lo = g.filter(~pl.col("hp")).select(["pid"] + [pl.col(c).alias(f"{c}_ord") for c in cols])
    return hi.join(lo, on="pid", how="left")


def consistency(rows):
    """Per-match pressure credit: share of positive matches, share of positive total from best 2 matches."""
    pm = rows.filter(pl.col("hp")).group_by(["pid", "match_id"]).agg(pl.col("credit").sum().alias("c"))
    return pm.group_by("pid").agg(
        (pl.col("c") > 0).mean().alias("share_positive_matches"),
        pl.len().alias("pressure_matches"),
        (pl.col("c").filter(pl.col("c") > 0).sort(descending=True).head(2).sum()
         / pl.col("c").filter(pl.col("c") > 0).sum()).alias("best2_share_of_gains"),
    )


def split_half(df, role):
    """Correlation of odd-match vs even-match estimates across players eligible in both halves."""
    out = {}
    halves = [df.filter(pl.col("match_id") % 2 == k) for k in (0, 1)]
    rs = [rate(h, role, elig_scale=0.5, B=200) for h in halves]
    if any(r is None for r in rs):
        return out
    a, b = rs[0][0], rs[1][0]
    j = a.join(b, on="pid", suffix="_b")
    out["_uplift_noise_half"] = float((a["uplift_raw"].var() + b["uplift_raw"].var()) / 2)
    out["_players"] = j.height
    for k in ("all_raw", "usual_raw", "uplift_raw", "pressure_raw"):
        r = float(np.corrcoef(j[k], j[f"{k}_b"])[0, 1])
        out[k] = {"r_half": round(r, 3), "spearman_brown_full": round(2 * r / (1 + r), 3) if r > -1 else None,
                  "players": j.height}
    return out


def clutch_bound(sh, p_target=0.10, seed=5):
    """Largest clutch spread (between-player variance of true uplift) the split-half test cannot rule out:
    the tau2 at which P(split-half r <= observed r) = 10%, by simulation with the measured half-sample noise."""
    r_obs, noise, n = sh["uplift_raw"]["r_half"], sh["_uplift_noise_half"], sh["_players"]
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 4 * noise, 81)
    probs = []
    for tau2 in grid:
        t = rng.normal(0, np.sqrt(tau2), (3000, n))
        a = t + rng.normal(0, np.sqrt(noise), (3000, n))
        b = t + rng.normal(0, np.sqrt(noise), (3000, n))
        ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        r = (ac * bc).sum(1) / np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1))
        probs.append(float(np.mean(r <= r_obs)))
    probs = np.array(probs)
    idx = int(np.argmax(probs <= p_target)) if (probs <= p_target).any() else len(grid) - 1
    return float(grid[idx]), {"observed_r_half": r_obs, "half_sample_noise_var": noise, "players": n,
                              "tau_u2_upper90": float(grid[idx]),
                              "p_r_le_observed": dict(zip([round(float(g), 5) for g in grid[::10]],
                                                          [round(float(x), 3) for x in probs[::10]]))}


def fmt_mix(df, role):
    idc = "striker_id" if role == "bat" else "bowler_id"
    team = "batting_team_id" if role == "bat" else "bowling_team_id"
    return df.group_by(idc).agg(
        (pl.col("fmt") == "IPL").mean().alias("share_ipl_balls"),
        pl.col("match_date").max().alias("last_match"),
        pl.col("year").min().alias("first_year"), pl.col("year").max().alias("last_year"),
    ).rename({idc: "pid"})


full_raw = base
full = baseline_adjust(full_raw)
chk = full.group_by(["innings", "phase"]).agg((pl.col("cr_bowler").sum() / pl.col("stake").sum()).round(4).alias("bowl"),
                                              ((pl.col("cr_striker_runs") + pl.col("cr_wkt_expected")).sum()
                                               + pl.col("cr_wicket").sum()).round(3).alias("bat_sum"))
print("[check] after baseline adjustment, bowler credit per stake by innings x phase (should be ~0):",
      sorted(chk.select("innings", "phase", "bowl").rows()))
VARIANTS = {
    "main": dict(),
    "threshold_top10pct": dict(thr=0.90, elig_scale=0.5),
    "threshold_top30pct": dict(thr=0.70, elig_scale=1.5),
    "pressure_pooled_innings": dict(pcol="pressure_pct_pooled"),
    "pressure_at_over_start": dict(pcol="pressure_pct_overstart"),
    "cautious_10th_pct": dict(q=0.10),
    "median_estimate": dict(q=0.50),
    "ipl_only": dict(data="ipl", elig_scale=0.5),
    "t20i_only": dict(data="t20i", elig_scale=0.5),
    "recent_2024_2026": dict(data="recent", elig_scale=0.6),
    "no_mismatches": dict(data="close", elig_scale=0.8),
    "baseline_by_pace_spin": dict(data="pace_spin"),
    "no_situation_baseline": dict(data="raw"),
    "all_balls_no_clutch": dict(tau_u2=0.0),
    "pressure_balls_only": dict(tau_u2=None, target="pressure"),
    "chase_pressure_only": dict(data="chase", elig_scale=0.6),
}


def subset(name):
    if name == "ipl":
        return full.filter(pl.col("fmt") == "IPL")
    if name == "t20i":
        return full.filter(pl.col("fmt") == "T20I")
    if name == "recent":
        return full.filter(pl.col("year") >= 2024)
    if name == "close":  # drop matches with a big pre-match strength gap (|Elo diff| > 150)
        return full.filter(pl.col("elo_diff").abs().max().over("match_id") <= 150)
    if name == "pace_spin":  # bowlers compared only with bowlers of their own type in the same situation
        return baseline_adjust(full_raw, extra_cell=("pace_or_spin",))
    if name == "chase":  # only chase balls can be high pressure; first-innings balls count as ordinary
        return full.with_columns(pl.when(pl.col("innings") == 1).then(0.0).otherwise(pl.col("pressure_pct"))
                                 .alias("pressure_pct"))
    if name == "raw":  # no situation baseline: model-expected credit only
        return full_raw.with_columns(pl.lit(0.0).alias("cr_wkt_expected"))
    return full


report = {"settings": {"eligibility": ELIG, "bootstrap_main": B_MAIN, "rank_quantile": Q_RANK,
                       "high_pressure": "top 20% of stake within format x innings"}}
for role, names in (("bat", names_bat), ("bowl", names_bowl)):
    label = "batters" if role == "bat" else "bowlers"
    idc = "striker_id" if role == "bat" else "bowler_id"
    sh = split_half(full, role)
    TU, power = clutch_bound(sh)
    res, rows, meta = rate(full, role, detail=True, tau_u2=TU)
    res = (res.join(names.rename({idc: "pid"}), on="pid", how="left")
           .join(readable(rows, role), on="pid", how="left")
           .join(consistency(rows), on="pid", how="left")
           .join(fmt_mix(full, role), on="pid", how="left")
           .with_columns(pl.col("name").is_in(list(dup_names)).alias("name_shared_with_other_id")))
    lead = ["rank", "pid", "name", "rating", "rating_lo", "rating_hi", "p_top10", "pressure_raw", "pressure_lo",
            "pressure_hi", "usual_raw", "uplift_raw", "all_shrunk", "uplift_shrunk", "uplift_weight",
            "b_hi", "b_or", "hp_inns", "matches"]
    res = res.select(lead + [c for c in res.columns if c not in lead])
    res.write_csv(OUT + f"{label}.csv")

    # robustness
    ranks = res.select("pid", "name", pl.col("rank").alias("main"))
    var_meta = {}
    for vname, kw in VARIANTS.items():
        if vname == "main":
            continue
        kw = dict(kw)
        data = subset(kw.pop("data", None))
        kw.setdefault("tau_u2", TU)
        r = rate(data, role, B=B_VAR, **kw)
        if r is None:
            var_meta[vname] = "too few eligible players"
            continue
        ranks = ranks.join(r[0].select("pid", pl.col("rank").alias(vname)), on="pid", how="left")
        var_meta[vname] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r[1].items()}
    vcols = [c for c in ranks.columns if c not in ("pid", "name", "main")]
    ranks = ranks.with_columns(
        pl.sum_horizontal([(pl.col(c) <= 10).cast(pl.Int32) for c in vcols]).alias("variants_top10"),
        pl.sum_horizontal([(pl.col(c) <= 20).cast(pl.Int32) for c in vcols]).alias("variants_top20"),
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in vcols]).alias("variants_eligible"),
    ).sort("main")
    ranks.write_csv(OUT + f"robustness_{label}.csv")

    top = res.head(10).join(ranks.select("pid", "variants_top10", "variants_top20", "variants_eligible"), on="pid")
    top.write_csv(OUT + f"top10_{label}.csv")
    report[label] = {"main": meta, "variant_eligible_counts": var_meta,
                     "split_half": {k: v for k, v in sh.items() if not k.startswith("_")},
                     "clutch_power": power}

    print(f"\n==================== {label.upper()} — eligible {meta['eligible']} ====================")
    print(f"clutch spread upper bound (90%) tau_u2={TU:.5f} vs half-sample uplift noise {power['half_sample_noise_var']:.5f}; "
          f"mean weight on pressure balls {meta['mean_weight_on_pressure_balls']:.2f}; mean raw uplift {meta['mean_uplift_raw']:+.4f}")
    print("split-half reliability:", report[label]["split_half"])
    show = top.select(
        "rank", "name", pl.col("rating").round(3), pl.col("rating_lo").round(3), pl.col("p_top10").round(2),
        pl.col("pressure_raw").round(3), pl.col("usual_raw").round(3), "b_hi", "hp_inns",
        pl.col("share_positive_matches").round(2), pl.col("best2_share_of_gains").round(2),
        pl.col("share_ipl_balls").round(2), "variants_top10", "variants_top20", "variants_eligible",
    )
    print(show.rows())
    nxt = res.slice(10, 10).select("rank", "name", pl.col("rating_lo").round(3), pl.col("p_top10").round(2))
    print("ranks 11-20:", nxt.rows())

with open(OUT + "reliability.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print("\nOK")
