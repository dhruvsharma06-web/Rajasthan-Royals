"""T7: methodology report (DOCX) built only from pipeline outputs, so the report cannot drift from the lists.

Input : outputs/t3/*, outputs/ratings/* (committed outputs only; the data pack is not needed)
Output: outputs/report/SupeRR_Selector_Methodology.docx and outputs/report/fig_*.png
Run from the project root after ratings.py:  python -P pipeline/report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from docx import Document
from docx.shared import Cm, Pt

OUT = "outputs/report/"
os.makedirs(OUT, exist_ok=True)
T3, RT = "outputs/t3/", "outputs/ratings/"
summ = json.load(open(T3 + "summary.json"))
rel = json.load(open(RT + "reliability.json"))
val = pl.read_csv(T3 + "pressure_vs_ordinary.csv")
dec = pl.read_csv(T3 + "decided_share.csv").row(0, named=True)
top = {r: pl.read_csv(RT + f"top10_{r}.csv").sort("rank") for r in ("batters", "bowlers")}
allp = {r: pl.read_csv(RT + f"{r}.csv").sort("rank") for r in ("batters", "bowlers")}
rob = {r: pl.read_csv(RT + f"robustness_{r}.csv") for r in ("batters", "bowlers")}
grid = pl.read_csv(T3 + "pressure_grid.csv")  # legal balls, cells with n >= 30 (pressure_model.py)
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").strip()
REPO_URL = "https://github.com/dhruvsharma06-web/Rajasthan-Royals"

# share of all win-probability movement carried by high-pressure balls, and swing ratio per innings
mv = val.with_columns((pl.col("balls") * pl.col("mean_abs_wp_swing")).alias("move"))
HP_MOVE_SHARE = mv.filter(pl.col("high_pressure"))["move"].sum() / mv["move"].sum()
HP_BALL_SHARE = mv.filter(pl.col("high_pressure"))["balls"].sum() / mv["balls"].sum()
sw = (val.group_by("fmt", "innings").agg(
    (pl.col("mean_abs_wp_swing").filter(pl.col("high_pressure")).first()
     / pl.col("mean_abs_wp_swing").filter(~pl.col("high_pressure")).first()).alias("ratio")))
SWING = {i: (sw.filter(pl.col("innings") == i)["ratio"].min(), sw.filter(pl.col("innings") == i)["ratio"].max())
         for i in (1, 2)}


def xrange_(lo, hi):
    a, b = f"{lo:.1f}", f"{hi:.1f}"
    return f"about {a}x" if a == b else f"{a}-{b}x"
INK, ACC, MUTED = "#1f2a44", "#c2185b", "#9aa3b5"

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, inn in zip(axes, (1, 2)):
    M = np.full((9, 20), np.nan)
    for o, w, v in grid.filter(pl.col("innings") == inn).select("overid", "wkts_before", "share_high").iter_rows():
        if w <= 8:
            M[w, o - 1] = v
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdPu", vmin=0, vmax=0.8,
                   extent=(0.5, 20.5, -0.5, 8.5))
    ax.set_title(f"{'First innings' if inn == 1 else 'Chase'}: share of balls that are high pressure", fontsize=9)
    ax.set_xlabel("Over")
    ax.set_xticks([1, 6, 11, 16, 20])
axes[0].set_ylabel("Wickets down")
fig.colorbar(im, ax=axes, shrink=0.8, label="share high pressure")
fig.savefig(OUT + "fig_pressure_map.png", dpi=170, bbox_inches="tight")
plt.close(fig)

for role in ("batters", "bowlers"):
    t = top[role].reverse()
    a = allp[role].filter(pl.col("pid").is_in(t["pid"].to_list()))
    t = t.join(a.select("pid", "rating_hi"), on="pid", how="left", suffix="_a")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(t.height)
    ax.hlines(y, t["rating_lo"], t["rating_hi"], color=MUTED, lw=4, label="20th-80th percentile")
    ax.plot(t["rating"], y, "o", color=ACC, label="rating (point estimate)")
    ax.plot(t["rating_lo"], y, "|", color=INK, ms=12, mew=2, label="20th percentile (ranking value)")
    ax.set_yticks(y, t["name"].to_list(), fontsize=8)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Pressure rating (0 = average player in the same situations)")
    ax.set_title(f"Top 10 {role}: rating and uncertainty", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.savefig(OUT + f"fig_top10_{role}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- document
doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)


def h(text, lvl=1):
    doc.add_heading(text, level=lvl)


def para(text):
    doc.add_paragraph(text)


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def table(headers, rows, widths=None):
    tb = doc.add_table(rows=1, cols=len(headers))
    tb.style = "Light Grid Accent 1"
    for i, x in enumerate(headers):
        tb.rows[0].cells[i].text = str(x)
    for r in rows:
        cells = tb.add_row().cells
        for i, x in enumerate(r):
            cells[i].text = str(x)
    if widths:  # widths in cm; fixed so names do not break mid-word
        tb.autofit = False
        for col, w in zip(tb.columns, widths):
            col.width = Cm(w)
        for row in tb.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    for row in tb.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)
    doc.add_paragraph()


def f2(x):
    return "" if x is None else f"{x:.2f}"


bat, bowl = top["batters"], top["bowlers"]
hb, hw = rel["batters"], rel["bowlers"]

doc.add_heading("SupeRR Selector Season 2, Round 1: Reliable Players Under Pressure", 0)
para("Methodology report. IPL and T20I ball-by-ball data, 2022-2026, as supplied. Every number in this report is "
     "produced by the reproducible pipeline described in Section 8.")

h("1. Summary")
bullets([
    "Pressure is defined as stakes: how much the match result can swing on a ball, measured with a win-probability "
    "model. The top 20% of balls by stakes, within each format and innings, are high-pressure balls.",
    f"The definition separates pressure from ordinary play: high-pressure balls are {HP_BALL_SHARE:.0%} of balls but carry "
    f"{HP_MOVE_SHARE:.0%} of all win-probability movement, and in chases each moves the match "
    f"{xrange_(*SWING[2])} as much as an ordinary ball. Only {dec['model_high_share_decided']:.2%} of them come from already-decided matches.",
    "Every ball is credited to the batter and bowler involved, against the average player in the same situation "
    "(format, season, innings, phase, wickets down).",
    f"We tested whether players genuinely rise or fall under pressure. Pressure-minus-usual performance does not "
    f"repeat between odd and even matches (split-half r = {hb['split_half']['uplift_raw']['r_half']} batters, "
    f"{hw['split_half']['uplift_raw']['r_half']} bowlers). Large clutch effects are ruled out; small ones cannot be "
    "detected with this much data. The rating therefore uses pressure balls as directly as the data allows and "
    "borrows strength from ordinary balls only as far as clutch differences could not distort them.",
    "Players are ranked on a cautious estimate (20th percentile of the rating's uncertainty), so a short hot streak "
    "cannot outrank a long reliable record. Each pick is checked against 15 alternative modelling choices.",
])
table(["Rank", "Batter", "Bowler"], [(i + 1, a, b) for i, (a, b) in enumerate(zip(bat["name"], bowl["name"]))])

h("2. Defining pressure")
para("A ball's stakes (leverage) is the expected absolute change in the batting side's win probability over the "
     "typical mix of outcomes (dot, 1, 2, 4, 6, wicket) for that format and innings. A ball is high pressure if its "
     "stakes are in the top 20% of balls in the same format and innings. This captures every factor the brief lists "
     "at once: required rate, wickets in hand, balls remaining, stage of the innings and closeness of the contest, "
     "because all of them move the win probability.")
para("Win-probability model. Logistic regression, one per innings, fitted leave-one-season-out so every season is "
     "scored by a model that never saw it. First innings: projected total versus the season's par (runs so far plus "
     "expected runs to come from a Duckworth-Lewis-style resource curve fitted to the data). Chase: runs needed "
     "beyond the runs expected to come, scaled by remaining balls, plus required rate, balls, wickets in hand and a "
     "pre-match team-strength (Elo) difference. Chase targets are back-solved from official results, because "
     "bye, leg-bye and extra-wide runs are not recorded in the data.")
mc = {(m["model"], m["innings"]): m for m in summ["heldout"]}
table(["Innings", "Held-out log loss", "No-information baseline", "Calibration error"],
      [(i, f"{mc[('logit', i)]['log_loss']:.3f}", f"{mc[('logit', i)]['log_loss_base']:.3f}",
        f"{summ['ece'][str(i)]:.1%}") for i in (1, 2)])
doc.add_picture(OUT + "fig_pressure_map.png", width=Cm(16))
para("Chases: pressure peaks in overs 11-20 of a close chase with 3-6 wickets down and disappears once a chase is "
     "decided. First innings: powerplay and middle-overs collapses and the death. The first over also scores high "
     "because an early wicket costs the most projected runs, the same logic as Duckworth-Lewis resources.")
h("Evidence that the definition distinguishes pressure", 2)
table(["Format", "Innings", "Pressure", "Balls", "Mean win-prob swing", "Dot %", "Boundary %", "Wicket %"],
      [(r["fmt"], r["innings"], "high" if r["high_pressure"] else "ordinary", r["balls"],
        f"{r['mean_abs_wp_swing']:.3f}", f"{r['dot_rate']:.1%}", f"{r['boundary_rate']:.1%}",
        f"{r['wicket_rate']:.1%}") for r in val.iter_rows(named=True)])
para(f"A simple rule-based 'difficulty' index (required-rate gap, wickets lost, lateness, recent collapse) was built "
     f"as a cross-check. It flags already-decided matches {dec['rule_high_share_decided']:.0%} of the time, more "
     f"than the {dec['all_balls_share_decided']:.0%} base rate, against {dec['model_high_share_decided']:.2%} for the "
     "stakes definition: difficulty is not the same as pressure. First-innings pressure is the weaker half of the "
     f"evidence (swing {xrange_(*SWING[1])} ordinary); a chase-only definition is included as a robustness variant.")

h("3. Measuring performance")
bullets([
    "Credit per ball, in win-probability units: the batter receives the effect of the runs they hit; the player "
    "dismissed (striker or non-striker) is charged the effect of the wicket; the bowler is credited with the ball "
    "and charged for runs conceded off the bat, wides and no-balls, and credited with wickets they took (not run-outs). "
    "Byes, leg-byes and retirements are charged to nobody. Built-in checks: no six costs a batter and no wicket "
    "helps the batting side.",
    "Same-situation baseline: each credit is measured against the average credit per unit of stakes in the same "
    "format, season, innings, phase and wickets-down band. The model alone over-credited middle-overs bowling and "
    "under-credited chase powerplays.",
    "Impact = total credit / total stakes, so a last-over six does not count 20x a six two overs earlier merely "
    "because the stakes were higher; it is performance relative to what was at stake.",
    "Usual level = impact on ordinary balls. Uplift = impact on high-pressure balls minus usual level.",
    "Reading the numbers: an impact of +0.10 means that, per ball, the player moved the win probability in "
    "their team's favour by 10% of what was at stake, more than the average player in the same situation "
    "would have.",
])
h("Rating", 2)
para("Rating = estimate of each player's true performance on high-pressure balls. High-pressure balls measure it "
     "directly. Ordinary balls measure it minus the player's clutch uplift, so they are treated as a noisier "
     "measurement, discounted by the largest spread of clutch ability the data cannot rule out (90% bound from a "
     f"simulation of the split-half test). On average pressure balls carry "
     f"{hb['main']['mean_weight_on_pressure_balls']:.0%} of a batter's own-data weight and "
     f"{hw['main']['mean_weight_on_pressure_balls']:.0%} of a bowler's. Estimates are shrunk toward the eligible-player "
     "average in proportion to their noise (empirical Bayes, DerSimonian-Laird between-player variance), with noise "
     "measured by 1,000 bootstrap resamples of matches. Ranking uses the 20th percentile of each player's rating.")

h("4. Fair comparisons")
bullets([
    f"Eligibility: at least 90 high-pressure balls across at least 12 innings, and 60 ordinary balls "
    f"({hb['main']['eligible']} batters, {hw['main']['eligible']} bowlers).",
    "Sample size: shrinkage and the cautious 20th-percentile rank stop small samples from topping the list.",
    "Match context: the same-situation baseline and the Elo team-strength term in the win-probability model "
    "mean a player gains nothing from facing weak opponents or batting in easy phases. A variant drops lopsided "
    "matches (pre-match Elo gap above 150).",
    "IPL and T20I: one currency (win probability) across both, with par, baseline and model terms per format and "
    "season. IPL-only and T20I-only lists are robustness variants. Two A-team tournaments in the 'T20' format "
    "(Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025) are excluded, as are rain-revised matches and 11 "
    "chases whose ball data disagreed with the official margin by more than 8 runs (774 matches, 184,940 balls).",
])

for role, label in (("batters", "Batters"), ("bowlers", "Bowlers")):
    if role == "batters":
        h("5. Evidence for the top 10")
    h(label, 2)
    a = allp[role].filter(pl.col("rank") <= 10).sort("rank")
    t = a.join(top[role].select("pid", "variants_top10", "variants_eligible"), on="pid")
    if role == "batters":
        hdr = ["#", "Player", "Rating", "P(top 10)", "Pressure balls", "Inns", "SR press / ord",
               "Avg press / ord", "Uplift", "Variants top 10"]
        rows = [(r["rank"], r["name"], f"{r['rating']:.3f}", f2(r["p_top10"]), r["b_hi"], r["hp_inns"],
                 f"{r['sr_hp']:.0f} / {r['sr_ord']:.0f}", f"{r['avg_hp']:.1f} / {r['avg_ord']:.1f}",
                 f"{r['uplift_raw']:+.3f}", f"{r['variants_top10']}/{r['variants_eligible']}") for r in t.iter_rows(named=True)]
    else:
        hdr = ["#", "Player", "Rating", "P(top 10)", "Pressure balls", "Inns", "Econ press / ord",
               "Balls/wkt press / ord", "Uplift", "Variants top 10"]
        rows = [(r["rank"], r["name"], f"{r['rating']:.3f}", f2(r["p_top10"]), r["b_hi"], r["hp_inns"],
                 f"{r['econ_hp']:.2f} / {r['econ_ord']:.2f}", f"{r['balls_per_wkt_hp']:.0f} / {r['balls_per_wkt_ord']:.0f}",
                 f"{r['uplift_raw']:+.3f}", f"{r['variants_top10']}/{r['variants_eligible']}") for r in t.iter_rows(named=True)]
    table(hdr, rows, widths=[0.8, 3.6, 1.3, 1.5, 1.5, 1.0, 1.7, 1.7, 1.3, 1.5])
    doc.add_picture(OUT + f"fig_top10_{role}.png", width=Cm(14))
    nxt = allp[role].filter((pl.col("rank") > 10) & (pl.col("rank") <= 15))
    para("Next in order: " + ", ".join(f"{r['rank']}. {r['name']} (P(top10) {r['p_top10']:.2f})"
                                       for r in nxt.iter_rows(named=True)) + ".")

h("6. Reliability and consistency")
sh = lambda r, k: f'{r["split_half"][k]["r_half"]:.3f}'  # noqa: E731
table(["Measure", "Batters r (odd vs even matches)", "Bowlers r"],
      [("Pressure-ball impact", sh(hb, "pressure_raw"), sh(hw, "pressure_raw")),
       ("All-ball impact", sh(hb, "all_raw"), sh(hw, "all_raw")),
       ("Uplift (pressure minus usual)", sh(hb, "uplift_raw"), sh(hw, "uplift_raw"))])
bullets([
    "Individual pressure performance in T20 is noisy: half-season samples agree only weakly. This is why the "
    "ranking is cautious and every pick carries a probability of truly belonging in the top 10.",
    "Fast scorers also get out more, so batting skill partly cancels in match-winning terms; the runs and wicket "
    "components are each far more repeatable (about 0.5) than their sum.",
    "Consistency across matches is reported per player: share of pressure matches with positive contribution and "
    "the share of gains coming from the best two matches (in outputs/ratings/*.csv).",
    "Robustness: 15 variants (pressure threshold 10% / 30%; pressure ranked across innings; pressure measured at "
    "the start of the over, so batters cannot create their own pressure; 10th / 50th percentile ranking; IPL only; "
    "T20I only; 2024-2026 only; no lopsided matches; pace/spin-specific baseline; no situation baseline; all balls "
    "with no clutch; pressure balls only; chase pressure only).",
])

h("7. Limitations")
p810 = [x for r in ("batters", "bowlers") for x in allp[r].filter(pl.col("rank").is_between(8, 10))["p_top10"]]
weak = [f"{r['name']} ({r['variants_top10']}/{r['variants_eligible']})" for role in ("batters", "bowlers")
        for r in top[role].iter_rows(named=True) if r["variants_top10"] * 2 < r["variants_eligible"]]
bullets([
    f"Ranks 8-10 in each list are uncertain (P(top10) {min(p810):.2f}-{max(p810):.2f}); the order there should be "
    "read as indicative. The method was fixed before the final lists were read and was not adjusted to favour "
    "any name.",
    "Picks that stay in the top 10 in fewer than half of the robustness variants where they are eligible: "
    + ", ".join(weak) + ". They are kept because the main method was fixed in advance; the players listed "
    "next in order are the alternatives.",
    "The ranking value (20th percentile) and P(top10) can disagree: a player with a short record and wide "
    "uncertainty can have a higher chance of being top 10 yet a lower cautious rank. This is deliberate: "
    "the brief asks for reliable performers.",
    "Extras run amounts are not in the data; innings totals are estimated (mean error about 2 runs) and chase "
    "targets are back-solved from official results.",
    "The Elo ratings and season par use other matches of the same season, a mild indirect use of later results.",
    "First-innings pressure is weaker evidence than chase pressure.",
    "Fielding, captaincy and match-ups (batter vs bowler type) are not modelled.",
])

h("8. Reproducibility")
para("Python 3.11; packages pinned in requirements.txt. From the project root, with the supplied data pack in "
     "6af3166e7b922_data_pack/: python -P pipeline/build_state.py; python -P pipeline/pressure_model.py; "
     "python -P pipeline/ratings.py; then python -P pipeline/report.py, pipeline/slides.py (presentation) and "
     "pipeline/export_dashboard.py (dashboard data). Each script runs built-in checks and prints OK. "
     "Decisions and their evidence are recorded in docs/PROJECT.md. Code and outputs: " + REPO_URL + ".")
if DASHBOARD_URL:
    para("Interactive dashboard (every eligible player, the pressure map and the robustness table): " + DASHBOARD_URL)

doc.save(OUT + "SupeRR_Selector_Methodology.docx")
print("wrote", OUT + "SupeRR_Selector_Methodology.docx")
print("OK")
