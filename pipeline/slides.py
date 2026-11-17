"""Presentation deck (PPTX) built only from pipeline outputs, like report.py, so slides cannot drift from the lists.

Input : outputs/t3/*, outputs/ratings/*
Output: outputs/report/SupeRR_Selector_Pressure_Ratings.pptx and outputs/report/slide_fig_*.png
Run from the project root after ratings.py:  python -P pipeline/slides.py
Optional: set DASHBOARD_URL to print the deployed dashboard link on the last slide.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = "outputs/report/"
os.makedirs(OUT, exist_ok=True)
T3, RT = "outputs/t3/", "outputs/ratings/"
summ = json.load(open(T3 + "summary.json"))
rel = json.load(open(RT + "reliability.json"))
val = pl.read_csv(T3 + "pressure_vs_ordinary.csv")
dec = pl.read_csv(T3 + "decided_share.csv").row(0, named=True)
cal = pl.read_csv(T3 + "calibration.csv")
grid = pl.read_csv(T3 + "pressure_grid.csv")
allp = {r: pl.read_csv(RT + f"{r}.csv").sort("rank") for r in ("batters", "bowlers")}
top = {r: pl.read_csv(RT + f"top10_{r}.csv").sort("rank") for r in ("batters", "bowlers")}
rob = {r: pl.read_csv(RT + f"robustness_{r}.csv").sort("main") for r in ("batters", "bowlers")}
for r in ("batters", "bowlers"):  # the deck must show exactly the submitted lists
    assert top[r]["pid"].to_list() == allp[r].head(10)["pid"].to_list()
    assert top[r]["name"].n_unique() == 10
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").strip()
REPO_URL = "https://github.com/dhruvsharma06-web/Rajasthan-Royals"

VARIANTS = [("threshold_top10pct", "Top 10% threshold"), ("threshold_top30pct", "Top 30% threshold"),
            ("pressure_pooled_innings", "Ranked across innings"), ("pressure_at_over_start", "Pressure at over start"),
            ("cautious_10th_pct", "10th percentile"), ("median_estimate", "Median"), ("ipl_only", "IPL only"),
            ("t20i_only", "T20I only"), ("recent_2024_2026", "2024-2026"), ("no_mismatches", "No lopsided"),
            ("baseline_by_pace_spin", "Pace/spin baseline"), ("no_situation_baseline", "No situation baseline"),
            ("all_balls_no_clutch", "All balls, no clutch"), ("pressure_balls_only", "Pressure balls only"),
            ("chase_pressure_only", "Chase only")]
assert [k for k, _ in VARIANTS] == rob["batters"].columns[3:-3]

# derived numbers (same formulas as report.py)
mv = val.with_columns((pl.col("balls") * pl.col("mean_abs_wp_swing")).alias("move"))
HP_MOVE = mv.filter(pl.col("high_pressure"))["move"].sum() / mv["move"].sum()
HP_BALLS = mv.filter(pl.col("high_pressure"))["balls"].sum() / mv["balls"].sum()
ratio = {(f, i): val.filter((pl.col("fmt") == f) & (pl.col("innings") == i) & pl.col("high_pressure"))["mean_abs_wp_swing"][0]
         / val.filter((pl.col("fmt") == f) & (pl.col("innings") == i) & ~pl.col("high_pressure"))["mean_abs_wp_swing"][0]
         for f in ("IPL", "T20I") for i in (1, 2)}
chase_rng = f"{min(ratio[('IPL', 2)], ratio[('T20I', 2)]):.1f}-{max(ratio[('IPL', 2)], ratio[('T20I', 2)]):.1f}x"
mc = {(m["model"], m["innings"]): m for m in summ["heldout"]}
hb, hw = rel["batters"], rel["bowlers"]

# ---------------------------------------------------------------- palette
# dark, cinematic, glass: same tokens as dashboard/style.css
NAVY, PINK, INK, MUTED, GREY = "8FB6EC", "FF3F8E", "F3F4F8", "9298A8", "7D89A8"
INK2, DARK = "C9CCD6", "06070B"
GLASS = "GLASS"  # box(fill=GLASS) draws a translucent panel
rgb = lambda h: RGBColor.from_string(h)  # noqa: E731
GRIDC, EDGE = (1, 1, 1, 0.08), (1, 1, 1, 0.18)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": EDGE,
                     "axes.labelcolor": "#" + INK2, "xtick.color": "#" + INK2, "ytick.color": "#" + INK2,
                     "text.color": "#" + INK, "axes.facecolor": (0, 0, 0, 0), "figure.facecolor": (0, 0, 0, 0),
                     "legend.labelcolor": "#" + INK2, "axes.spines.top": False, "axes.spines.right": False})
HEAT = matplotlib.colors.LinearSegmentedColormap.from_list("heat", ["#1c1e2a", "#5a2140", "#" + PINK])


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return OUT + name


def backdrop(name, strong=False):
    """Slide background: near-black ground, two floodlight pools, faint ground outline, grain."""
    from PIL import Image, ImageDraw, ImageFilter
    W, H = 1920, 1080
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32) + np.array([6, 7, 11], np.float32)
    for cx, cy, r, col, a in ((W * 0.92, -H * 0.05, W * 0.55, (255, 63, 142), 0.30 if strong else 0.20),
                              (-W * 0.05, H * 1.05, W * 0.5, (90, 140, 220), 0.16 if strong else 0.11)):
        f = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2.2) ** 2)))[..., None] * a
        img = img * (1 - f) + np.array(col, np.float32) * f
    rng = np.random.default_rng(7)
    img += rng.normal(0, 3.2, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for rx, ry in ((1250, 300), (640, 150)):
        d.ellipse((W / 2 - rx, H * 0.86 - ry, W / 2 + rx, H * 0.86 + ry), outline=(255, 255, 255, 16), width=3)
    d.rectangle((W / 2 - 22, H * 0.86 - 90, W / 2 + 22, H * 0.86 + 90), outline=(255, 255, 255, 14), width=2)
    ov = ov.filter(ImageFilter.GaussianBlur(0.8))
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    im.save(OUT + name, quality=92)
    return OUT + name


# ---------------------------------------------------------------- figures
def fig_pressure_map():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    for ax, inn in zip(axes, (1, 2)):
        M = np.full((9, 20), np.nan)
        for o, w, v in grid.filter(pl.col("innings") == inn).select("overid", "wkts_before", "share_high").iter_rows():
            if w <= 8:
                M[w, o - 1] = v
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=HEAT, vmin=0, vmax=0.8, extent=(0.5, 20.5, -0.5, 8.5))
        ax.set_title("First innings" if inn == 1 else "Chase", fontsize=13, loc="left", color="#" + INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Over")
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[0].set_ylabel("Wickets down")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Share of balls that are high pressure")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    return save(fig, "slide_fig_pressure_map.png")


def fig_swing():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels, x = ["IPL\n1st innings", "IPL\nchase", "T20I\n1st innings", "T20I\nchase"], np.arange(4)
    keys = [("IPL", 1), ("IPL", 2), ("T20I", 1), ("T20I", 2)]
    o = [val.filter((pl.col("fmt") == f) & (pl.col("innings") == i) & ~pl.col("high_pressure"))["mean_abs_wp_swing"][0] for f, i in keys]
    h = [val.filter((pl.col("fmt") == f) & (pl.col("innings") == i) & pl.col("high_pressure"))["mean_abs_wp_swing"][0] for f, i in keys]
    ax.bar(x - 0.19, np.array(o) * 100, 0.36, color="#" + GREY, label="Ordinary balls")
    ax.bar(x + 0.19, np.array(h) * 100, 0.36, color="#" + PINK, label="High-pressure balls")
    for i, k in enumerate(keys):
        ax.text(x[i] + 0.19, h[i] * 100 + 0.08, f"{ratio[k]:.1f}x", ha="center", fontsize=11, fontweight="bold", color="#" + INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean win-probability swing per ball (%)")
    ax.set_ylim(0, max(h) * 100 * 1.32)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", color=GRIDC)
    ax.set_axisbelow(True)
    return save(fig, "slide_fig_swing.png")


def fig_calibration():
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=(1, 1, 1, 0.35), lw=1, label="Perfect calibration")
    for inn, col, lab in ((1, GREY, "First innings"), (2, PINK, "Chase")):
        c = cal.filter(pl.col("innings") == inn).sort("bin")
        ax.plot(c["predicted"], c["observed"], "-o", color="#" + col, lw=2, ms=6, mec="#" + DARK, mew=1.5,
                label=f"{lab} (error {summ['ece'][str(inn)]:.1%})")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=GRIDC)
    ax.set_axisbelow(True)
    return save(fig, "slide_fig_calibration.png")


def fig_split_half():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ms = [("pressure_raw", "Impact on\npressure balls"), ("all_raw", "Impact on\nall balls"),
          ("uplift_raw", "Clutch uplift\n(pressure - usual)")]
    y = np.arange(len(ms))[::-1]
    b = [hb["split_half"][k]["r_half"] for k, _ in ms]
    w = [hw["split_half"][k]["r_half"] for k, _ in ms]
    ax.barh(y + 0.18, b, 0.34, color="#" + NAVY, label="Batters")
    ax.barh(y - 0.18, w, 0.34, color="#" + PINK, label="Bowlers")
    for yy, v in list(zip(y + 0.18, b)) + list(zip(y - 0.18, w)):
        ax.text(max(v, 0) + 0.005, yy, f"{v:.3f}", va="center", fontsize=10, color="#" + INK)
    ax.axvline(0, color=(1, 1, 1, 0.4), lw=1)
    ax.set_yticks(y, [l for _, l in ms])
    ax.set_xlim(-0.05, 0.3)
    ax.set_xlabel("Correlation between odd- and even-numbered matches (r)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", color=GRIDC)
    ax.set_axisbelow(True)
    return save(fig, "slide_fig_split_half.png")


def fig_top10(role):
    t = allp[role].head(10).reverse()
    fig, ax = plt.subplots(figsize=(5.0, 5.6))
    y = np.arange(t.height)
    ax.hlines(y, t["rating_lo"], t["rating_hi"], color="#" + GREY, lw=6, alpha=0.6, label="20th-80th percentile")
    ax.plot(t["rating_lo"], y, "|", color="#" + NAVY, ms=16, mew=2.5, label="Ranking value (20th pct)")
    ax.plot(t["rating"], y, "o", color="#" + PINK, ms=8, mec="#" + DARK, mew=1.5, label="Rating")
    ax.set_yticks(y, [str(r) for r in t["rank"]], fontsize=11)
    ax.set_ylabel("Rank")
    ax.axvline(0, color=(1, 1, 1, 0.4), lw=1)
    ax.set_xlabel("Pressure rating\n(0 = average player in the same situations)")
    ax.set_xlim(min(0, t["rating_lo"].min()) - 0.005, t["rating_hi"].max() * 1.05)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax.grid(axis="x", color=GRIDC)
    ax.set_axisbelow(True)
    return save(fig, f"slide_fig_top10_{role}.png")


def fig_robust(role):
    r = rob[role].head(10)
    M = np.array([[np.nan if v is None else v for v in r[k].to_list()] for k, _ in VARIANTS]).T
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    C = np.where(np.isnan(M), 0, np.where(M <= 10, 2, np.where(M <= 20, 1, 0)))
    cmap = matplotlib.colors.ListedColormap(["#171924", "#4a2038", "#" + PINK])
    ax.imshow(C, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=9,
                    color="white" if C[i, j] else "#" + INK2, fontweight="bold" if C[i, j] == 2 else "normal")
    ax.set_xticks(range(len(VARIANTS)), [l for _, l in VARIANTS], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(r.height), [f"{m}. {n}  ({a}/{b})" for m, n, a, b in
                                    zip(r["main"], r["name"], r["variants_top10"], r["variants_eligible"])], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(VARIANTS)), minor=True)
    ax.set_yticks(np.arange(-0.5, r.height), minor=True)
    ax.grid(which="minor", color="#" + DARK, lw=2.5)
    return save(fig, f"slide_fig_robust_{role}.png")


F = {"bg": backdrop("slide_bg.jpg"), "bg_title": backdrop("slide_bg_title.jpg", strong=True),
     "map": fig_pressure_map(), "swing": fig_swing(), "cal": fig_calibration(), "sh": fig_split_half(),
     "t_bat": fig_top10("batters"), "t_bowl": fig_top10("bowlers"),
     "r_bat": fig_robust("batters"), "r_bowl": fig_robust("bowlers")}

# ---------------------------------------------------------------- deck helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height
FOOT = "SupeRR Selector S2 · Round 1 · Pressure ratings"


def _alpha(parent, a):
    """Add <a:alpha> to the first srgbClr under parent (a = 0..1 opacity)."""
    clr = parent.find(".//" + qn("a:srgbClr"))
    el = clr.makeelement(qn("a:alpha"), {"val": str(int(a * 100000))})
    clr.append(el)


def box(slide, x, y, w, h, fill=None, line=None, shape=None, alpha=None, line_alpha=0.16):
    """fill=GLASS: translucent white panel with a light edge, rounded. Otherwise a flat colour (optional alpha)."""
    glass = fill == GLASS
    if shape is None:
        shape = MSO_SHAPE.ROUNDED_RECTANGLE if glass else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, Inches(0.22) / min(w, h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb("FFFFFF" if glass else fill)
        if glass or alpha is not None:
            _alpha(s._element.spPr.find(qn("a:solidFill")), 0.065 if glass else alpha)
    else:
        s.fill.background()
    if glass or line:
        s.line.color.rgb = rgb("FFFFFF" if glass else line)
        s.line.width = Pt(0.9)
        _alpha(s._element.spPr.find(qn("a:ln")), line_alpha)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def tb(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    """paras: list of (text, size, bold, color) or (text, size, bold, color, bullet)."""
    t = slide.shapes.add_textbox(x, y, w, h)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        text, size, bold, color = p[:4]
        bullet = len(p) > 4 and p[4]
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(size * 0.45)
        r = para.add_run()
        r.text = ("•  " if bullet else "") + text
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
        r.font.color.rgb = rgb(color)
    return t


def slide(title, kicker, notes=""):
    s = prs.slides.add_slide(BLANK)
    s.shapes.add_picture(F["bg"], 0, 0, width=SW, height=SH)
    tb(s, Inches(0.6), Inches(0.36), Inches(12), Inches(0.35), [(kicker, 13, True, PINK)])
    tb(s, Inches(0.6), Inches(0.68), Inches(12.2), Inches(0.9), [(title, 28, True, INK)])
    tb(s, Inches(0.6), Inches(7.02), Inches(8), Inches(0.3), [(FOOT, 10, False, MUTED)])
    tb(s, Inches(11.7), Inches(7.02), Inches(1.1), Inches(0.3), [(str(len(prs.slides)), 10, False, MUTED)], align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(path, x, y, width=w, height=h)


def stat(s, x, y, w, value, label, color=PINK):
    box(s, x, y, w, Inches(1.25), fill=GLASS)
    tb(s, x + Inches(0.18), y + Inches(0.1), w - Inches(0.3), Inches(0.55), [(value, 26, True, color)])
    tb(s, x + Inches(0.18), y + Inches(0.66), w - Inches(0.3), Inches(0.55), [(label, 11.5, False, INK)])


def table(s, x, y, w, headers, rows, widths, row_h=0.36, font=11, hi_rows=()):
    shp = s.shapes.add_table(len(rows) + 1, len(headers), x, y, w, Inches(row_h * (len(rows) + 1)))
    t = shp.table
    t._tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # No Style, No Grid
    tot = sum(widths)
    for j, wd in enumerate(widths):
        t.columns[j].width = Emu(int(w * wd / tot))
    for i in range(len(rows) + 1):
        t.rows[i].height = Inches(row_h)
        for j in range(len(headers)):
            c = t.cell(i, j)
            c.text = str(headers[j] if i == 0 else rows[i - 1][j])
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            hi = i > 0 and i - 1 in hi_rows
            c.fill.fore_color.rgb = rgb(PINK if hi else "FFFFFF")
            _alpha(c._tc.get_or_add_tcPr(), 0.22 if hi else (0.13 if i == 0 else (0.06 if i % 2 else 0.025)))
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 1 else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size, r.font.name = Pt(font), "Calibri"
                    r.font.bold = i == 0 or j == 1
                    r.font.color.rgb = rgb(INK)
    return t


# ---------------------------------------------------------------- slides
# 1 title
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(F["bg_title"], 0, 0, width=SW, height=SH)
box(s, Inches(0.6), Inches(6.55), Inches(12.13), Inches(0.62), fill=GLASS)
tb(s, Inches(0.8), Inches(1.0), Inches(11), Inches(0.4),
   [("SupeRR Selector Season 2 · Round 1", 16, True, PINK)])
tb(s, Inches(0.8), Inches(1.6), Inches(11.5), Inches(2.4), [("Who delivers under pressure?", 60, True, "FFFFFF")])
tb(s, Inches(0.8), Inches(3.75), Inches(11), Inches(1.2),
   [(f"A pressure rating for IPL and T20I batters and bowlers, built from {summ['scope_balls']:,} deliveries "
     f"in {summ['scope_matches']} matches ({summ['years'][0]}-{summ['years'][-1]}).", 20, False, "C9CDD8")])
tb(s, Inches(0.8), Inches(5.2), Inches(11), Inches(0.9),
   [("Pressure defined as stakes · credit against the same situation · clutch tested, not assumed · "
     "cautious ranking · 15 robustness checks", 15, False, "FFFFFF")])
tb(s, Inches(0.85), Inches(6.55), Inches(11), Inches(0.62), [("Methodology summary and final top-10 lists", 14, True, "FFFFFF")],
   anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = ("Our entry for Round 1: two ordered top-10 lists of the most reliable performers under "
                                       "pressure, and the evidence behind them. Every number comes from the supplied data pack.")

# 2 approach
s = slide("Four steps, one currency: win probability", "The approach",
          "Everything is measured in win probability, so runs and wickets, batting and bowling, IPL and T20I are on one scale.")
steps = [("1", "Define pressure", "A win-probability model gives every ball its stakes: how far the result can swing on it. "
          "The top 20% within format and innings are high pressure."),
         ("2", "Measure performance", "Each ball's change in win probability is credited to batter and bowler, against the "
          "average player in the same situation."),
         ("3", "Test clutch", "Does doing better under pressure than usual repeat across matches? We tested it before "
          "deciding how to rate."),
         ("4", "Rank cautiously", "Shrink small samples toward the average and rank on the 20th percentile of 1,000 match "
          "bootstraps. Re-test every pick 15 ways.")]
for i, (n, h, d) in enumerate(steps):
    x = Inches(0.6 + i * 3.1)
    box(s, x, Inches(1.85), Inches(2.85), Inches(3.75), fill=GLASS)
    tb(s, x + Inches(0.25), Inches(2.05), Inches(1), Inches(0.8), [(n, 40, True, PINK)])
    tb(s, x + Inches(0.25), Inches(2.95), Inches(2.4), Inches(0.6), [(h, 18, True, INK)])
    tb(s, x + Inches(0.25), Inches(3.6), Inches(2.4), Inches(1.9), [(d, 13.5, False, INK2)])
box(s, Inches(0.6), Inches(5.85), Inches(12.15), Inches(0.85), fill=PINK, alpha=0.2, line=PINK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_alpha=0.45)
tb(s, Inches(0.85), Inches(5.85), Inches(11.7), Inches(0.85),
   [("Why win probability? Runs and wickets, batting and bowling, IPL and T20I all land on one scale: "
     "the chance of winning the match.", 15, True, "FFFFFF")], anchor=MSO_ANCHOR.MIDDLE)

# 3 pressure definition
s = slide("Pressure = what is at stake on this ball", "1 · Defining pressure",
          "Stakes capture required rate, wickets in hand, balls left, phase and closeness all at once, because they all move win probability.")
tb(s, Inches(0.6), Inches(1.6), Inches(12.2), Inches(1.0),
   [("Stakes (leverage) = expected absolute change in the batting side's win probability over the usual mix of outcomes "
     "(dot, 1, 2, 4, 6, wicket). A ball is high pressure if its stakes are in the top 20% of its format and innings.",
     14, False, INK2)])
pic(s, F["map"], Inches(0.6), Inches(2.55), w=Inches(9.0))
tb(s, Inches(9.85), Inches(2.75), Inches(3.0), Inches(4.0),
   [("Chase", 15, True, PINK), ("Peaks in overs 11-20 of a close chase with 3-6 wickets down; vanishes once a chase is decided.", 12.5, False, INK),
    ("First innings", 15, True, PINK), ("Early and middle-overs collapses and the death. An early wicket costs the most projected runs, as in Duckworth-Lewis.", 12.5, False, INK),
    ("Why not across innings?", 15, True, PINK), ("Ranking both innings together put 98.7% of pressure balls in chases, so batters who bat first could not be assessed.", 12.5, False, INK)])

# 4 evidence for definition
s = slide("The definition separates pressure from ordinary play", "1 · Does it work?",
          "Three checks: pressure balls move the match more; the model is calibrated; dead matches are excluded.")
stat(s, Inches(0.6), Inches(1.75), Inches(3.0), f"{HP_MOVE:.0%}", f"of all win-probability movement happens on the {HP_BALLS:.0%} of balls rated high pressure")
stat(s, Inches(0.6), Inches(3.15), Inches(3.0), chase_rng, "how much more a high-pressure ball moves a chase than an ordinary ball")
stat(s, Inches(0.6), Inches(4.55), Inches(3.0), f"{dec['model_high_share_decided']:.2%}",
     f"of high-pressure balls come from already-decided matches (base rate {dec['all_balls_share_decided']:.0%}; "
     f"a rule-based difficulty index: {dec['rule_high_share_decided']:.0%})")
pic(s, F["swing"], Inches(3.85), Inches(1.75), w=Inches(4.75))
pic(s, F["cal"], Inches(8.85), Inches(1.75), w=Inches(3.95))
tb(s, Inches(3.85), Inches(6.1), Inches(9), Inches(0.8),
   [(f"Model: logistic regression, fitted leaving each season out. Held-out log loss {mc[('logit', 1)]['log_loss']:.3f} vs "
     f"{mc[('logit', 1)]['log_loss_base']:.3f} with no information (1st innings), {mc[('logit', 2)]['log_loss']:.3f} vs "
     f"{mc[('logit', 2)]['log_loss_base']:.3f} (chase). It beat spline and gradient-boosted alternatives. "
     f"First-innings pressure is the weaker half of the evidence (swing about {ratio[('IPL', 1)]:.1f}x).", 11.5, False, MUTED)])

# 5 measuring performance
s = slide("Credit every ball against the average player in the same spot", "2 · Measuring performance",
          "The same-situation baseline matters: the model alone over-credited middle-overs bowling and under-credited chase powerplays.")
pts = [("Credit per ball", "Batter: the win-probability effect of the runs they hit. Dismissed player (striker or non-striker): "
        "the cost of the wicket. Bowler: runs off the bat, wides and no-balls conceded, wickets taken (never run-outs)."),
       ("Same-situation baseline", "Each credit is compared with the average in the same format, season, innings, phase and "
        "wickets-down band, so nobody gains from batting in easy phases or bowling to weak sides."),
       ("Impact = credit ÷ stakes", "Performance relative to what was at stake. +0.10 means the player moved win probability "
        "in their team's favour by 10% of the stakes, more than an average player would have."),
       ("Usual vs pressure", "Usual level = impact on ordinary balls. Uplift = impact on high-pressure balls minus usual level "
        "(the brief's 'compared with their usual performance').")]
for i, (h, d) in enumerate(pts):
    x, y = Inches(0.6 + (i % 2) * 6.2), Inches(1.75 + (i // 2) * 2.55)
    box(s, x, y, Inches(5.95), Inches(2.3), fill=GLASS)
    tb(s, x + Inches(0.3), y + Inches(0.2), Inches(5.4), Inches(0.5), [(h, 18, True, PINK)])
    tb(s, x + Inches(0.3), y + Inches(0.75), Inches(5.4), Inches(1.5), [(d, 14, False, INK)])

# 6 clutch
s = slide("Players' level repeats; extra lift under pressure does not", "3 · Is clutch a skill?",
          "Split-half test: compare each player's numbers in odd- and even-numbered matches.")
pic(s, F["sh"], Inches(0.6), Inches(1.75), w=Inches(7.6))
tb(s, Inches(8.55), Inches(1.8), Inches(4.3), Inches(5.0),
   [("What we found", 16, True, PINK),
    (f"Uplift does not repeat: r = {hb['split_half']['uplift_raw']['r_half']:.3f} for batters, "
     f"{hw['split_half']['uplift_raw']['r_half']:.3f} for bowlers ({hb['split_half']['uplift_raw']['players']} and "
     f"{hw['split_half']['uplift_raw']['players']} players).", 13, False, INK, True),
    ("Large clutch effects are ruled out; small ones cannot be detected with this much data.", 13, False, INK, True),
    ("What we did about it", 16, True, PINK),
    ("Rate pressure performance directly from pressure balls, and borrow from ordinary balls only as far as clutch "
     "could not distort them.", 13, False, INK, True),
    (f"Result: pressure balls carry {hb['main']['mean_weight_on_pressure_balls']:.0%} of a batter's own-data weight and "
     f"{hw['main']['mean_weight_on_pressure_balls']:.0%} of a bowler's.", 13, False, INK, True)])

# 7 rating & fair comparison
s = slide("A cautious ranking that rewards a long, reliable record", "4 · Rating and fair comparisons",
          "Ranking on the 20th percentile means a player needs both a high level and enough evidence.")
L = [("Rating", f"Posterior estimate of pressure performance, shrunk toward the eligible average in proportion to its noise "
      f"(empirical Bayes, DerSimonian-Laird). Noise from {rel['settings']['bootstrap_main']:,} match bootstraps."),
     ("Ranking", "20th percentile of each player's rating: a short hot streak cannot outrank a long reliable record. "
      "Each player also gets P(top 10), the share of bootstrap draws in which they rank in the top 10."),
     ("Eligibility", f"At least 90 high-pressure balls across 12+ innings, and 60 ordinary balls: "
      f"{hb['main']['eligible']} batters and {hw['main']['eligible']} bowlers.")]
R = [("IPL and T20I", "One currency for both; par, baseline and model terms per format and season. IPL-only and T20I-only "
      "lists are robustness variants."),
     ("Opposition", "Pre-match team strength (Elo) is in the win-probability model; a variant drops lopsided matches."),
     ("Scope", f"{summ['scope_matches']} matches. Excluded: two A-team tournaments (not IPL/T20I), rain-revised matches and "
      "11 chases whose ball data disagreed with the official margin.")]
for col, items in ((0, L), (1, R)):
    y = 1.75
    for h, d in items:
        tb(s, Inches(0.6 + col * 6.3), Inches(y), Inches(5.9), Inches(1.6), [(h, 17, True, PINK), (d, 13.5, False, INK)])
        y += 1.72


# 8/9 top 10 slides
def top_slide(role):
    bat = role == "batters"
    t = allp[role].head(10).join(top[role].select("pid", "variants_top10", "variants_eligible"), on="pid").sort("rank")
    s = slide(f"Top 10 {role}", "The lists",
              "Ranks 8-10 are uncertain; the order is the cautious ranking fixed in advance, not adjusted by hand.")
    if bat:
        hdr = ["#", "Batter", "Rating", "P(top 10)", "Press. balls", "SR press / ord", "Avg press / ord", "Variants"]
        rows = [(r["rank"], r["name"], f"{r['rating']:+.3f}", f"{r['p_top10']:.0%}", r["b_hi"],
                 f"{r['sr_hp']:.0f} / {r['sr_ord']:.0f}", f"{r['avg_hp']:.1f} / {r['avg_ord']:.1f}",
                 f"{r['variants_top10']}/{r['variants_eligible']}") for r in t.iter_rows(named=True)]
    else:
        hdr = ["#", "Bowler", "Rating", "P(top 10)", "Press. balls", "Econ press / ord", "Balls/wkt press / ord", "Variants"]
        rows = [(r["rank"], r["name"], f"{r['rating']:+.3f}", f"{r['p_top10']:.0%}", r["b_hi"],
                 f"{r['econ_hp']:.2f} / {r['econ_ord']:.2f}", f"{r['balls_per_wkt_hp']:.0f} / {r['balls_per_wkt_ord']:.0f}",
                 f"{r['variants_top10']}/{r['variants_eligible']}") for r in t.iter_rows(named=True)]
    weak = [i for i, r in enumerate(t.iter_rows(named=True)) if r["variants_top10"] * 2 < r["variants_eligible"]]
    table(s, Inches(0.6), Inches(1.65), Inches(7.6), hdr, rows, [0.4, 2.3, 0.9, 0.9, 0.9, 1.3, 1.5, 0.9],
          row_h=0.42, font=11.5, hi_rows=weak)
    pic(s, F["t_bat" if bat else "t_bowl"], Inches(8.5), Inches(1.55), h=Inches(4.75))
    nxt = allp[role].slice(10, 3)
    msg = ("Shaded: top 10 in fewer than half of the robustness variants. " if weak else "") + \
          "Next in order: " + ", ".join(f"{r['rank']}. {r['name']}" for r in nxt.iter_rows(named=True)) + "."
    tb(s, Inches(0.6), Inches(6.4), Inches(12.2), Inches(0.5), [(msg, 11.5, False, MUTED)])


top_slide("batters")
top_slide("bowlers")

# 10/11 robustness
for role in ("batters", "bowlers"):
    t = top[role]
    full = int((t["variants_top10"] == t["variants_eligible"]).sum())
    most = int((t["variants_top10"] * 2 >= t["variants_eligible"]).sum())
    s = slide(f"Robustness: the top 10 {role} under 15 alternative methods", "Evidence",
              "Each column changes one modelling choice. Bright pink = top 10 in that variant; dim pink = 11-20; dash = not eligible.")
    pic(s, F["r_bat" if role == "batters" else "r_bowl"], Inches(0.6), Inches(1.6), w=Inches(12.1))
    tb(s, Inches(0.6), Inches(6.35), Inches(12.2), Inches(0.6),
       [(f"{most} of 10 stay in the top 10 in at least half of the variants where they are eligible; {full} in every one. "
         "Bright pink = top 10, dim pink = 11-20, dash = not eligible in that variant.", 12.5, False, INK)])

# 12 limitations
p810 = [x for r in ("batters", "bowlers") for x in allp[r].filter(pl.col("rank").is_between(8, 10))["p_top10"]]
s = slide("Reliability and limitations", "Honest caveats",
          "We report the weak spots rather than hide them.")
tb(s, Inches(0.6), Inches(1.7), Inches(6.0), Inches(5.0),
   [("Reliability", 17, True, PINK),
    (f"Pressure performance is noisy in T20: split-half r = {hb['split_half']['pressure_raw']['r_half']:.2f} (batters), "
     f"{hw['split_half']['pressure_raw']['r_half']:.2f} (bowlers). Hence the cautious ranking and a P(top 10) for every pick.", 13.5, False, INK, True),
    (f"Ranks 8-10 are uncertain (P(top 10) {min(p810):.2f}-{max(p810):.2f}).", 13.5, False, INK, True),
    ("The method was fixed before the final lists were read and was not adjusted to favour any name.", 13.5, False, INK, True)])
tb(s, Inches(6.9), Inches(1.7), Inches(6.0), Inches(5.0),
   [("Limitations", 17, True, PINK),
    ("Extras run amounts are not recorded; chase targets are back-solved from official results.", 13.5, False, INK, True),
    ("Elo ratings and season par use other matches of the same season.", 13.5, False, INK, True),
    ("First-innings pressure is weaker evidence than chase pressure (a chase-only variant is included).", 13.5, False, INK, True),
    ("Fielding, captaincy and batter-bowler match-ups are not modelled.", 13.5, False, INK, True)])

# 13 reproduce
s = slide("Reproducible end to end", "Transparency",
          "Anyone with the data pack can regenerate the lists, the report, this deck and the dashboard.")
tb(s, Inches(0.6), Inches(1.75), Inches(12), Inches(3.4),
   [("From the project root, with the supplied data pack in place (Python 3.11, packages pinned in requirements.txt):", 14, False, INK),
    ("python -P pipeline/build_state.py   →  ball-by-ball match state", 13, False, NAVY),
    ("python -P pipeline/pressure_model.py  →  win probability, stakes, validation", 13, False, NAVY),
    ("python -P pipeline/ratings.py   →  ratings, top-10 lists, robustness, reliability", 13, False, NAVY),
    ("python -P pipeline/report.py · slides.py · export_dashboard.py  →  report, deck, dashboard", 13, False, NAVY),
    ("Each script runs built-in checks and prints OK. Only the supplied data is used; no manual overrides.", 14, False, INK)])
links = [("Code and outputs", REPO_URL)] + ([("Interactive dashboard", DASHBOARD_URL)] if DASHBOARD_URL else [])
y = 4.3
for h, u in links:
    tb(s, Inches(0.6), Inches(y), Inches(12), Inches(0.6), [(h, 14, True, PINK), (u, 14, False, INK)])
    y += 0.85

out = OUT + "SupeRR_Selector_Pressure_Ratings.pptx"
prs.save(out)
print("wrote", out, f"({len(prs.slides)} slides)")
print("OK")
