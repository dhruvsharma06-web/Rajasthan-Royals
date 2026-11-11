# SupeRR Selector S2, Round 1 — Pressure Ratings

Single project file (solo project): brief, requirements, decisions, tasks, status.
Source of requirements: `6af3166e7b922_data_pack/Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf`.

## Brief
- **Owner / acceptance:** solo entrant (also deploys the dashboard to Vercel).
- **Deadline:** 18 Nov 2026. Timezone not stated on the brief; assumed IST end of day (A1).
- **Outcome:** two ordered top-10 lists (batters, bowlers) plus a methodology report, submitted on Unstop.
- **Extra (not required by the brief):** a web dashboard showing the results, linked from the report.
- **Boundaries:** no commit, push, publish or deploy without the owner's go-ahead. No paid services.

## Requirements
| ID | Exact wording | Priority | Check |
|---|---|---|---|
| R1 | "Build a data-driven system that measures how batters and bowlers perform in high-pressure match situations compared with their usual performance levels." | MUST | Pipeline outputs pressure vs baseline numbers per player |
| R2 | "Choose your own approach, explain why your definition is meaningful, and show how it distinguishes pressure situations from ordinary ones." | MUST | Report section + pressure-vs-ordinary comparison table |
| R3 | "Your method should work for both batting and bowling performance." | MUST | Same pressure definition feeds both ratings |
| R4 | "Two ordered top-10 lists ... Select players in rank order, from 1 to 10" / "Do not repeat a player within the same top-10 list" | MUST | Output has 10 unique names per list |
| R5 | "ensure the dropdown selections match the rankings explained in your report" | MUST | Report tables generated from the same output file |
| R6 | Report covers: pressure definition; performance measurement; fair comparisons (sample sizes, eligibility, match contexts, IPL and T20I); evidence for top 10; reliability and limitations | MUST | Report checklist |
| R7 | "transparent enough for someone else to understand and reproduce your approach" | MUST | Fresh run of the pipeline reproduces the lists |
| R8 | "Rankings must be supported by analysis of the supplied data, not reputation alone." | MUST | No external data or manual overrides |
| R9 | Optional journey photos / 30–60 s video ("will not affect the evaluation") | OPTIONAL | — |

## Assumptions and unknowns
- A1 Deadline is end of day IST. Unverified.
- A2 Judging criteria and report page limit are not published; the report follows the five bullets in R6.
- A3 RESOLVED (audit, 11 Nov): `match_format = T20` is IPL (367 matches with balls) plus two A-team tournaments, Emerging Teams Asia Cup 2024 and Asia Cup Rising Stars 2025 (30 matches). These are labelled `fmt = OTHER`. Excluded from player ratings because the brief covers IPL and T20I only (owner confirmed 11 Nov).
- A4 Player names in the Unstop dropdowns match `striker_name` / `bowler_name`. To check on the submission page.

## Data findings (measured)
- 200,999 deliveries, 867 matches with balls (16 abandoned matches have none), 22 Jan 2022 – 1 Nov 2026.
- `extras1_runs` is a type code, not runs. Bye, leg-bye and extra-wide run amounts are not recorded.
- `extras_type = other` deliveries do not count toward the over (treating them as legal gave 121-ball innings).
- `first_inn_revised_overs` / `second_inn_revised_overs` hold balls, not overs.
- Result strings are free text ("Won By 13Runs", "Won 3 Wkts", "won the super over").
- Innings 3 and 4 are super overs (38 balls); excluded.
- `bowl_speed` is 0 on 39% of balls; not used.

## Decisions
- **D1 Chase targets are back-solved from the official result**, not from the rebuilt first-innings total.
  Evidence (`spikes/score_rebuild.py`): rebuilt totals miss the official run margin by 2.5 runs on average
  (held-out half, n=201; 30% exact). Back-solving makes the error in runs-needed shrink toward the end of a chase.
- **D2 Team runs per ball = bat runs + 1 per extras event** (+ `extras_2_runs`, 5 for a penalty). Fitted per-type
  values were no better than this on held-out matches.
- **D3 Chases where the official margin and ball data disagree by more than 8 runs are flagged `target_suspect`**
  (11 matches) and kept out of model fitting.
- **D5 Pressure = leverage (stakes)**, ranked within format x innings; high pressure = top 20%.
  Leverage = expected absolute change in win probability over one outcome mix per format x innings
  (a per-phase mix created steps at overs 7 and 16). Ranking across both innings was tried and rejected:
  98.7% of high-pressure balls fell in chases, so batters who bat first could not be assessed. It is kept
  as a sensitivity variant (`pressure_pct_pooled`). Evidence in `outputs/t3/`.
- **D6 Win-probability model = plain logistic regression** on Duckworth-Lewis-style features (projected
  total vs par; runs-needed deficit vs expected runs to come, scaled by remaining balls), fitted
  leave-one-season-out. Beat spline logistic and gradient boosting on held-out log loss in both innings,
  and cannot bend the wrong way. The spline version gave 19% to "need 97 off 1 ball". Expected runs to come is
  a smooth DL curve fitted to the data; a tree version was step-shaped and produced spurious leverage.
  The first-innings model also values the end of the innings (switching to the chase model at the break
  made a last-ball six lose up to 0.30). Chases needing more than 6 per remaining ball are capped at 0.2%.
- **D7 Rating uses all balls, stake-weighted; uplift counts only as far as it is real.** Improvement under
  pressure does not repeat across matches (split-half r = 0.01 batters, -0.03 bowlers), and the shrinkage
  estimator gives it zero weight. Ordinary balls alone were also unreliable for batters (r near 0) because the
  runs part (r 0.53) and the wicket part (r 0.53) cancel: fast scorers get out more. So the estimate of pressure
  performance is the stake-weighted impact on all balls (a high-pressure ball counts 2-4x), plus shrunk uplift.
  Shrinkage uses the DerSimonian-Laird estimator; the plain moment estimator collapsed to 0 in small-sample
  variants and made the order arbitrary. Ranked on the 20th percentile of 1,000 match bootstraps.
- **D8 Credit is measured against the average player in the same situation** (format x innings x phase x
  wickets-down band). The win-probability model is calibrated overall but not by phase: bowlers averaged -0.043
  per unit stake in chase powerplays and +0.02 in middle overs. Pace vs spin is not adjusted in the main
  version; it is a robustness variant.
- **D9 Rating = posterior estimate of pressure performance.** Pressure balls measure it directly. Ordinary
  balls measure it minus the player's clutch uplift, so they count as a noisier measurement, discounted by the
  largest clutch spread the split-half test cannot rule out (90% bound, by simulation). This gives pressure balls
  as much weight as the data allows (on average 55% of a batter's own-data weight, 41% of a bowler's). Ranked on the
  20th percentile of posterior draws. Supersedes the all-balls rating of D7, which is kept as a variant.
- **D4 Stack:** Python + polars for analysis; static Vite/React dashboard reading exported JSON; no backend.

## Method v2 (revised 11 Nov 2026; as built, superseded where decisions D5-D8 differ)
Weaknesses found in v1 and the fix for each:

| v1 weakness | v2 fix |
|---|---|
| Win-probability swings are largest late in chases, so first-innings batters and bowlers would rarely count as "under pressure" | First innings uses a par-score model (expected final total from this state); pressure = how much the final total can move relative to par. Both innings are rescaled to the same 0–100 pressure percentile within format |
| Runs and wickets are two currencies; strike-rate vs dismissal trade-off unhandled | One currency: **win probability added (WPA)** per ball, credited to striker and bowler. Runs and wickets above expected kept as readable supporting metrics |
| Ranking on uplift vs own baseline alone favours average players with lucky streaks and punishes players who are excellent everywhere | Rank on **pressure performance level**; report uplift vs own baseline for every player (R1) and use it as evidence, not the sort key |
| Small samples dominate the top of any per-ball table | Rank on a **conservative estimate**: the 20th percentile of the match-bootstrap distribution of each player's shrunk pressure impact. Rewards quality, sample size and consistency together, with no invented weights |
| T20I opponents range from full members to A-teams and associates | Opposition-strength adjustment from team ratings fitted on results; sensitivity run restricted to full-member and IPL matches |
| "Clutch" may not be a repeatable skill | Test it: split-half (odd/even matches) correlation of pressure uplift. Report the answer whichever way it goes; this is the main analytical finding |
| A definition judged only on our own model | Pressure validated three ways: (a) higher pressure balls change the match result more often, (b) league-wide dismissal rate and scoring shift under pressure, (c) rule-based index ranks the same balls as high pressure |

Steps:
1. Win-probability model (2nd innings) and par-score model (1st innings); calibration on held-out seasons.
2. Pressure per ball = outcome-swing (wicket vs boundary) on win probability or par; percentile within format and innings. High pressure = top 20%; threshold varied 10–30% in sensitivity.
3. Per-ball credit: WPA plus runs/wicket above the average player in the same state (phase, wickets down, format, pace or spin).
4. Player estimates: empirical-Bayes shrinkage toward the role average; opposition adjustment.
5. Eligibility: batters at least 120 high-pressure balls faced across at least 15 innings; bowlers at least 120 high-pressure balls across at least 15 matches. Thresholds checked for sensitivity.
6. Rank by the bootstrap 20th percentile. Report each top-10 player's probability of being top 10, uplift vs own baseline, match-level consistency, format split and season trend.
7. Robustness table: the lists under alternative thresholds, IPL-only, T20I-only, no opposition adjustment, rule-based pressure.

## Audit of 11 Nov 2026 (open items feed T3-T8)
Fixed in code: `fmt` mislabelled 30 A-team matches as IPL; `runs_last12` / `wkts_last12` covered 11 balls, not 12.

Rules the remaining tasks must follow:
- F1 The 9 tied matches carry the super-over winner as `win_team_id`. For model fitting a tie is not a win or a loss: exclude or weight 0.5.
- F2 Fit models only on chases with `target_src` in margin_runs / margin_wkts / tie and `target_suspect` false. Rain-revised and rebuilt chases (31 matches) are not validated.
- F3 One currency in both innings: a single win-probability model. First innings uses score relative to par for that format and season. The v2 table's separate "par" pressure is only an input to it.
- F4 Baselines must be per format and season. IPL scoring changed sharply across 2022-2026 (impact-player rule from 2023).
- F5 Wicket cost goes to `dismissed_player_id`, not the striker: 270 run-outs dismissed the non-striker. Run-outs are never credited to the bowler.
- F6 Bowler credit uses `bowler_runs` (no byes or leg-byes), not `team_runs`.
- F7 Players are keyed by id. Three names map to two ids each (Haider Ali, Abdul Samad, Basil Hameed).
- F8 `knock_out` flags only 5 T20I matches and 226 matches have stage "Others"; it is not reliable enough to use as a pressure input.
- F9 Eligibility thresholds are provisional. Only about 100 batters and 100 bowlers have 600+ balls; set the cut after seeing the high-pressure ball distribution, then test sensitivity.
- F10 29 eligible-size batters are IPL-only and 42 are T20I-only, so cross-format comparison rests on the within-format standardisation and opposition adjustment; the robustness table must show IPL-only and T20I-only lists.
- F11 The bootstrap 20th percentile is a choice, not a derived value; show 10th and 50th as sensitivity.
- F12 Reproducibility is missing: no pinned requirements file, no version control, no tests on the state table beyond the built-in checks.
- G1 Ranking metric. Raw WPA per ball is dominated by the few last-over balls (a six in the 20th over can be worth 20x one in the 16th), so rankings would be noise. Rank on leverage-normalised impact: each ball's WPA divided by its leverage. Raw WPA is reported, not ranked on.
- G2 Pressure is partly self-made. A batter who scores slowly raises the required rate and then "faces pressure" he caused. Sensitivity: measure pressure at the start of the over and at the batter's arrival, and confirm the batting list holds.
- G3 Pressure level vs uplift. Rank on pressure level, but add the player's own uplift shrunk by its measured split-half reliability r (true uplift is estimated as r x observed uplift). The data decides how much "rises to the occasion" counts: if clutch is not repeatable, r is near 0 and it barely moves the list. This replaces "uplift is evidence only".
- G4 No leakage from the win-probability model. Fit it leaving one season out and score each season with the model that did not see it.
- G5 Team strength and scoring era belong in the win-probability model (team rating difference; format and season), otherwise India chasing 150 against an associate side looks like pressure.
- G6 SUPERSEDED by D5. Ranking within each innings made the first ball of a first innings look like pressure (57% of first-over balls flagged). Kept only as a sensitivity variant (`pressure_pct_by_innings`).
- G7 Consistency is measured directly (R6): per-match pressure impact, share of positive pressure matches, and how many matches account for half of a player's total. A list built on one or two innings is flagged.
- G8 Recency is a sensitivity, not a weight: lists recomputed on 2024-2026 only, to show whether a pick rests on old form.
- G9 The Unstop dropdown may not list every player in the data. Check the player pool before 14 Nov; an ineligible pick cannot be submitted.
- G10 Cut order if time slips: dashboard animations, then the dashboard itself, then G8, then G2. The lists, the report and the reliability evidence are never cut. Lists frozen 15 Nov.
- F13 The project folder is inside OneDrive. A dashboard `node_modules` folder there will sync tens of thousands of files and can lock files mid-build; build the dashboard outside OneDrive or exclude the folder from sync.

## Tasks
| ID | Task | Reqs | State | Evidence |
|---|---|---|---|---|
| T1 | Score-rebuild spike | R1 | VERIFIED | `python -P spikes/score_rebuild.py` |
| T2 | Ball-state table | R1, R7 | VERIFIED | `python -P pipeline/build_state.py` prints OK |
| T3 | Win-probability model, pressure per ball, validation of the definition | R2, R3 | VERIFIED 11 Nov | `python -P pipeline/pressure_model.py` prints OK; results below |
| T4 | Credit split, same-situation baseline, shrinkage | R1, R3, R6 | VERIFIED 11 Nov | `python -P pipeline/ratings.py` prints OK |
| T5 | Bootstrap ranking, clutch-repeatability test, robustness table, provisional top 10s | R4, R6, R8 | VERIFIED 11 Nov | `outputs/ratings/` |
| T5b | Independent method review and fixes | R6 | VERIFIED 11 Nov | review section above |
| T6 | Dashboard reading exported JSON | extra | PAUSED (usage budget) | — |
| T7 | Methodology report (DOCX), generated from outputs | R5, R6, R7 | DRAFT 11 Nov | `python -P pipeline/report.py` -> outputs/report/ |
| T8 | Clean-run reproduction, name check vs dropdowns, owner deploys | R4, R5, R7 | BLOCKED on T7 (17 Nov) | — |
| — | Buffer; submit early | — | 18 Nov | — |

## T3 results (11 Nov 2026, held-out seasons, 774 matches / 184,940 balls; excludes A-team, rain-affected and flagged matches)
- Win-probability model, log loss vs no-information baseline: innings 1 0.626 vs 0.694; innings 2 0.397 vs 0.693.
  Calibration error 2.1% (innings 1) and 1.0% (innings 2).
- High-pressure balls (top 20%) carry 39% of all win-probability movement. Mean absolute swing per ball is
  about 4x ordinary balls in chases and 1.3-1.4x in first innings.
- Chases: peak in overs 11-20 of a close chase with 3-6 wickets down. First innings: powerplay and middle-overs
  collapses, the death, and the first over (early wickets cost the most projected runs, as in DLS).
- Only 0.15% of high-pressure balls come from already-decided matches (win probability <5% or >95%).
  A rule-based "difficulty" index flags 20%, worse than the 14% base rate, so it is reported as a contrast,
  not a validation.
- Checks built into the script: no six costs the batter and no wicket helps the batting side (0 of 11,816 and
  0 of 9,931).

## T4/T5 results (11 Nov 2026, after review fixes) — D9 rating
Eligibility: at least 90 high-pressure balls in at least 12 innings, and 60 ordinary balls
(115 batters, 112 bowlers). Robustness = 15 variants: top 10% / top 30% threshold, pressure ranked across
innings, pressure at the start of the over, 10th / 50th percentile, IPL only, T20I only, 2024-2026,
no mismatches (|Elo gap| <= 150), pace/spin baseline, no situation baseline, all balls with no clutch
(old D7), pressure balls only, chase pressure only.

| Rank | Batter | P(top 10) | Top 10 in variants | Bowler | P(top 10) | Top 10 in variants |
|---|---|---|---|---|---|---|
| 1 | Heinrich Klaasen | 0.84 | 14/14 | Mujeeb Ur Rahman | 0.76 | 14/14 |
| 2 | Vaibhav Sooryavanshi | 0.67 | 12/13 | Sunil Narine | 0.56 | 14/14 |
| 3 | Tim David | 0.60 | 15/15 | Rashid Khan | 0.46 | 15/15 |
| 4 | Rinku Singh | 0.37 | 12/15 | Lockie Ferguson | 0.51 | 11/15 |
| 5 | Glenn Phillips | 0.26 | 12/13 | Varun Chakravarthy | 0.37 | 14/15 |
| 6 | Shreyas Iyer | 0.31 | 13/15 | Josh Hazlewood | 0.44 | 15/15 |
| 7 | Abhishek Sharma | 0.28 | 11/14 | Krunal Pandya | 0.33 | 13/14 |
| 8 | Namboori Tilak Varma | 0.14 | 12/15 | Kuldeep Yadav | 0.28 | 11/15 |
| 9 | Mohammad Nawaz | 0.27 | 6/13 | Wanindu Hasaranga | 0.24 | 10/15 |
| 10 | Suryakumar Yadav | 0.15 | 9/15 | Mohsin Khan | 0.26 | 2/14 |

Clutch test (split-half, odd vs even matches): uplift r = 0.071 (batters), -0.016 (bowlers). Low power: the largest
between-player clutch variance not ruled out at 90% is 0.0123 (batters) and 0.0047 (bowlers), against
half-sample noise of 0.041 / 0.032. Report wording: large clutch effects are ruled out; small ones cannot be detected.
Weak places: batters #9 (Nawaz, 6/13 variants) and #10 (Suryakumar); bowlers #10 (Mohsin Khan, 2/14 where eligible). Next in line:
Russell, Sikandar Raza; Santner, Axar Patel, Bumrah.

## Review of 11 Nov 2026 (T5b)
- Method review (adversarial): ranking on all balls dodged the question (#1 batter declined under pressure) ->
  D9. Uncertainty too narrow -> fixed. First-innings pressure is weak evidence (swing 1.3x) -> report presents
  innings separately; chase-only variant added.
- Pipeline review: no blocking defects; lists reproduced independently. Fixed: posterior draws (shrink^2 * v ->
  shrink * v), stumpings/run-outs off wides now charged to the striker, par is leave-one-match-out, baseline
  cells include season. Disclosed, not fixed: Elo built on all past results feeds later training folds; par uses
  other matches of the same season; model chosen on the folds it is scored on (3 candidates).
- Consistency check (Haiku): all doc numbers matched outputs except two rounding differences, now regenerated.

## Status (end of 11 Nov 2026) — resume here
- Git repo with GitHub remote `origin`, no commits. Nothing committed, pushed or deployed. Data pack is git-ignored.
- Verified: T1-T5b. Draft: T7 report at outputs/report/SupeRR_Selector_Methodology.docx (not proofread in Word).
- Lists (main variant, outputs/ratings/top10_*.csv) are the submission candidates; ranks 8-10 are uncertain.
- Next (12 Nov): T6 dashboard outside OneDrive (proposed C:\Users\maste\rr-dashboard; owner to confirm), then
  owner proofreads the report, clean re-run of all four scripts, name check against the Unstop dropdowns,
  owner deploys the dashboard to Vercel, add the dashboard link to the report.
- Owner preference: Haiku subagents for small tasks and reviews.
