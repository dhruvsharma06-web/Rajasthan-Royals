"""R4/R5 check: the two top-10 lists are identical in every deliverable.

Compares outputs/ratings/top10_*.csv (the submission source) with the report (DOCX), the deck (PPTX) and the
dashboard data (dashboard/data.js). Also checks 10 unique names per list.
Run from the project root after report.py, slides.py and export_dashboard.py:  python -P pipeline/check_outputs.py
"""
import json
import sys

import polars as pl
from docx import Document
from pptx import Presentation

ok = True


def check(label, got, want):
    global ok
    if got != want:
        ok = False
        print(f"MISMATCH {label}:\n  got  {got}\n  want {want}")
    else:
        print(f"ok  {label}")


want = {r: pl.read_csv(f"outputs/ratings/top10_{r}.csv").sort("rank")["name"].to_list() for r in ("batters", "bowlers")}
for r, names in want.items():
    check(f"{r}: 10 unique names", (len(names), len(set(names))), (10, 10))

# report: summary table (Rank | Batter | Bowler) and the two evidence tables
doc = Document("outputs/report/SupeRR_Selector_Methodology.docx")
summary = next(t for t in doc.tables if [c.text for c in t.rows[0].cells] == ["Rank", "Batter", "Bowler"])
rows = [[c.text for c in row.cells] for row in summary.rows[1:]]
check("report summary ranks", [r[0] for r in rows], [str(i) for i in range(1, 11)])
check("report summary batters", [r[1] for r in rows], want["batters"])
check("report summary bowlers", [r[2] for r in rows], want["bowlers"])
evidence = [t for t in doc.tables if t.rows[0].cells[1].text == "Player"]
check("report evidence tables found", len(evidence), 2)
for t, r in zip(evidence, ("batters", "bowlers")):
    check(f"report evidence {r}", [row.cells[1].text for row in t.rows[1:]], want[r])

# deck: the two top-10 tables
prs = Presentation("outputs/report/SupeRR_Selector_Pressure_Ratings.pptx")
found = {}
for s in prs.slides:
    for sh in s.shapes:
        if sh.has_table:
            hdr = sh.table.cell(0, 1).text
            role = {"Batter": "batters", "Bowler": "bowlers"}.get(hdr)
            if role:
                found[role] = [sh.table.cell(i, 1).text for i in range(1, len(sh.table.rows))]
for r in ("batters", "bowlers"):
    check(f"deck {r}", found.get(r), want[r])

# dashboard data
txt = open("dashboard/data.js").read()
data = json.loads(txt[txt.index("{"):txt.rstrip().rindex(";")])
for r in ("batters", "bowlers"):
    check(f"dashboard {r}", [p["name"] for p in data[r][:10]], want[r])

print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
