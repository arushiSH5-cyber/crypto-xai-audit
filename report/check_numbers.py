"""Check that every decimal number in report.md appears in results/ (tables, key numbers, run log).

    python report/check_numbers.py

Each number in the report (ignoring the References section) is matched against every numeric
value in results/tables/*.csv, results/tables/key_numbers.json and results/run_log.json,
rounded to the same number of decimals as written. Unmatched numbers are listed for manual
review. Most of those turn out to be configuration values (eps, subsample rate) or simple
derived quantities, which are explained in the text.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _walk(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)


def known_values() -> list[float]:
    vals = []
    for p in (ROOT / "results" / "tables").glob("*.csv"):
        df = pd.read_csv(p)
        for c in df.select_dtypes("number").columns:
            vals += df[c].dropna().astype(float).tolist()
    for p in [ROOT / "results" / "tables" / "key_numbers.json", ROOT / "results" / "run_log.json"]:
        if p.exists():
            vals += list(_walk(json.loads(p.read_text())))
    return vals


def main() -> int:
    text = (ROOT / "report" / "report.md").read_text()
    text = text.split("# References")[0]
    text = re.sub(r"!\[.*?\]\(.*?\)\{.*?\}", "", text)              # figure captions/paths
    nums = sorted(set(re.findall(r"(?<![\w.])[−-]?\d+\.\d+", text)))
    vals = known_values()
    missing = []
    for s in nums:
        x = abs(float(s.replace("−", "-")))
        d = len(s.split(".")[1])
        if not any(round(abs(v), d) == round(x, d) for v in vals):
            missing.append(s)
    print(f"{len(nums)} distinct decimal numbers in report; {len(nums) - len(missing)} matched results/.")
    if missing:
        print("Not found verbatim in results/ (check by hand):", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
