"""Check that every decimal number in the report and docs appears in results/.

    python report/check_numbers.py                 # report/report.md
    python report/check_numbers.py README.md docs/*.md   # any other documents

Each number in the report (ignoring the References section) is matched against every numeric
value in results/tables/*.csv, results/tables/key_numbers.json, results/run_log.json and the
saved logs (results/*.log, results/*.txt),
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
    # timings quoted from the saved pytest and fresh-clone logs
    for p in list((ROOT / "results").glob("*.log")) + list((ROOT / "results").glob("*.txt")):
        vals += [float(x) for x in re.findall(r"\d+\.\d+", p.read_text())]
    return vals


def check(path: Path, vals: list[float]) -> list[str]:
    text = path.read_text()
    text = text.split("# References")[0]
    text = re.sub(r"!\[.*?\]\(.*?\)(\{.*?\})?", "", text)          # figure captions/paths
    text = re.sub(r"\d+\.\d+\.\d+", "", text)                        # version strings like 3.12.15
    nums = sorted(set(re.findall(r"(?<![\w.])[−-]?\d+\.\d+", text)))
    missing = []
    for s in nums:
        x = abs(float(s.replace("−", "-")))
        d = len(s.split(".")[1])
        if not any(round(abs(v), d) == round(x, d) for v in vals):
            missing.append(s)
    print(f"{path.relative_to(ROOT)}: {len(nums)} distinct decimal numbers, {len(nums) - len(missing)} matched results/.")
    if missing:
        print("   not found in results/ (check by hand):", ", ".join(missing))
    return missing


def main() -> int:
    files = [Path(f).resolve() for f in sys.argv[1:]] or [ROOT / "report" / "report.md"]
    vals = known_values()
    for f in files:
        check(f, vals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
