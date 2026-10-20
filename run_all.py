"""Run the whole audit: experiments -> tidy CSVs -> tables, tests and figures.

    python run_all.py            # full run, writes results/
    python run_all.py --quick    # smoke test (2 coins, 2024+), writes results/quick/
    python run_all.py --analysis-only   # re-make tables/figures from results/cache/
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xaiaudit.analysis import run_analysis  # noqa: E402
from xaiaudit.config import ROOT, get_config  # noqa: E402
from xaiaudit.experiments import run_all_coins  # noqa: E402


def versions() -> dict:
    import lightgbm, numpy, pandas, shap, sklearn, xgboost  # noqa: E401
    import lime
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "shap": shap.__version__,
            "lime": getattr(lime, "__version__", "0.2.0.1")}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke-test configuration")
    ap.add_argument("--analysis-only", action="store_true", help="skip experiments, reuse results/cache/")
    ap.add_argument("--results-dir", default=None, help="override output directory")
    args = ap.parse_args(argv)

    default_dir = ROOT / "results" / "quick" if args.quick else ROOT / "results"
    cfg = get_config(quick=args.quick, overrides={"results_dir": str(args.results_dir or default_dir)})
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.time()
    if not args.analysis_only:
        print(f"experiments: coins={cfg['coins']} models={cfg['models']} n_jobs={cfg['n_jobs']}", flush=True)
        run_all_coins(cfg)
        timings["experiments_s"] = round(time.time() - t0, 1)
        print(f"experiments done in {timings['experiments_s']}s", flush=True)

    t1 = time.time()
    out = run_analysis(cfg)
    timings["analysis_s"] = round(time.time() - t1, 1)
    timings["total_s"] = round(time.time() - t0, 1)
    print(f"analysis done in {timings['analysis_s']}s; total {timings['total_s']}s", flush=True)

    log = {"quick": args.quick, "timings": timings, "versions": versions(),
           "config": {k: v for k, v in cfg.items() if not k.endswith("_dir")}}
    name = "run_log_analysis_only.json" if args.analysis_only else "run_log.json"
    (Path(cfg["results_dir"]) / name).write_text(json.dumps(log, indent=2))
    kn = out["key_numbers"]
    print(f"regime tests: {kn['n_significant_q05']}/{kn['n_tests']} significant at BH q<0.05")
    return out


if __name__ == "__main__":
    main()
