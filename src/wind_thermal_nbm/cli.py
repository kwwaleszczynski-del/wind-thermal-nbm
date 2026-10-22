"""Command line interface.

Usage:
    python -m wind_thermal_nbm fetch                 # download the yearly Kelmarsh SCADA archives (~1.8 GB)
    python -m wind_thermal_nbm prepare               # extract temperatures and status logs to Parquet
    python -m wind_thermal_nbm fit --jobs 3          # fit RC / GBM / fleet models, cache residuals
    python -m wind_thermal_nbm evaluate              # thresholds, false alarms, injected faults, real events
    python -m wind_thermal_nbm derating              # thermal power limitation and its energy cost
    python -m wind_thermal_nbm report                # figures for the README and the HTML dashboard
    python -m wind_thermal_nbm all --jobs 3          # everything after fetch

The raw archives are read from $KELMARSH_DATA_DIR/raw if that variable is set, else from data/raw.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import data, pipeline

YEARS = list(range(2016, 2022))
STAGES = ("prepare", "fit", "evaluate", "derating", "report")
log = logging.getLogger("wind_thermal_nbm")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="wind-thermal-nbm",
                                description="Thermal normal behaviour models on the open Kelmarsh SCADA data.")
    p.add_argument("command", choices=("fetch", *STAGES, "all"))
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="cache and intermediate results")
    p.add_argument("--results-dir", type=Path, default=Path("results"), help="tables committed with the repo")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"), help="dashboard and README figures")
    p.add_argument("--jobs", type=int, default=1, help="test years fitted in parallel processes")
    p.add_argument("--refresh", action="store_true", help="rebuild caches instead of reusing them")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def cmd_prepare(args, raw_dir: Path) -> None:
    data.load(YEARS, raw_dir, args.data_dir / "processed", refresh=args.refresh)


def cmd_fit(args, raw_dir: Path) -> None:
    out = args.data_dir / "models"
    cache = args.data_dir / "processed"
    trend = out / "rc_by_year.csv"
    if args.refresh or not trend.exists():
        out.mkdir(parents=True, exist_ok=True)
        pipeline.rc_by_year(raw_dir, cache).to_csv(trend, index=False)
    todo = [y for y in pipeline.TEST_YEARS if args.refresh or not (out / f"residuals_{y}.parquet").exists()]
    if not todo:
        log.info("all test years cached in %s", out)
        return
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(todo))) as ex:
            for y in ex.map(pipeline.fit_and_save, todo, [raw_dir] * len(todo), [cache] * len(todo),
                            [out] * len(todo)):
                log.info("test year %d done", y)
    else:
        for y in todo:
            pipeline.fit_and_save(y, raw_dir, cache, out)


def cmd_evaluate(args, raw_dir: Path) -> None:
    from .evaluate import evaluate
    evaluate(raw_dir, args.data_dir, args.results_dir)


def cmd_derating(args, raw_dir: Path) -> None:
    from .derating import run
    run(raw_dir, args.data_dir, args.results_dir)


def cmd_report(args, raw_dir: Path) -> None:
    from .report import report
    report(raw_dir, args.data_dir, args.results_dir, args.docs_dir)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw_dir = data.raw_dir_default(args.data_dir)
    if args.command == "fetch":
        data.fetch(YEARS, raw_dir)
        return
    stages = STAGES if args.command == "all" else (args.command,)
    for stage in stages:
        log.info("== %s", stage)
        globals()[f"cmd_{stage}"](args, raw_dir)


if __name__ == "__main__":
    main()
