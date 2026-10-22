"""Evaluation stage: residual spread, thresholds, false alarms, injected faults and real events.

Reads the cached residuals of ``fit`` and writes summary tables to ``results/`` (committed) and the full
per-injection table and daily z-scores to ``data/models`` (not committed).

Usage:
    evaluate(raw_dir=Path("data/raw"), data_dir=Path("data"), results_dir=Path("results"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline, validation
from .data import load
from .pipeline import TEST_YEARS, split_column
from .prep import turbine_frame, year_bounds

log = logging.getLogger(__name__)


def _frames(scada: pd.DataFrame, test_year: int) -> dict[int, pd.DataFrame]:
    start, end = year_bounds(test_year - 1)[0], year_bounds(test_year)[1]
    return {t: turbine_frame(scada, t, start, end) for t in pipeline.TURBINES}


def residual_spread(cal: validation.YearCalibration) -> pd.DataFrame:
    """Robust spread (1.4826 MAD, degC) of the daily residual in the calibration year, per channel."""
    rows = [(cal.test_year, *split_column(c), cal.scale[c][1]) for c in cal.z.columns]
    return pd.DataFrame(rows, columns=["test_year", "method", "component", "turbine", "daily_sd_K"])


def evaluate(raw_dir: Path, data_dir: Path, results_dir: Path) -> None:
    models = data_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    scada, status = load(list(range(min(TEST_YEARS) - 2, max(TEST_YEARS) + 1)), raw_dir, data_dir / "processed")
    spread, thresholds, fa, inj, alarms, z_test = [], [], [], [], [], []
    for y in TEST_YEARS:
        frames = _frames(scada, y)
        residuals = pipeline.load_residuals(models, y)
        for baseline in validation.BASELINES:
            cal = validation.calibrate_year(residuals, frames, y, baseline)
            if baseline == "fixed":
                spread.append(residual_spread(cal))
            thresholds += [{"test_year": y, "baseline": baseline, "method": m, "component": comp, "chart": c,
                            "threshold": h} for (m, comp, c), h in cal.thresholds.items()]
            for chart in ("cusum", "ewma"):
                fa.append(validation.false_alarms(cal, chart))
                a = validation.alarm_days(cal, chart)
                a["chart"] = chart
                a["test_year"] = y
                alarms.append(a)
            inj.append(validation.injection_study(cal, frames))
            z_test.append(cal.z.iloc[cal.n_cal_days:].assign(baseline=baseline))
            log.info("test %d, %s baseline: median CUSUM threshold %.1f, %d injection runs", y, baseline,
                     np.median([v for k, v in cal.thresholds.items() if k[2] == "cusum"]), len(inj[-1]))
    spread = pd.concat(spread)
    fa = pd.concat(fa)
    inj = pd.concat(inj, ignore_index=True)
    alarms = pd.concat(alarms, ignore_index=True)
    inj.to_parquet(models / "injections.parquet")
    z = pd.concat(z_test)
    z.set_index("baseline", append=True).astype("float32").to_parquet(models / "daily_z.parquet")
    alarms.to_csv(results_dir / "alarms.csv", index=False)
    pd.DataFrame(thresholds).to_csv(results_dir / "thresholds.csv", index=False)
    spread.to_csv(results_dir / "residual_spread.csv", index=False)
    pipeline.rc_params(models).to_csv(results_dir / "rc_params.csv", index=False)
    pd.read_csv(models / "rc_by_year.csv").to_csv(results_dir / "rc_by_year.csv", index=False)

    a_hot = alarms[alarms.direction > 0].groupby(["baseline", "method", "chart"]).size()
    fa_sum = (fa.groupby(["baseline", "method", "chart"]).alarms.agg(["sum", "count"])
              .assign(hot=a_hot).fillna({"hot": 0})
              .assign(per_turbine_year=lambda d: 9 * d["sum"] / d["count"],
                      hot_per_turbine_year=lambda d: 9 * d["hot"] / d["count"]).reset_index())
    fa.to_csv(results_dir / "false_alarms_by_channel.csv", index=False)
    fa_sum.to_csv(results_dir / "false_alarms.csv", index=False)
    summary = validation.summarise_injections(inj)
    summary.to_csv(results_dir / "injection_summary.csv", index=False)
    by_comp = (inj[(inj.valid_days >= 30) & (inj.chart == "cusum")]
               .groupby(["baseline", "fault", "method", "component"]).detected.mean().mul(100).unstack("method"))
    by_comp.round(1).to_csv(results_dir / "injection_by_component.csv")

    episodes = validation.event_episodes(status, TEST_YEARS)
    days = pd.date_range(year_bounds(min(TEST_YEARS))[0], year_bounds(max(TEST_YEARS))[1], freq="D",
                         inclusive="left")
    events = []
    for baseline in validation.BASELINES:
        cus = alarms[(alarms.chart == "cusum") & (alarms.baseline == baseline)]
        events.append(validation.score_events(episodes, cus, days, pipeline.METHODS).assign(baseline=baseline))
    pd.concat(events).to_csv(results_dir / "events.csv", index=False)
    log.info("wrote results to %s", results_dir)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info("false alarms:\n%s", fa_sum.round(2))
        for baseline in validation.BASELINES:
            s = summary[(summary.chart == "cusum") & (summary.baseline == baseline)]
            log.info("injections, %s baseline (cusum):\n%s", baseline,
                     s.pivot(index="fault", columns="method", values="detected_pct").round(0))
