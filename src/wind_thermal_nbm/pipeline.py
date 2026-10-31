"""Rolling protocol: fit on year Y-2, calibrate alarm thresholds on Y-1, test on Y.

For every test year, component and turbine the three normal behaviour models are fitted on the training year
and their 10-minute residuals (measured - expected) are computed for the calibration and the test year. The
residuals are cached as Parquet (``residuals_<Y>.parquet``) together with the fitted RC parameters, so the
alarm and validation stages can be rerun without refitting.

Methods:
    rc        physical first-order thermal model (own method)
    gbm       gradient-boosted trees on lagged operating conditions (industry-standard NBM)
    fleet     temperature minus the median of the other turbines, load-corrected
    rc_fleet  RC residual minus the median RC residual of the other turbines (model + fleet combined)
    gbm_fleet the same combination for the gradient-boosting residual

Usage:
    residuals, params = fit_test_year(scada, 2020)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data import COMPONENTS, load
from .fleet import FleetModel, operating_matrix
from .ml import GBMModel
from .prep import median_of_others, turbine_frame, year_bounds
from .thermal import RCModel

AIR = ("nacelle_temp", "ambient_temp")
OIL = ("gear_oil", "nacelle_temp", "ambient_temp")
# what each component gives its heat to: the gearbox bearings and the cooler outlet sit in the gear oil
COMPONENT_REFS = {c: OIL if c in ("front_bearing", "rear_bearing", "gear_inlet") else AIR for c in COMPONENTS.values()}
TEST_YEARS = (2019, 2020, 2021)
METHODS = ("rc", "gbm", "fleet", "rc_fleet", "gbm_fleet")
TURBINES = tuple(range(1, 7))
log = logging.getLogger(__name__)


def column(method: str, component: str, turbine: int) -> str:
    return f"{method}|{component}|{turbine}"


def split_column(name: str) -> tuple[str, str, int]:
    m, c, t = name.split("|")
    return m, c, int(t)


def frames_for(scada: pd.DataFrame, year: int, turbines=TURBINES) -> dict[int, pd.DataFrame]:
    return {t: turbine_frame(scada, t, *year_bounds(year)) for t in turbines}


def fit_test_year(scada: pd.DataFrame, test_year: int, components=None,
                  turbines=TURBINES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residuals (10-minute, calibration + test year, float32) and RC parameters for one test year."""
    components = list(components or COMPONENTS.values())
    train = frames_for(scada, test_year - 2, turbines)
    evaluate = {t: pd.concat([turbine_frame(scada, t, *year_bounds(test_year - 1)),
                              turbine_frame(scada, t, *year_bounds(test_year))]) for t in turbines}
    cols, params = {}, []
    for comp in components:
        refs = COMPONENT_REFS[comp]
        res = {"rc": {}, "gbm": {}}
        for t in turbines:
            rc = RCModel(refs=refs).fit(train[t], comp)
            gbm = GBMModel(refs=refs).fit(train[t], comp)
            y = evaluate[t][comp]
            res["rc"][t] = y - rc.predict(evaluate[t])
            res["gbm"][t] = y - gbm.predict(evaluate[t])
            params.append({"test_year": test_year, "train_year": test_year - 2, "component": comp, "turbine": t,
                           **rc.params()})
        res = fleet_methods(comp, train, evaluate, pd.DataFrame(res["rc"]), pd.DataFrame(res["gbm"]))
        for m in METHODS:
            for t in turbines:
                cols[column(m, comp, t)] = res[m][t]
        log.info("test %d  %-18s fitted (RC tau %s h)", test_year, comp,
                 " ".join(f"{p['tau_h']:.2f}" for p in params[-len(turbines):]))
    residuals = pd.DataFrame(cols).astype("float32")
    return residuals, pd.DataFrame(params)


def fleet_methods(comp: str, train: dict[int, pd.DataFrame], evaluate: dict[int, pd.DataFrame],
                  rc: pd.DataFrame, gbm: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The fleet reference and the model residuals referenced to the other producing turbines' residuals."""
    op = operating_matrix(evaluate)
    out = {"rc": rc, "gbm": gbm, "fleet": FleetModel().fit(train, comp).residuals(evaluate)}
    for m in ("rc", "gbm"):
        out[f"{m}_fleet"] = out[m] - median_of_others(out[m].where(op))
    return out


def load_residuals(out_dir: Path, test_year: int) -> pd.DataFrame:
    return pd.read_parquet(out_dir / f"residuals_{test_year}.parquet")


def rc_params(out_dir: Path, test_years=TEST_YEARS) -> pd.DataFrame:
    return pd.concat([pd.read_csv(out_dir / f"rc_params_{y}.csv") for y in test_years], ignore_index=True)


def fit_and_save(test_year: int, raw_dir: Path, cache_dir: Path, out_dir: Path) -> int:
    """Worker for one test year (loads its own data so test years can run in parallel processes)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scada, _ = load(list(range(test_year - 2, test_year + 1)), raw_dir, cache_dir)
    residuals, params = fit_test_year(scada, test_year)
    out_dir.mkdir(parents=True, exist_ok=True)
    residuals.to_parquet(out_dir / f"residuals_{test_year}.parquet")
    params.to_csv(out_dir / f"rc_params_{test_year}.csv", index=False)
    return test_year


def rc_by_year(raw_dir: Path, cache_dir: Path, years=range(2017, 2022)) -> pd.DataFrame:
    """RC parameters fitted separately on every calendar year: a physical health trend per channel.

    The cooling-side weights are constrained to a mixture (>= 0, sum 1); left free they drift from year to year
    and carry the temperature rise with them."""
    scada, _ = load(list(years), raw_dir, cache_dir)
    rows = []
    for y in years:
        frames = frames_for(scada, y)
        for comp in COMPONENTS.values():
            for t, f in frames.items():
                p = RCModel(refs=COMPONENT_REFS[comp], mixture=True).fit(f, comp).params()
                rows.append({"year": y, "component": comp, "turbine": t, **p})
        log.info("RC parameters fitted for %d", y)
    return pd.DataFrame(rows)
