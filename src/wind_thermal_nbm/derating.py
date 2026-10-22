"""Thermal power limitation: do the turbines give away power when they run hot, and what does it cost?

Below rated wind a turbine running freely keeps its blades at fine pitch. Pitched-out blades with the power
still well below rated mean the turbine is limiting itself (or being limited). Two kinds are told apart by
who is limited at the same moment:

- farm-wide: at least ``FARM_LIMITED`` turbines limited together - an external set-point (curtailment, grid
  or park control), not a property of one turbine's temperatures;
- single-turbine: only this turbine (or fewer than ``FARM_LIMITED``). It is attributed to heat when one of
  its component temperatures is in that component's hottest 1 % or the ambient air is above 25 degC.

The lost energy is the turbine's own power curve (density-normalised wind speed, median power per 0.5 m/s
bin from samples running at fine pitch) minus the measured power. Logged "Reduced power" warnings are reported
on their own.

Usage:
    run(raw_dir=Path("data/raw"), data_dir=Path("data"), results_dir=Path("results"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COMPONENTS, load
from .prep import clean

YEARS = tuple(range(2017, 2022))
SITE_PRESSURE_PA = 98_700.0  # ~225 m above sea level (ground 135-157 m + hub 68.5-78.5 m), standard atmosphere
RHO_REF = 1.225
LIMIT_PITCH_DEG = 3.0
RATED_KW = 2050.0
LIMIT_BELOW_KW = 0.93 * RATED_KW
WS_RANGE = (4.0, 10.0)  # m/s, density-normalised; well below rated, where pitch should be fine
FARM_LIMITED = 3
HOT_QUANTILE = 0.99
HOT_AMBIENT_C = 25.0
THERMAL_CODES = (75, 3220)  # "Reduced power gearbox", "Reduced power converter"
log = logging.getLogger(__name__)


def normalised_wind(df: pd.DataFrame) -> pd.Series:
    """Mean of both nacelle anemometers, normalised to standard air density (IEC 61400-12-1 style)."""
    ws = df[["ws1", "ws2"]].mean(axis=1)
    rho = SITE_PRESSURE_PA / (287.05 * (df["ambient_temp"] + 273.15))
    return ws * (rho / RHO_REF) ** (1.0 / 3.0)


def expected_power(df: pd.DataFrame) -> pd.Series:
    """Per turbine, median power per 0.5 m/s bin of samples at fine pitch (the turbine's unlimited curve)."""
    b = (df["ws_n"] * 2).round() / 2
    free = df["pitch"] < 1.0
    curve = df[free].groupby([df.loc[free, "turbine"], b[free]])["power_kw"].median()
    curve.index.names = ["turbine", "bin"]
    return pd.Series(curve.reindex(pd.MultiIndex.from_arrays([df["turbine"], b])).to_numpy(), index=df.index)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add ws_n, p_exp, limited, farm_limited, hot (component name or ''), thermal, lost_kwh."""
    out = df.copy()
    out["ws_n"] = normalised_wind(out)
    running = out["rotor_rpm"] > 5
    out["p_exp"] = expected_power(out)
    window = out["ws_n"].between(*WS_RANGE)
    out["limited"] = running & window & (out["pitch"] > LIMIT_PITCH_DEG) & (out["power_kw"] < LIMIT_BELOW_KW)
    n_limited = out.groupby("timestamp")["limited"].transform("sum")
    out["farm_limited"] = out["limited"] & (n_limited >= FARM_LIMITED)
    hot = pd.Series("", index=out.index)
    for comp in COMPONENTS.values():
        q = out.groupby("turbine")[comp].transform(lambda s: s.quantile(HOT_QUANTILE))
        hot = hot.where(hot != "", np.where(out[comp] >= q, comp, ""))
    hot = hot.where(hot != "", np.where(out["ambient_temp"] > HOT_AMBIENT_C, "ambient", ""))
    out["hot"] = hot
    out["thermal"] = out["limited"] & ~out["farm_limited"] & (out["hot"] != "")
    out["lost_kwh"] = ((out["p_exp"] - out["power_kw"]).clip(lower=0) / 6.0).where(out["limited"], 0.0)
    return out


def logged_derating(status: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Hours and lost energy during logged 'Reduced power' warnings."""
    ev = status[status["code"].isin(THERMAL_CODES)]
    rows = []
    for _, e in ev.iterrows():
        end = e["end"] if pd.notna(e["end"]) else e["start"] + pd.Timedelta(minutes=10)
        m = (df["turbine"] == e["turbine"]) & (df["timestamp"] >= e["start"].floor("10min")) & (df["timestamp"] < end)
        lost = ((df.loc[m, "p_exp"] - df.loc[m, "power_kw"]).clip(lower=0) / 6.0).sum()
        rows.append({"turbine": int(e["turbine"]), "start": e["start"], "end": end, "message": e["message"],
                     "hours": (end - e["start"]) / pd.Timedelta(hours=1), "lost_kwh": float(lost)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Hours and MWh per year and class (farm-wide, single-turbine hot, single-turbine other)."""
    cls = np.select([df["farm_limited"], df["thermal"], df["limited"]],
                    ["farm-wide limitation", "single turbine, hot", "single turbine, not hot"], "")
    d = df.assign(cls=cls, year=df["timestamp"].dt.year)
    produced = d.groupby("year")["power_kw"].apply(lambda p: p.clip(lower=0).sum() / 6.0 / 1000.0)
    lim = d[d["cls"] != ""]
    out = lim.groupby(["year", "cls"]).agg(hours=("limited", lambda s: s.size / 6.0),
                                           lost_mwh=("lost_kwh", lambda s: s.sum() / 1000.0)).reset_index()
    out["produced_mwh"] = out["year"].map(produced)
    out["lost_pct"] = 100 * out["lost_mwh"] / out["produced_mwh"]
    return out


def by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Share of below-rated running time spent limited, per ambient temperature band and class."""
    w = df[df["ws_n"].between(*WS_RANGE) & (df["rotor_rpm"] > 5)]
    band = pd.cut(w["ambient_temp"], [-15, 0, 5, 10, 15, 20, 25, 30, 40])
    g = w.groupby(band, observed=True)
    single = w["limited"] & ~w["farm_limited"]
    return pd.DataFrame({
        "samples": g.size(),
        "farm_wide_pct": 100 * g["farm_limited"].mean(),
        "single_turbine_pct": 100 * single.groupby(band, observed=True).mean(),
        "single_turbine_hot_component_pct": 100 * (single & ~w["hot"].isin(["", "ambient"]))
        .groupby(band, observed=True).mean(),
    }).reset_index().rename(columns={"ambient_temp": "ambient_band"})


def run(raw_dir: Path, data_dir: Path, results_dir: Path) -> pd.DataFrame:
    scada, status = load(list(YEARS), raw_dir, data_dir / "processed")
    df = classify(clean(scada).reset_index())
    summary = summarise(df)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "derating_by_year.csv", index=False)
    by_temperature(df).to_csv(results_dir / "derating_by_ambient.csv", index=False)
    hot = df[df["thermal"]]
    hot.groupby(["turbine", "hot"]).agg(hours=("thermal", lambda s: s.size / 6.0),
                                        lost_mwh=("lost_kwh", lambda s: s.sum() / 1000.0)).reset_index() \
        .to_csv(results_dir / "derating_hot_by_component.csv", index=False)
    logged = logged_derating(status[status["start"].dt.year.isin(list(YEARS))], df)
    logged.to_csv(results_dir / "derating_logged.csv", index=False)
    df[["timestamp", "turbine", "ws_n", "p_exp", "power_kw", "pitch", "ambient_temp", "limited", "farm_limited",
        "thermal", "hot", "lost_kwh"]].to_parquet(data_dir / "derating.parquet")
    with pd.option_context("display.width", 200):
        log.info("limitation by year:\n%s", summary.round(2))
    return summary
