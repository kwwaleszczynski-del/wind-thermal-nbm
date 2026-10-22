"""Per-turbine regular 10-minute frames, plausibility cleaning and the operating mask.

Usage:
    frame = turbine_frame(scada, turbine=3, start="2019-01-01", end="2020-01-01")
    op = operating(frame)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import COMPONENTS

TEMPERATURES = [*COMPONENTS.values(), "nacelle_temp", "ambient_temp", "hub_temp"]
TEMP_RANGE = (-40.0, 150.0)  # the export contains e.g. -273.2 degC sensor error values
OPERATING_KW = 100.0  # ~5 % of rated power
OPERATING_RPM = 5.0
STEP = pd.Timedelta(minutes=10)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in TEMPERATURES:
        if c in out:
            v = out[c]
            out[c] = v.where((v > TEMP_RANGE[0]) & (v < TEMP_RANGE[1]))
    return out


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def turbine_frame(scada: pd.DataFrame, turbine: int, start: str | pd.Timestamp,
                  end: str | pd.Timestamp) -> pd.DataFrame:
    """Rows of one turbine on a complete 10-minute UTC grid [start, end), missing rows as NaN, float64."""
    start, end = _utc(start), _utc(end)
    grid = pd.date_range(start, end - STEP, freq=STEP, name="timestamp")
    one = scada.xs(turbine, level="turbine") if turbine in scada.index.get_level_values(0) else scada.iloc[:0]
    one = one[(one.index >= start) & (one.index < end)].astype("float64")
    return clean(one.reindex(grid))


def operating(frame: pd.DataFrame) -> pd.Series:
    """True while the turbine is producing (power and rotor speed above small thresholds)."""
    return (frame["power_kw"] > OPERATING_KW) & (frame["rotor_rpm"] > OPERATING_RPM)


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")


def fleet_matrix(frames: dict[int, pd.DataFrame], column: str) -> pd.DataFrame:
    """One column per turbine, shared 10-minute index."""
    return pd.DataFrame({t: f[column] for t, f in frames.items()})


def median_of_others(values: pd.DataFrame, min_others: int = 3) -> pd.DataFrame:
    """For each turbine column, the row-wise median of the other turbines (NaN if fewer than ``min_others``)."""
    arr = values.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    for j in range(arr.shape[1]):
        others = np.delete(arr, j, axis=1)
        enough = np.isfinite(others).sum(axis=1) >= min_others
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            med = np.nanmedian(np.where(enough[:, None], others, np.nan), axis=1)
        out[:, j] = med
    return pd.DataFrame(out, index=values.index, columns=values.columns)
