"""Synthetic faults injected into real temperature measurements.

Each fault returns the change it makes to the measured temperature, ``delta(t)``, so the faulty signal is
``T + delta``. None of the three normal behaviour models uses the measured target temperature as an input, so a
fault moves each model's residual by exactly ``delta`` and the injection study needs no refitting.

The faults are defined from measured data only (never from one of the models), so no model is favoured:

- ``drift_2``/``drift_5``: sensor or component drift, ramping linearly to +2/+5 degC over 30 days, then held;
- ``cooling_25``: cooling degradation (fouled cooler, weak fan), the thermal resistance to the nacelle grows
  by 25 % over 30 days, so the temperature rise above the nacelle air grows by 25 % - a load-dependent fault;
- ``step_3``: +3 degC step;
- ``stuck``: the sensor freezes at its value at onset.

Usage:
    delta = inject("cooling_25", frame["gear_oil"], frame["nacelle_temp"], onset)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAMP_DAYS = 30.0
FAULTS = {
    "drift_2": "drift +2 °C (30-day ramp)",
    "drift_5": "drift +5 °C (30-day ramp)",
    "cooling_25": "cooling degraded, R +25 % (30-day ramp)",
    "step_3": "step +3 °C",
    "stuck": "stuck sensor",
}
UPWARD = {"drift_2", "drift_5", "cooling_25", "step_3"}  # faults that make the reading too hot


def ramp(index: pd.DatetimeIndex, onset: pd.Timestamp, days: float = RAMP_DAYS) -> np.ndarray:
    """0 before onset, rising linearly to 1 over ``days``, then 1."""
    elapsed = (index - onset) / pd.Timedelta(days=1)
    return np.clip(np.asarray(elapsed, dtype=float) / days, 0.0, 1.0)


def inject(fault: str, temp: pd.Series, nacelle: pd.Series, onset: pd.Timestamp) -> np.ndarray:
    """Additive change ``delta`` to the measured temperature for the given fault (0 before onset)."""
    after = np.asarray(temp.index >= onset)
    t = temp.to_numpy(dtype=float)
    if fault == "drift_2":
        return 2.0 * ramp(temp.index, onset)
    if fault == "drift_5":
        return 5.0 * ramp(temp.index, onset)
    if fault == "step_3":
        return np.where(after, 3.0, 0.0)
    if fault == "cooling_25":
        rise = np.clip(t - nacelle.to_numpy(dtype=float), 0.0, None)
        return np.nan_to_num(0.25 * ramp(temp.index, onset) * rise)
    if fault == "stuck":
        first = np.flatnonzero(after & np.isfinite(t))
        if not len(first):
            return np.zeros(len(t))
        return np.where(after, t[first[0]] - t, 0.0)
    raise ValueError(f"unknown fault {fault!r}")
