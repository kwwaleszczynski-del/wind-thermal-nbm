"""Control charts on daily temperature residuals: two-sided CUSUM and EWMA, with thresholds set on a calibration year.

Residuals are averaged per day over the hours when the turbine is producing, then standardised with the
median and MAD of the calibration year (so the calibration year is centred and has unit spread) and clipped to
+-5 so a single bad day cannot trip a chart on its own. Both charts reset after an alarm, so an alarm count is
a count of alarm episodes. The CUSUM reference value k = 1 tunes it to shifts of about two daily standard
deviations: the daily residuals of every model wander by about one standard deviation for weeks with the
weather, and a chart tuned to smaller shifts would mostly accumulate those.

Usage:
    alarm, statistic = run_chart("cusum", z, threshold=12.0)   # alarm: +1 hot / -1 cold / 0 per day
"""
from __future__ import annotations

import numpy as np

MIN_OPERATING_SAMPLES = 36  # 6 h of production per day
Z_CLIP = 5.0


def cusum_run(z: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided tabular CUSUM. Returns (alarm direction per day: +1/-1/0, statistic max(S+, S-) per day).

    Missing days leave the statistics unchanged; both sides reset to zero after an alarm.
    """
    sp = sn = 0.0
    alarm = np.zeros(len(z), dtype=np.int8)
    stat = np.zeros(len(z))
    for i, v in enumerate(z):
        if np.isfinite(v):
            sp = max(0.0, sp + v - k)
            sn = max(0.0, sn - v - k)
        stat[i] = max(sp, sn)
        if sp > h or sn > h:
            alarm[i] = 1 if sp >= sn else -1
            sp = sn = 0.0
    return alarm, stat


def ewma_run(z: np.ndarray, lam: float, L: float) -> tuple[np.ndarray, np.ndarray]:
    """EWMA chart with asymptotic limits +-L sqrt(lam / (2 - lam)); resets to zero after an alarm.

    The returned statistic is |e| in units of the asymptotic standard deviation, comparable with L."""
    sd = np.sqrt(lam / (2.0 - lam))
    e = 0.0
    alarm = np.zeros(len(z), dtype=np.int8)
    stat = np.zeros(len(z))
    for i, v in enumerate(z):
        if np.isfinite(v):
            e = lam * v + (1.0 - lam) * e
        stat[i] = abs(e) / sd
        if abs(e) > L * sd:
            alarm[i] = 1 if e > 0 else -1
            e = 0.0
    return alarm, stat


CHARTS = {
    "cusum": (cusum_run, 1.0),  # reference value k (half the shift, in sigmas, the chart is tuned for)
    "ewma": (ewma_run, 0.2),  # smoothing lambda
}


def run_chart(chart: str, z: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    fn, param = CHARTS[chart]
    return fn(np.clip(z, -Z_CLIP, Z_CLIP), param, threshold)


def zero_alarm_threshold(chart: str, series: list[np.ndarray]) -> float:
    """Smallest threshold at which none of the calibration ``series`` raises an alarm: the largest value the
    chart statistic reaches on them (an alarm needs the statistic to exceed the threshold)."""
    return float(max(run_chart(chart, z, np.inf)[1].max() for z in series))
