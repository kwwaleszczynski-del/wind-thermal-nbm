"""Alarm calibration and the three validations: false alarms, injected faults and real logged events.

Per test year Y the residuals of the calibration year Y-1 give, for every channel (method, component, turbine),
the standardisation of the daily residual and, per method, component and chart, one alarm threshold shared by
the six turbines: the smallest one at which none of them alarms during Y-1. Everything measured afterwards uses
only year Y:

- false alarms: alarm episodes on the unmodified test year (chart running continuously from the start of Y-1);
- injected faults: each fault starts on the first day of a month (January-October) and the chart is restarted
  at onset; a fault counts as detected if the chart alarms in the fault's direction within ``HORIZON_DAYS``.
  The same windows without a fault give the chance of a "detection" that has nothing to do with it;
- real events: logged alarms and warnings that point at a component; an event counts as anticipated if the
  continuously running chart raised a hot-side alarm on one of the component's channels in the
  ``HORIZON_DAYS`` before it. The fraction of all days with such an alarm in the preceding window is the hit
  rate expected by chance.

Two baselines for the daily residual are compared. ``fixed``: the calibration year's median, so the chart asks
"is this turbine different from how the model expected it?". ``trailing``: the median of the days t-120 ... t-31
(causal), so the chart asks "has this turbine changed in the last one to two months?". The trailing baseline
forgets slow year-to-year shifts of the model error, at the price of also forgetting a fault that develops
over many months.

Usage:
    cal = calibrate_year(residuals, frames, test_year, baseline="trailing")
    fa = false_alarms(cal)
    inj = injection_study(cal, frames)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alarms import CHARTS, MIN_OPERATING_SAMPLES, run_chart, zero_alarm_threshold
from .faults import FAULTS, UPWARD, inject
from .pipeline import split_column
from .prep import operating, year_bounds

HORIZON_DAYS = 60
ONSET_MONTHS = tuple(range(1, 11))
BASELINES = ("fixed", "trailing")
BASELINE_WINDOW = 90  # days in the trailing baseline
BASELINE_GAP = 30  # the most recent days are left out, so a developing fault is not absorbed at once
log = logging.getLogger(__name__)


def day_codes(index: pd.DatetimeIndex, start: pd.Timestamp) -> np.ndarray:
    """Whole days since ``start`` (independent of the datetime resolution of the index)."""
    return np.floor(np.asarray((index - start) / pd.Timedelta(days=1), dtype=float)).astype(int)


def daily_means(values: np.ndarray, mask: np.ndarray, days: np.ndarray, n_days: int,
                min_samples: int = MIN_OPERATING_SAMPLES) -> np.ndarray:
    """Mean of ``values`` per day over rows where ``mask`` holds; NaN with fewer than ``min_samples`` rows."""
    ok = mask & np.isfinite(values)
    cnt = np.bincount(days[ok], minlength=n_days)[:n_days]
    tot = np.bincount(days[ok], weights=values[ok], minlength=n_days)[:n_days]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= min_samples, tot / cnt, np.nan)


def standardise_daily(d: np.ndarray, med: float, mad: float, baseline: str) -> np.ndarray:
    """z-score of the daily residual against the fixed or the trailing baseline."""
    if baseline == "fixed":
        return (d - med) / mad
    base = (pd.Series(d).shift(BASELINE_GAP).rolling(BASELINE_WINDOW, min_periods=30).median()
            .fillna(med).to_numpy())
    return (d - base) / mad


def robust_scale(ref: np.ndarray) -> tuple[float, float]:
    ref = ref[np.isfinite(ref)]
    if len(ref) < 30:
        return float("nan"), float("nan")
    med = float(np.median(ref))
    return med, max(1.4826 * float(np.median(np.abs(ref - med))), 1e-6)


@dataclass
class YearCalibration:
    """Daily standardised residuals of one test year (calibration year first) and the chart thresholds."""

    test_year: int
    baseline: str
    start: pd.Timestamp  # first day of the calibration year
    n_cal_days: int
    z: pd.DataFrame  # index: day, columns: channel names "method|component|turbine"
    daily: pd.DataFrame  # the daily residual itself [K], same layout
    scale: dict[str, tuple[float, float]]
    thresholds: dict[tuple[str, str, str], float]  # (method, component, chart)
    residuals: pd.DataFrame  # 10-minute residuals of the test year only
    days_test: np.ndarray  # day code (from start) of each 10-minute row of the test year


def calibrate_year(residuals: pd.DataFrame, frames: dict[int, pd.DataFrame], test_year: int,
                   baseline: str = "fixed") -> YearCalibration:
    """Daily z-scores for calibration + test year and per (method, component, chart) thresholds from the
    calibration year.

    ``frames``: per turbine the regular 10-minute frame covering both years (for the operating mask).
    """
    start = year_bounds(test_year - 1)[0]
    end = year_bounds(test_year)[1]
    n_days = int(round((end - start) / pd.Timedelta(days=1)))
    n_cal = int(round((year_bounds(test_year)[0] - start) / pd.Timedelta(days=1)))
    days = day_codes(residuals.index, start)
    op = {t: operating(f.reindex(residuals.index)).to_numpy() for t, f in frames.items()}
    z, scale, daily = {}, {}, {}
    for col in residuals.columns:
        _, _, t = split_column(col)
        d = daily_means(residuals[col].to_numpy(dtype=float), op[t], days, n_days)
        med, mad = robust_scale(d[:n_cal])
        scale[col] = (med, mad)
        daily[col] = d
        z[col] = standardise_daily(d, med, mad, baseline)
    day_index = pd.date_range(start, periods=n_days, freq="D", name="day")
    z = pd.DataFrame(z, index=day_index)
    daily = pd.DataFrame(daily, index=day_index)
    thresholds = {}
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    for c in z.columns:
        m, comp, _ = split_column(c)
        s = z[c].to_numpy()[:n_cal]
        if np.isfinite(s).sum() > 30:
            groups.setdefault((m, comp), []).append(s)
    for (m, comp), series in groups.items():
        for chart in CHARTS:
            thresholds[(m, comp, chart)] = zero_alarm_threshold(chart, series)
    test_rows = residuals.index >= year_bounds(test_year)[0]
    return YearCalibration(test_year, baseline, start, n_cal, z, daily, scale, thresholds, residuals[test_rows],
                           days[test_rows])


def alarm_days(cal: YearCalibration, chart: str = "cusum") -> pd.DataFrame:
    """Alarms of the continuously running chart (from the start of the calibration year), test year only.

    Returns rows: day, channel, method, component, turbine, direction (+1 hot, -1 cold)."""
    rows = []
    for col in cal.z.columns:
        m, c, t = split_column(col)
        alarm, _ = run_chart(chart, cal.z[col].to_numpy(), cal.thresholds[(m, c, chart)])
        for i in np.flatnonzero(alarm[cal.n_cal_days:]) + cal.n_cal_days:
            rows.append((cal.z.index[i], col, m, c, t, int(alarm[i])))
    out = pd.DataFrame(rows, columns=["day", "channel", "method", "component", "turbine", "direction"])
    out["baseline"] = cal.baseline
    return out


def false_alarms(cal: YearCalibration, chart: str = "cusum") -> pd.DataFrame:
    """Alarm episodes per channel in the unmodified test year."""
    a = alarm_days(cal, chart)
    cols = pd.DataFrame([(c, *split_column(c)) for c in cal.z.columns],
                        columns=["channel", "method", "component", "turbine"])
    n = a.groupby("channel").size() if len(a) else pd.Series(dtype=int)
    cols["alarms"] = cols["channel"].map(n).fillna(0).astype(int)
    cols["test_year"] = cal.test_year
    cols["baseline"] = cal.baseline
    cols["chart"] = chart
    return cols


def first_alarm(alarm: np.ndarray, upward_only: bool) -> int:
    """Index of the first alarm (hot-side only if ``upward_only``), -1 if none."""
    hits = np.flatnonzero(alarm > 0) if upward_only else np.flatnonzero(alarm)
    return int(hits[0]) if len(hits) else -1


def injection_study(cal: YearCalibration, frames: dict[int, pd.DataFrame], faults=tuple(FAULTS),
                    charts=tuple(CHARTS)) -> pd.DataFrame:
    """Detection of every fault x onset x channel x chart, one row each.

    The pseudo-faults ``none_hot`` / ``none_any`` run the same windows without a fault: the chance that a chart
    raises a hot-side / any alarm within the horizon anyway."""
    idx = cal.residuals.index
    n_days = len(cal.z)
    onsets = [pd.Timestamp(f"{cal.test_year}-{m:02d}-01", tz="UTC") for m in ONSET_MONTHS]
    by_channel: dict[tuple[str, int], list[str]] = {}
    for col in cal.residuals.columns:
        _, comp, t = split_column(col)
        if np.isfinite(cal.scale[col][1]):
            by_channel.setdefault((comp, t), []).append(col)
    rows = []
    for (comp, t), cols in by_channel.items():
        f = frames[t].reindex(idx)
        op = operating(f).to_numpy()
        for fault in ("none", *faults):
            for onset in onsets:
                delta = np.zeros(len(idx)) if fault == "none" else inject(fault, f[comp], f["nacelle_temp"], onset)
                d0 = int(round((onset - cal.start) / pd.Timedelta(days=1)))
                for col in cols:
                    m = split_column(col)[0]
                    med, mad = cal.scale[col]
                    d = daily_means(cal.residuals[col].to_numpy(dtype=float) + delta, op, cal.days_test, n_days)
                    d[:cal.n_cal_days] = cal.daily[col].to_numpy()[:cal.n_cal_days]  # history for the baseline
                    window = standardise_daily(d, med, mad, cal.baseline)[d0:d0 + HORIZON_DAYS]
                    n_valid = int(np.isfinite(window).sum())
                    for chart in charts:
                        alarm, _ = run_chart(chart, window, cal.thresholds[(m, comp, chart)])
                        if fault == "none":
                            cases = [("none_hot", first_alarm(alarm, True)), ("none_any", first_alarm(alarm, False))]
                        else:
                            cases = [(fault, first_alarm(alarm, fault in UPWARD))]
                        for name, k in cases:
                            rows.append((cal.test_year, cal.baseline, name, onset, m, comp, t, chart, k >= 0,
                                         float(k) if k >= 0 else np.nan, n_valid))
    return pd.DataFrame(rows, columns=["test_year", "baseline", "fault", "onset", "method", "component", "turbine",
                                       "chart", "detected", "days_to_detect", "valid_days"])


def summarise_injections(inj: pd.DataFrame, min_valid_days: int = 30) -> pd.DataFrame:
    """Detection rate, median days to detection and chance rate per (fault, method, chart)."""
    inj = inj[inj.valid_days >= min_valid_days]
    g = inj.groupby(["baseline", "fault", "method", "chart"])
    out = pd.DataFrame({"n": g.size(), "detected_pct": 100 * g.detected.mean(),
                        "median_days": g.days_to_detect.median()})
    return out.reset_index()


# Logged Senvion status codes that point at a monitored component (Warning/Stop entries only).
EVENT_GROUPS = {
    "gear bearing high temperature": ((1700, 1710), ("front_bearing", "rear_bearing")),
    "gear oil system": ((75, 1510, 1800, 1815, 1860), ("gear_oil", "gear_inlet", "front_bearing", "rear_bearing")),
    "transformer cooling": ((3870, 3875), ("transformer",)),
    "generator cooling": ((2550, 2650, 2655, 2674), ("stator", "gen_bearing_front", "gen_bearing_rear")),
}
MERGE_DAYS = 14
CAMPAIGN_TURBINES = 3  # same group on this many turbines within a few days looks like a service visit


def event_episodes(status: pd.DataFrame, years) -> pd.DataFrame:
    """Merge logged events per turbine and group into episodes; flag farm-wide 'campaigns'."""
    rows = []
    for group, (codes, comps) in EVENT_GROUPS.items():
        ev = status[status["code"].isin(codes) & status["status"].isin(["Warning", "Stop"])]
        ev = ev[ev["start"].dt.year.isin(list(years))].sort_values("start")
        for t, e in ev.groupby("turbine"):
            cur = None
            for _, r in e.iterrows():
                end = r["end"] if pd.notna(r["end"]) else r["start"]
                if cur is not None and r["start"] - cur["end"] <= pd.Timedelta(days=MERGE_DAYS):
                    cur["end"] = max(cur["end"], end)
                    cur["n_events"] += 1
                    cur["messages"].add(r["message"])
                    continue
                if cur is not None:
                    rows.append(cur)
                cur = {"group": group, "turbine": int(t), "start": r["start"], "end": end, "n_events": 1,
                       "messages": {r["message"]}, "components": comps}
            if cur is not None:
                rows.append(cur)
    ep = pd.DataFrame(rows)
    if ep.empty:
        return ep
    ep["messages"] = ep["messages"].map(lambda s: "; ".join(sorted(s)))
    ep["campaign"] = False
    for i, r in ep.iterrows():
        near = ep[(ep.group == r.group) & ((ep.start - r.start).abs() <= pd.Timedelta(days=3))]
        ep.loc[i, "campaign"] = near.turbine.nunique() >= CAMPAIGN_TURBINES
    return ep.sort_values("start").reset_index(drop=True)


def day_number(x) -> np.ndarray:
    """Whole days since 1970-01-01 UTC for timestamps (resolution-independent)."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return np.floor(np.asarray((pd.DatetimeIndex(x) - epoch) / pd.Timedelta(days=1), dtype=float)).astype(int)


def score_events(episodes: pd.DataFrame, alarms: pd.DataFrame, days: pd.DatetimeIndex, methods) -> pd.DataFrame:
    """Per episode and method: was there a hot-side alarm on one of its channels in the HORIZON_DAYS before it,
    how many days ahead, and the chance of that for a random day (``days``: the monitored test-year days)."""
    hot = alarms[alarms.direction > 0]
    all_days = day_number(days)
    rows = []
    for _, ep in episodes.iterrows():
        d0 = int(day_number([ep.start])[0])
        for m in methods:
            a = hot[(hot.method == m) & (hot.turbine == ep.turbine) & hot.component.isin(ep.components)]
            a_days = np.unique(day_number(a.day)) if len(a) else np.array([], dtype=int)
            before = a_days[(a_days < d0) & (a_days >= d0 - HORIZON_DAYS)]
            pos = np.searchsorted(a_days, all_days, side="left")
            prev = np.where(pos > 0, a_days[np.maximum(pos - 1, 0)] if len(a_days) else 0, -10**9)
            chance = float(np.mean(all_days - prev <= HORIZON_DAYS))
            first_comp = ""
            if len(before):
                first = a[day_number(a.day) == before.min()]
                first_comp = ", ".join(sorted(first.component.unique()))
            rows.append({**ep.drop(labels=["components"]).to_dict(), "method": m, "anticipated": len(before) > 0,
                         "lead_days": float(d0 - before.min()) if len(before) else np.nan,
                         "first_alarm_component": first_comp, "chance": chance})
    return pd.DataFrame(rows)
