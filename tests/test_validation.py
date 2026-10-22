import numpy as np
import pandas as pd
import pytest

from wind_thermal_nbm import validation
from wind_thermal_nbm.pipeline import column


def test_day_codes_do_not_depend_on_datetime_resolution():
    start = pd.Timestamp("2020-01-01", tz="UTC")
    idx = pd.date_range(start, periods=300, freq="10min", tz="UTC")
    a = validation.day_codes(idx.as_unit("ns"), start)
    b = validation.day_codes(idx.as_unit("us"), start)
    c = validation.day_codes(idx.as_unit("s"), start)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    assert a[143] == 0 and a[144] == 1


def test_daily_means_minimum_samples():
    days = np.repeat([0, 1], 144)
    vals = np.r_[np.full(144, 2.0), np.full(144, 4.0)]
    mask = np.r_[np.ones(144, bool), np.zeros(144, bool)]
    mask[150:160] = True
    out = validation.daily_means(vals, mask, days, 3)
    assert out[0] == 2.0 and np.isnan(out[1]) and np.isnan(out[2])


def _synthetic_calibration(test_year: int = 2021, seed: int = 0):
    """Two years of iid 10-minute residuals (sd 1 K) for two channels and matching operating frames."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{test_year - 1}-01-01", f"{test_year + 1}-01-01", freq="10min", tz="UTC",
                        inclusive="left")
    # daily-level noise of 0.4 K plus 10-minute noise, so the daily residual is not almost constant
    day = np.floor(np.asarray((idx - idx[0]) / pd.Timedelta(days=1))).astype(int)
    daily = rng.normal(0, 0.4, day.max() + 1)[day]
    res = pd.DataFrame({column("rc", "gear_oil", 1): daily + rng.normal(0, 1, len(idx)),
                        column("gbm", "gear_oil", 1): daily + rng.normal(0, 1, len(idx))}, index=idx)
    frame = pd.DataFrame({"power_kw": 1000.0, "rotor_rpm": 12.0, "gear_oil": 55.0, "nacelle_temp": 25.0}, index=idx)
    return res, {1: frame}


def test_calibration_standardises_the_calibration_year():
    res, frames = _synthetic_calibration()
    cal = validation.calibrate_year(res, frames, 2021)
    z_cal = cal.z.iloc[:cal.n_cal_days]
    assert z_cal.median().abs().max() < 0.1
    assert cal.n_cal_days == 366  # 2020 is a leap year
    assert set(cal.thresholds) == {(m, "gear_oil", c) for m in ("gbm", "rc") for c in ("cusum", "ewma")}


def test_injection_study_detects_large_drift_but_rarely_by_chance():
    res, frames = _synthetic_calibration()
    cal = validation.calibrate_year(res, frames, 2021)
    inj = validation.injection_study(cal, frames, faults=("drift_5", "step_3"), charts=("cusum",))
    s = inj.groupby("fault").detected.mean()
    assert s["drift_5"] == 1.0  # 5 K on a 0.4 K daily noise
    assert s["step_3"] == 1.0
    assert s["none_hot"] <= 0.25
    step = inj[inj.fault == "step_3"].days_to_detect
    assert step.max() <= 2


def test_event_episodes_merge_and_flag_campaigns():
    t0 = pd.Timestamp("2020-05-01 10:00", tz="UTC")
    rows = [
        (1, t0, t0 + pd.Timedelta(hours=1), "Warning", 1700, "High temp. gear bearing 1"),
        (1, t0 + pd.Timedelta(days=5), t0 + pd.Timedelta(days=5, hours=1), "Warning", 1700, "High temp. gear bearing 1"),
        (1, t0 + pd.Timedelta(days=60), t0 + pd.Timedelta(days=60, hours=1), "Warning", 1710, "High temp. gear bearing 2"),
    ] + [(t, t0 + pd.Timedelta(days=100), t0 + pd.Timedelta(days=100, hours=2), "Warning", 2550,
          "Overload generator fan 1") for t in (2, 3, 4)]
    status = pd.DataFrame(rows, columns=["turbine", "start", "end", "status", "code", "message"])
    ep = validation.event_episodes(status, [2020])
    gear = ep[ep.group == "gear bearing high temperature"]
    assert len(gear) == 2 and gear.n_events.tolist() == [2, 1]
    gen = ep[ep.group == "generator cooling"]
    assert len(gen) == 3 and gen.campaign.all()
    assert not gear.campaign.any()


def test_score_events_lead_time_and_chance():
    start = pd.Timestamp("2020-06-15 08:00", tz="UTC")
    ep = pd.DataFrame([{"group": "gear bearing high temperature", "turbine": 6, "start": start, "end": start,
                        "n_events": 1, "messages": "x", "components": ("front_bearing", "rear_bearing"),
                        "campaign": False}])
    alarms = pd.DataFrame({"day": [pd.Timestamp("2020-05-26", tz="UTC"), pd.Timestamp("2020-06-01", tz="UTC")],
                           "channel": "", "method": "rc", "component": ["front_bearing", "gear_oil"],
                           "turbine": 6, "direction": [1, 1]})
    days = pd.date_range("2020-01-01", "2021-01-01", freq="D", tz="UTC", inclusive="left")
    out = validation.score_events(ep, alarms, days, ["rc", "gbm"])
    rc = out[out.method == "rc"].iloc[0]
    assert rc.anticipated and rc.lead_days == 20 and rc.first_alarm_component == "front_bearing"
    # one alarm day "covers" the 60 following days out of 366
    assert rc.chance == pytest.approx(60 / 366)
    assert not out[out.method == "gbm"].iloc[0].anticipated


def test_trailing_baseline_is_causal_and_forgets_old_level_shifts():
    d = np.r_[np.zeros(200), np.full(300, 2.0)]  # model error shifts by 2 K and stays
    z_fixed = validation.standardise_daily(d, med=0.0, mad=1.0, baseline="fixed")
    z_trail = validation.standardise_daily(d, med=0.0, mad=1.0, baseline="trailing")
    assert z_fixed[-1] == 2.0
    assert z_trail[-1] == 0.0  # the shift became the new normal
    # for the first BASELINE_GAP days after the shift nothing of it is in the baseline yet
    np.testing.assert_allclose(z_trail[200:200 + validation.BASELINE_GAP], 2.0)
    # causal: changing the future does not change past z
    d2 = d.copy()
    d2[400:] = 10.0
    np.testing.assert_allclose(validation.standardise_daily(d2, 0.0, 1.0, "trailing")[:400], z_trail[:400])
