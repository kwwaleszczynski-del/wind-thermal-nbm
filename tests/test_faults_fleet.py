import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_turbine
from wind_thermal_nbm.faults import inject, ramp
from wind_thermal_nbm.fleet import FleetModel
from wind_thermal_nbm.prep import median_of_others


@pytest.fixture
def series():
    idx = pd.date_range("2020-01-01", periods=144 * 60, freq="10min", tz="UTC")
    temp = pd.Series(60 + 5 * np.sin(np.arange(len(idx)) / 50), index=idx)
    nacelle = pd.Series(20.0, index=idx)
    return temp, nacelle, pd.Timestamp("2020-01-11", tz="UTC")


def test_ramp_is_zero_before_onset_and_full_after_30_days(series):
    temp, _, onset = series
    r = ramp(temp.index, onset)
    assert r[temp.index < onset].max() == 0.0
    assert r[temp.index >= onset + pd.Timedelta(days=30)].min() == 1.0
    assert r[temp.index == onset + pd.Timedelta(days=15)][0] == pytest.approx(0.5)


def test_drift_step_and_stuck(series):
    temp, nacelle, onset = series
    after = np.asarray(temp.index >= onset + pd.Timedelta(days=30))
    assert inject("drift_5", temp, nacelle, onset)[after] == pytest.approx(5.0)
    d = inject("step_3", temp, nacelle, onset)
    assert d[np.asarray(temp.index < onset)].max() == 0 and d[after].min() == 3.0
    stuck = temp.to_numpy() + inject("stuck", temp, nacelle, onset)
    on = np.asarray(temp.index >= onset)
    assert np.ptp(stuck[on]) == 0.0
    assert stuck[on][0] == temp[onset]


def test_cooling_fault_scales_the_rise_above_nacelle_air(series):
    temp, nacelle, onset = series
    d = inject("cooling_25", temp, nacelle, onset)
    after = np.asarray(temp.index >= onset + pd.Timedelta(days=30))
    np.testing.assert_allclose(d[after], 0.25 * (temp - nacelle).to_numpy()[after])


def test_median_of_others_excludes_self_and_needs_enough_turbines():
    v = pd.DataFrame({1: [1.0, 1.0], 2: [2.0, np.nan], 3: [3.0, np.nan], 4: [100.0, 4.0]})
    m = median_of_others(v, min_others=3)
    assert m.loc[0, 1] == 3.0  # median of 2, 3, 100
    assert m.loc[0, 4] == 2.0
    assert m.loc[1].isna().all()  # only one other turbine with data


def test_fleet_residual_cancels_shared_weather_and_offsets():
    frames = {}
    for t in range(1, 7):
        f, _ = synthetic_turbine(days=30, seed=10)  # same weather and load for all turbines
        f["comp"] += 0.1 * t  # constant per-turbine sensor offset
        frames[t] = f
    model = FleetModel().fit(frames, "comp")
    test = {t: f.copy() for t, f in frames.items()}
    test[3]["comp"] += 2.0  # one turbine runs hot
    r = model.residuals(test)
    assert r[3].median() == pytest.approx(2.0, abs=0.2)
    # the others move by at most one rank step of the median (0.1 K here), not by the 2 K fault
    assert r[1].abs().median() <= 0.11
