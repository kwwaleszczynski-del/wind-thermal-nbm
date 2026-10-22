import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_turbine
from wind_thermal_nbm.thermal import RCModel, exp_filter, fill_short_gaps, segments, warmup_mask


def test_exp_filter_starts_in_steady_state_and_converges():
    x = np.r_[np.full(5, 10.0), np.full(200, 20.0)]
    y = exp_filter(x, np.ones(len(x), bool), a=0.9)
    assert y[0] == pytest.approx(10.0)
    assert y[5] == pytest.approx(10.0 + 0.1 * 10.0)
    assert y[-1] == pytest.approx(20.0, abs=1e-6)


def test_exp_filter_restarts_after_gap():
    x = np.r_[np.full(50, 10.0), [np.nan], np.full(50, 30.0)]
    valid = np.isfinite(x)
    y = exp_filter(np.nan_to_num(x), valid, a=0.95)
    assert np.isnan(y[50])
    assert y[51] == pytest.approx(30.0)  # new segment starts at its own steady state, not from 10 degC


def test_segments_and_warmup():
    valid = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    assert segments(valid) == [(0, 3), (4, 8)]
    np.testing.assert_array_equal(warmup_mask(valid, 2), [0, 0, 1, 0, 0, 0, 1, 1])


def test_fill_short_gaps_only_fills_short_ones():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0])
    out = fill_short_gaps(pd.DataFrame({"x": s}), ["x"], limit=3)["x"]
    assert out[1] == pytest.approx(2.0)
    assert out[3:7].isna().all()  # a 4-step gap stays a gap


def test_rc_recovers_known_parameters(turbine):
    frame, truth = turbine
    m = RCModel().fit(frame, "comp")
    p = m.params()
    assert m.tau_h == pytest.approx(truth["tau_h"], rel=0.05)
    rise_true = truth["p_mw"] * 2.05 + truth["p_mw2"] * 2.05 ** 2
    assert p["rise_rated_K"] == pytest.approx(rise_true, rel=0.05)
    assert p["k_nacelle_temp"] + p["k_ambient_temp"] == pytest.approx(1.0, abs=0.05)
    assert m.rmse == pytest.approx(0.3, abs=0.05)  # only the measurement noise is left


def test_rc_prediction_does_not_follow_a_sensor_offset():
    frame, _ = synthetic_turbine(days=40, seed=1)
    m = RCModel().fit(frame, "comp")
    faulty = frame.copy()
    faulty.loc[faulty.index[3000]:, "comp"] += 2.0
    r = faulty["comp"] - m.predict(faulty)
    assert r.iloc[3200:].mean() == pytest.approx(2.0, abs=0.15)  # simulation, not one-step-ahead


def test_rc_result_independent_of_datetime_resolution():
    f_ns, _ = synthetic_turbine(days=20, seed=2, unit="ns")
    f_us, _ = synthetic_turbine(days=20, seed=2, unit="us")
    a = RCModel().fit(f_ns, "comp")
    b = RCModel().fit(f_us, "comp")
    assert a.tau_h == pytest.approx(b.tau_h)
    np.testing.assert_allclose(a.predict(f_ns).to_numpy(), b.predict(f_us).to_numpy(), equal_nan=True)


def test_rc_refuses_too_little_data(turbine):
    frame, _ = turbine
    with pytest.raises(ValueError):
        RCModel().fit(frame.iloc[:200], "comp")
