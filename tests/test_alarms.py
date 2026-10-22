import numpy as np
import pytest

from wind_thermal_nbm.alarms import cusum_run, ewma_run, run_chart, zero_alarm_threshold
from wind_thermal_nbm.validation import robust_scale


def test_cusum_detects_known_step_in_expected_time():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 400)
    z[200:] += 2.0
    alarm, _ = cusum_run(z, k=0.5, h=8.0)
    first = np.flatnonzero(alarm)
    assert first[0] >= 200  # in-control run length at h=8 is in the thousands
    # average run length for a 2-sigma shift with k=0.5, h=8 is about 6 samples
    assert first[0] - 200 <= 12
    assert alarm[first[0]] == 1


def test_cusum_detects_downward_shift_with_negative_direction():
    z = np.r_[np.zeros(50), np.full(50, -1.5)]
    alarm, _ = cusum_run(z, k=0.5, h=4.0)
    assert alarm[np.flatnonzero(alarm)[0]] == -1


def test_cusum_quiet_on_in_control_data_and_skips_nans():
    z = np.r_[np.zeros(100), np.full(20, np.nan), np.zeros(100)]
    alarm, stat = cusum_run(z, k=0.5, h=4.0)
    assert not alarm.any()
    assert stat.max() == 0.0


def test_cusum_resets_after_alarm():
    alarm, stat = cusum_run(np.full(30, 3.0), k=0.5, h=4.0)
    # increments of 2.5 per day: alarm on day 2, then the statistic starts again from zero
    np.testing.assert_array_equal(np.flatnonzero(alarm)[:3], [1, 3, 5])
    assert stat[2] == pytest.approx(2.5)


def test_ewma_detects_shift():
    z = np.r_[np.zeros(60), np.full(60, 1.5)]
    alarm, _ = ewma_run(z, lam=0.2, L=3.0)
    first = np.flatnonzero(alarm)[0]
    assert 60 <= first <= 70
    assert alarm[first] == 1


def test_zero_alarm_threshold_is_the_smallest_quiet_one():
    rng = np.random.default_rng(3)
    series = [rng.normal(0, 1, 365) for _ in range(6)]
    for chart in ("cusum", "ewma"):
        h = zero_alarm_threshold(chart, series)
        assert sum(int(np.count_nonzero(run_chart(chart, z, h)[0])) for z in series) == 0
        assert sum(int(np.count_nonzero(run_chart(chart, z, 0.98 * h)[0])) for z in series) > 0


def test_run_chart_clips_single_outliers():
    z = np.zeros(100)
    z[50] = 40.0  # one broken day
    alarm, stat = run_chart("cusum", z, threshold=5.0)
    assert not alarm.any() and stat.max() == pytest.approx(4.0)


def test_robust_scale_ignores_outliers():
    ref = np.r_[np.random.default_rng(1).normal(2.0, 0.5, 5000), [50.0, 80.0, 1e3]]
    med, sd = robust_scale(ref)
    assert med == pytest.approx(2.0, abs=0.03)
    assert sd == pytest.approx(0.5, abs=0.03)
