import numpy as np
import pandas as pd
import pytest

from wind_thermal_nbm.thermal import RATED_RPM, exp_filter


def synthetic_turbine(days: int = 60, seed: int = 0, tau_h: float = 1.5, rise_k: float = 30.0,
                      unit: str = "ns") -> tuple[pd.DataFrame, dict]:
    """10-minute frame with power, rotor speed, nacelle/ambient temperature and a component that follows a
    known first-order thermal model (plus 0.3 K noise)."""
    rng = np.random.default_rng(seed)
    n = days * 144
    idx = pd.date_range("2021-01-01", periods=n, freq="10min", tz="UTC", name="timestamp").as_unit(unit)
    # wind-driven power: smoothed random walk squashed into 0..2050 kW
    walk = np.cumsum(rng.normal(0, 0.08, n))
    walk = pd.Series(walk).rolling(12, min_periods=1).mean().to_numpy()
    p_kw = 2050 / (1 + np.exp(-2 * (walk - walk.mean())))
    rpm = np.where(p_kw > 50, 9 + 6 * np.sqrt(p_kw / 2050), 0.0)
    hours = np.arange(n) / 6
    ambient = 10 + 5 * np.sin(2 * np.pi * hours / 24) + 3 * np.sin(2 * np.pi * hours / (24 * 9))
    # nacelle air independent of power here, so the reference and loss terms are identifiable
    nacelle = ambient + 6 + 2 * np.sin(2 * np.pi * hours / (24 * 3.7) + 1.0)
    p = p_kw / 1000
    truth = {"tau_h": tau_h, "const": 5.0, "nacelle_temp": 0.8, "ambient_temp": 0.2,
             "p_mw": rise_k * 0.3 / 2.05, "p_mw2": rise_k * 0.7 / 2.05 ** 2, "speed": 0.0}
    t_ss = (truth["const"] + truth["nacelle_temp"] * nacelle + truth["ambient_temp"] * ambient
            + truth["p_mw"] * p + truth["p_mw2"] * p * p + truth["speed"] * rpm / RATED_RPM)
    a = np.exp(-10 / 60 / tau_h)
    temp = exp_filter(t_ss[:, None], np.ones(n, bool), a)[:, 0] + rng.normal(0, 0.3, n)
    frame = pd.DataFrame({"power_kw": p_kw, "rotor_rpm": rpm, "nacelle_temp": nacelle, "ambient_temp": ambient,
                          "comp": temp}, index=idx)
    return frame, truth


@pytest.fixture
def turbine():
    return synthetic_turbine()
