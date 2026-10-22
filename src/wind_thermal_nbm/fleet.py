"""Fleet reference: a turbine's temperature against the median of the other turbines at the same moment.

Weather (ambient and nacelle temperature, sun on the nacelle) is shared by all turbines of a farm, so the
difference to the median of the other producing turbines removes it without any model of it. What it does not
remove is a different load: a waked turbine runs cooler than its neighbours. That part is taken out with a quadratic in the difference of
1-hour mean power to the fleet median, fitted per turbine on the training year:

    r = (T_i - med_j T_j) - (c0 + c1 dP + c2 dP^2),    dP = P_i,1h - med_j P_j,1h  [MW]

The intercept c0 absorbs a turbine's constant offset (sensor position, calibration).

Usage:
    model = FleetModel().fit(train_frames, "stator")
    residuals = model.residuals(test_frames)      # DataFrame, one column per turbine
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .prep import median_of_others, operating


def operating_matrix(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({t: operating(f) for t, f in frames.items()})


def _differences(frames: dict[int, pd.DataFrame], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Own value minus the median of the other turbines that are producing at that moment (an idle turbine
    cools down and is no reference for a running one)."""
    op = operating_matrix(frames)
    temp = pd.DataFrame({t: f[target] for t, f in frames.items()})
    power = pd.DataFrame({t: f["power_kw"].rolling(6, min_periods=3).mean() / 1000.0 for t, f in frames.items()})
    return temp - median_of_others(temp.where(op)), power - median_of_others(power.where(op))


@dataclass
class FleetModel:
    target: str = ""
    coef: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(self, frames: dict[int, pd.DataFrame], target: str) -> "FleetModel":
        self.target = target
        d_temp, d_pow = _differences(frames, target)
        for t, f in frames.items():
            m = (operating(f) & d_temp[t].notna() & d_pow[t].notna()).to_numpy()
            if m.sum() < 500:
                continue
            dp = d_pow[t].to_numpy()[m]
            a = np.column_stack([np.ones(m.sum()), dp, dp * dp])
            self.coef[t], *_ = np.linalg.lstsq(a, d_temp[t].to_numpy()[m], rcond=None)
        return self

    def residuals(self, frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
        d_temp, d_pow = _differences(frames, self.target)
        out = {}
        for t in frames:
            if t not in self.coef:
                out[t] = pd.Series(np.nan, index=d_temp.index)
                continue
            c = self.coef[t]
            out[t] = d_temp[t] - (c[0] + c[1] * d_pow[t] + c[2] * d_pow[t] ** 2)
        return pd.DataFrame(out)
