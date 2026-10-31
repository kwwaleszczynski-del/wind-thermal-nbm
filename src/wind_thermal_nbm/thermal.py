"""First-order (RC) thermal model of a drivetrain component temperature.

The component is one thermal capacity C behind one thermal resistance R to what cools it:

    C dT/dt = Q_loss - (T - T_ref) / R        ->        tau dT/dt = T_ss - T,   tau = R C

with the steady-state temperature linear in the loss terms and the cooling-side temperatures,

    T_lin = theta_0 + sum_k theta_k T_ref,k + theta_P P + theta_P2 P^2 + theta_w w

(P in MW, w = rotor speed / rated rotor speed). Copper losses scale with P^2, gear mesh and bearing losses
with P and speed. The cooling side is the air for most components and the gear oil for the gearbox bearings
it lubricates and for the oil leaving the cooler. Over a 10-minute step the model is an exponential filter of
the steady state:

    T[t] = a T[t-1] + (1 - a) T_ss[t],    a = exp(-dt / tau),   T_ss = T_lin

The model is run as a pure simulation from the inputs (it never sees the measured temperature), so a fault in
the measurement stays in the residual instead of being followed. For a given tau the simulated temperature is
linear in theta, so the fit is a one-dimensional search over tau with least squares inside (variable
projection). tau is the thermal inertia and the temperature rise at rated power and speed, R * Q_loss(rated),
measures how well the component is cooled - but only if the cooling-side weights theta_k form a mixture
(theta_k >= 0, sum 1: the component sits at a weighted mean of the temperatures that cool it plus R * Q_loss).
Left free, the nacelle temperature, which the drivetrain itself heats, takes over part of the load term (weights
such as 2.1 and -1.1) and the rise drifts with the weights from year to year. The free fit predicts better and is
used for detection; ``mixture=True`` constrains the weights and is used for the parameter trend.

Usage:
    model = RCModel(refs=("nacelle_temp", "ambient_temp")).fit(frame, "transformer")
    predicted = model.predict(frame)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

STEP_MIN = 10.0
RATED_KW = 2050.0
RATED_RPM = 15.0  # MM92 rated rotor speed (99th percentile of the 10-minute means at Kelmarsh: 15.2 rpm)
LOSS_TERMS = ("p_mw", "p_mw2", "speed")
log = logging.getLogger(__name__)


def design_matrix(frame: pd.DataFrame, refs: tuple[str, ...]) -> np.ndarray:
    """Steady-state regressors [1, T_ref..., P, P^2, w] (rows with NaN where an input is missing)."""
    p = frame["power_kw"].to_numpy(dtype=float).clip(min=0.0) / 1000.0
    w = frame["rotor_rpm"].to_numpy(dtype=float).clip(min=0.0) / RATED_RPM
    cols = [np.ones(len(frame))] + [frame[r].to_numpy(dtype=float) for r in refs] + [p, p * p, w]
    return np.column_stack(cols)


def segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of consecutive valid rows."""
    v = np.concatenate([[False], valid, [False]]).astype(np.int8)
    d = np.diff(v)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def exp_filter(x: np.ndarray, valid: np.ndarray, a: float) -> np.ndarray:
    """y[t] = a y[t-1] + (1-a) x[t] within each valid segment, started in steady state (y = x)."""
    y = np.full_like(x, np.nan, dtype=float)
    for s, e in segments(valid):
        y[s:e] = lfilter([1.0 - a], [1.0, -a], x[s:e], axis=0, zi=a * x[s:s + 1])[0]
    return y


def warmup_mask(valid: np.ndarray, n_warmup: int) -> np.ndarray:
    """True for valid rows at least ``n_warmup`` steps after the start of their segment."""
    out = np.zeros(len(valid), dtype=bool)
    for s, e in segments(valid):
        out[s + n_warmup:e] = True
    return out


def mixture_lstsq(basis: np.ndarray, y: np.ndarray, mix: list[int]) -> np.ndarray:
    """Least squares with the coefficients of the columns ``mix`` non-negative and summing to 1.

    A convex quadratic over a simplex: its minimum is the unconstrained minimum on one face (a subset of
    ``mix`` with the weights summing to 1) whose weights come out non-negative, so every face is solved and
    the best feasible one kept. The faces are solved on the R factor of one QR decomposition.
    """
    q, r = np.linalg.qr(basis)
    qy = q.T @ y
    n = basis.shape[1]
    free = [j for j in range(n) if j not in mix]
    best, best_theta = np.inf, None
    for size in range(1, len(mix) + 1):
        for face in itertools.combinations(mix, size):
            last, rest = face[-1], list(face[:-1])
            t0 = np.zeros(n)
            t0[last] = 1.0
            cols = []
            for j in free + rest:
                c = np.zeros(n)
                c[j] = 1.0
                if j in rest:
                    c[last] = -1.0
                cols.append(c)
            t = np.column_stack(cols)
            z, *_ = np.linalg.lstsq(r @ t, qy - r @ t0, rcond=None)
            theta = t0 + t @ z
            if (theta[mix] < -1e-9).any():
                continue
            sse = float(np.sum((r @ theta - qy) ** 2))
            if sse < best:
                best, best_theta = sse, theta
    return best_theta


def fill_short_gaps(frame: pd.DataFrame, columns: list[str], limit: int) -> pd.DataFrame:
    """Linear interpolation over gaps of at most ``limit`` steps (input columns only)."""
    out = frame.copy()
    for c in columns:
        s = out[c].astype(float)
        filled = s.interpolate(limit=limit, limit_area="inside")
        # interpolate() fills the first `limit` points of longer gaps too; undo that
        gap_len = s.isna().groupby(s.notna().cumsum()).transform("sum")
        out[c] = filled.where(s.notna() | (gap_len <= limit))
    return out


@dataclass
class RCModel:
    """First-order thermal model fitted per turbine and component."""

    refs: tuple[str, ...] = ("nacelle_temp", "ambient_temp")
    tau_bounds_h: tuple[float, float] = (0.05, 12.0)
    warmup_h: float = 6.0
    gap_fill: int = 3
    mixture: bool = False  # True: cooling-side weights >= 0 and summing to 1 (physical parameters)
    tau_h: float = float("nan")
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    rmse: float = float("nan")
    target: str = ""

    @property
    def features(self) -> tuple[str, ...]:
        return ("const", *self.refs, *LOSS_TERMS)

    def _inputs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols = ["power_kw", "rotor_rpm", *self.refs]
        x = design_matrix(fill_short_gaps(frame, cols, self.gap_fill), self.refs)
        return x, np.isfinite(x).all(axis=1)

    def _n_warmup(self) -> int:
        return int(round(self.warmup_h * 60 / STEP_MIN))

    @staticmethod
    def _alpha(tau_h: float) -> float:
        return float(np.exp(-STEP_MIN / 60.0 / tau_h))

    def _simulate(self, x: np.ndarray, valid: np.ndarray, theta: np.ndarray, tau_h: float) -> np.ndarray:
        t_ss = np.where(valid[:, None], x, 0.0) @ theta
        return exp_filter(t_ss, valid, self._alpha(tau_h))

    def fit(self, frame: pd.DataFrame, target: str, mask: np.ndarray | None = None) -> "RCModel":
        """Fit on one turbine's regular 10-minute frame; ``mask`` limits the rows scored."""
        self.target = target
        x, valid = self._inputs(frame)
        y = frame[target].to_numpy(dtype=float)
        use = warmup_mask(valid, self._n_warmup()) & np.isfinite(y)
        if mask is not None:
            use &= mask
        if use.sum() < 500:
            raise ValueError(f"{target}: only {use.sum()} usable rows")
        xv = np.where(valid[:, None], x, 0.0)
        mix = list(range(1, 1 + len(self.refs))) if self.mixture else []

        def solve(log_tau: float) -> tuple[float, np.ndarray]:
            basis = exp_filter(xv, valid, self._alpha(float(np.exp(log_tau))))[use]
            if mix:
                theta = mixture_lstsq(basis, y[use], mix)
            else:
                theta, *_ = np.linalg.lstsq(basis, y[use], rcond=None)
            return float(np.mean((basis @ theta - y[use]) ** 2)), theta

        lo, hi = np.log(self.tau_bounds_h)
        grid = np.linspace(lo, hi, 15)
        i = int(np.argmin([solve(g)[0] for g in grid]))
        res = minimize_scalar(lambda g: solve(g)[0], bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]),
                              method="bounded", options={"xatol": 1e-3})
        mse, self.theta = solve(res.x)
        self.tau_h, self.rmse = float(np.exp(res.x)), float(np.sqrt(mse))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Simulated temperature; NaN where inputs are missing and during the warm-up after each gap."""
        x, valid = self._inputs(frame)
        yhat = self._simulate(x, valid, self.theta, self.tau_h)
        yhat[~warmup_mask(valid, self._n_warmup())] = np.nan
        return pd.Series(yhat, index=frame.index, name=self.target)

    def params(self) -> dict[str, float]:
        """Physical summary: time constant, rise at rated power and speed, sensitivity to the cooling side."""
        t = dict(zip(self.features, self.theta))
        p = RATED_KW / 1000.0
        out = {"tau_h": self.tau_h, "rise_rated_K": t["p_mw"] * p + t["p_mw2"] * p * p + t["speed"],
               "rmse_K": self.rmse}
        out.update({f"k_{r}": t[r] for r in self.refs})
        return out
