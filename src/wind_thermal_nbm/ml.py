"""Data-driven normal behaviour model: gradient-boosted trees on current and lagged operating conditions.

This is the usual industrial NBM. It gets the same information as the RC model (power, rotor speed and the same
cooling-side temperatures), plus trailing means over 1-12 h so the trees can learn the thermal lag. The target's
own past values are deliberately not used: a model that sees the measured temperature follows a slow fault
instead of flagging it.

Usage:
    model = GBMModel().fit(frame, "gear_oil")
    predicted = model.predict(frame)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

OPERATING = ("power_kw", "rotor_rpm")
WINDOWS_H = (1, 3, 6, 12)
STEPS_PER_H = 6


def features(frame: pd.DataFrame, inputs: tuple[str, ...]) -> pd.DataFrame:
    """Current inputs plus trailing means (regular 10-minute index assumed, at least half the window present)."""
    out = {}
    for c in inputs:
        s = frame[c].astype(float)
        out[c] = s
        for w in WINDOWS_H:
            n = w * STEPS_PER_H
            out[f"{c}_{w}h"] = s.rolling(n, min_periods=n // 2).mean()
    return pd.DataFrame(out, index=frame.index)


@dataclass
class GBMModel:
    refs: tuple[str, ...] = ("nacelle_temp", "ambient_temp")
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 100
    max_train_rows: int = 60_000
    seed: int = 0
    threads: int = 2  # OpenMP oversubscription makes the fit 10x slower on a busy machine
    target: str = ""
    model: HistGradientBoostingRegressor | None = field(default=None, repr=False)

    def fit(self, frame: pd.DataFrame, target: str, mask: np.ndarray | None = None) -> "GBMModel":
        self.target = target
        x = features(frame, self.inputs)
        y = frame[target].to_numpy(dtype=float)
        use = np.isfinite(y) & x[list(self.inputs)].notna().all(axis=1).to_numpy()
        if mask is not None:
            use &= mask
        idx = np.flatnonzero(use)
        if len(idx) < 500:
            raise ValueError(f"{target}: only {len(idx)} usable rows")
        if len(idx) > self.max_train_rows:  # thin evenly; neighbouring 10-minute rows are nearly duplicates
            idx = idx[np.linspace(0, len(idx) - 1, self.max_train_rows).astype(int)]
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf, early_stopping=False, random_state=self.seed)
        with threadpool_limits(self.threads):
            self.model.fit(x.iloc[idx].to_numpy(), y[idx])
        return self

    @property
    def inputs(self) -> tuple[str, ...]:
        return (*OPERATING, *self.refs)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        x = features(frame, self.inputs)
        ok = x[list(self.inputs)].notna().all(axis=1).to_numpy()
        yhat = np.full(len(x), np.nan)
        if ok.any():
            with threadpool_limits(self.threads):
                yhat[ok] = self.model.predict(x.to_numpy()[ok])
        return pd.Series(yhat, index=frame.index, name=self.target)
