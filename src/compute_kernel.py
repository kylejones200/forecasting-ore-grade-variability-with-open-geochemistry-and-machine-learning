"""Rolling mean and std (expanding window within fixed width)."""

from __future__ import annotations

import numpy as np


def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    w = max(window, 1)
    means = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - w + 1)
        sl = v[start : i + 1]
        mean = float(sl.mean())
        var = float(((sl - mean) ** 2).sum() / len(sl))
        means[i] = mean
        stds[i] = var**0.5
    return means, stds
