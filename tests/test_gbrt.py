from __future__ import annotations

import numpy as np

from src.models.gbrt import GBRTModel


def test_gbrt_fits_nonlinear_signal() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (800, 6))
    y = np.sin(X[:, 0]) + 0.5 * (X[:, 1] ** 2) - 0.3 * X[:, 2] + 0.1 * rng.normal(size=800)
    model = GBRTModel(max_iter=200, max_depth=4, min_samples_leaf=20, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.85
