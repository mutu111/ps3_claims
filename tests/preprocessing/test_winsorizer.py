import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, 1000)
    w = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    w.fit(X)
    X_trans = w.transform(X)
    assert X_trans.shape == X.shape
    expected_lower = np.quantile(X, lower_quantile)
    expected_upper = np.quantile(X, upper_quantile)
    assert np.isclose(w.lower_quantile_, expected_lower)
    assert np.isclose(w.upper_quantile_, expected_upper)

    assert X_trans.min() >= w.lower_quantile_
    assert X_trans.max() <= w.upper_quantile_

    if lower_quantile == 0 and upper_quantile == 1:
        assert np.allclose(X_trans, X)
    if lower_quantile == 0.5 and upper_quantile == 0.5:
        median = np.quantile(X, 0.5)
        assert np.allclose(X_trans, median)