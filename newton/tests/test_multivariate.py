from newton import multivariate
import numpy as np
import pytest


def test_parm_loc():
    with pytest.raises(TypeError, match='Parameters in incorrect order'):
        newton.optimize(np.array([1, 2]), np.poly1d([1, 0, 0]))


def test_basic_function():
    assert np.isclose(newton.optimize(np.poly1d([1, 0, 0]), np.array([1, 2]))['x'], np.array([0, 0]))
