from newton import multivariate
import numpy as np
import pytest


def test_parm_loc():
    with pytest.raises(TypeError, match='Parameters in incorrect order'):
        multivariate.optimize(np.array([1, 2]), np.poly1d([1, 0, 0]))


def test_basic_function():
    poly = np.poly1d([1, 0, 0])
    objective_function = lambda x: np.sum(poly(x), axis=0, keepdims=True)
    assert np.all(np.isclose(multivariate.optimize(objective_function, np.array([1.0, 2.0])), np.array([0, 0])))
