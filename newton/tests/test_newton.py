from newton import newton
import numpy as np
import math
import pytest


def test_basic_function():
    assert np.isclose(newton.optimize(np.cos, 2.95)['x'], math.pi)


def test_bad_input():
    with pytest.raises(TypeError, match='Parameters in incorrect order'):
        newton.optimize(2.95, np.cos)