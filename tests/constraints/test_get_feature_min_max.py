import numpy as np

from constrained_attacks.constraints.constraints import (
    Constraints,
    get_feature_min_max,
)


def test_no_dynamic():
    xl = np.array([0, 1, 2])
    xu = np.array([3, 4, 5])
    constraints = Constraints(None, None, xl, xu, [])
    lower, upper = get_feature_min_max(constraints, None)
    assert np.array_equal(lower, xl)
    assert np.array_equal(upper, xu)


def test_dynamic():
    xl = np.array([0, "dynamic", 2])
    xu = np.array([3, 4, "dynamic"])
    constraints = Constraints(None, None, xl, xu, [])
    x = np.array([[1, 0, 9], [2, 3, 20]])
    lower, upper = get_feature_min_max(constraints, x)
    assert np.array_equal(lower, np.array([[0, 0, 2], [0, 3, 2]]))
    assert np.array_equal(upper, np.array([[3, 4, 9], [3, 4, 20]]))
