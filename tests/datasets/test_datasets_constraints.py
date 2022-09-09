import numpy as np
import pytest

from constrained_attacks import datasets
from constrained_attacks.constraints.constraints_checker import (
    ConstraintChecker,
)


@pytest.mark.parametrize(
    "dataset_name, tolerance, input_proportion",
    [
        ("lcld_v2_time", 0.01, 0.001),
        ("ctu_13_neris", 0.0, 0.001),
        ("url", 0.0, 0.0),
    ],
)
def test_constraints(dataset_name, tolerance, input_proportion):
    dataset = datasets.load_dataset(dataset_name)
    x, _ = dataset.get_x_y()
    constraints_checker = ConstraintChecker(
        dataset.get_constraints(), tolerance
    )
    out = constraints_checker.check_constraints(x.to_numpy(), x.to_numpy())
    assert (1 - np.mean(out)) <= input_proportion
