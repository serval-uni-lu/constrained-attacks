from typing import Any

import numpy as np
import pytest
import torch

from constrained_attacks.constraints.backend import Backend
from constrained_attacks.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from constrained_attacks.constraints.constraints_parser import EPS
from constrained_attacks.constraints.numpy_backend import NumpyBackend
from constrained_attacks.constraints.pytorch_backend import PytorchBackend
from constrained_attacks.constraints.relation_constraint import (
    AndConstraint,
    Constant,
    EqualConstraint,
    Feature,
    LessConstraint,
    LessEqualConstraint,
    MathOperation,
    OrConstraint,
    SafeDivision,
)
from constrained_attacks.typing import NDANumber


def to_numpy(x: Any) -> NDANumber:
    if isinstance(x, torch.Tensor):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Unknown type {type(x)}")


def to_backend(backend: Backend, x: NDANumber) -> Any:
    if isinstance(backend, PytorchBackend):
        return torch.from_numpy(x)
    if isinstance(backend, NumpyBackend):
        return x
    else:
        raise ValueError(f"Unknown type {type(backend)}")


class TestRelationConstraint:

    # ------------ Values
    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    def test_or_features(self, backend: Backend):
        x = np.arange(12).reshape(4, 3)
        feature = Feature(1)
        constant = Constant(2)
        constraint = LessEqualConstraint(feature, constant)
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, x)))
        expected = np.array([0, 2, 5, 8])
        assert np.array_equal(out, expected)

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    def test_math_operator_plus(self, backend: Backend):
        x = np.arange(12).reshape(4, 3) / 2
        feature = MathOperation("+", Feature(1), Feature(1))
        constant = Constant(2)
        constraint = LessEqualConstraint(feature, constant)
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, x)))
        expected = np.array([0, 2, 5, 8])
        assert np.array_equal(out, expected)

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    def test_safe_division(self, backend: Backend):
        x = np.array(
            [
                [1, 1, 1],
                [1, 2, 1 / 2],
                [1, 0, -1],
                [1, 4, 1 / 4],
            ]
        )
        dividend = Feature(0)
        divisor = Feature(1)
        safe_value = Constant(-1.0)
        result = Feature(2)
        constraint = SafeDivision(dividend, divisor, safe_value) == result
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, x)))
        assert np.array_equal(out, np.zeros(4))

    # ------------ Constraints

    # ------ Binary

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    def test_or_constraints(self, backend: Backend):
        a = Constant(5)
        b = Constant(4)
        c1 = LessEqualConstraint(a, b)
        c2 = LessEqualConstraint(b, a)
        constraint = OrConstraint([c1, c2])
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, np.empty(0))))
        assert np.array_equal(out, np.array([0]))

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    @pytest.mark.parametrize(
        "left_1, right_1, left_2, right_2, expected_out",
        [
            (4, 5, 4, 5, 0),
            (5, 4, 4, 5, 1),
            (5, 4, 5, 4, 2),
            (
                np.array([4, 5, 5]),
                np.array([5, 4, 4]),
                np.array([4, 4, 5]),
                np.array([5, 5, 4]),
                np.array([0, 1, 2]),
            ),
        ],
    )
    def test_and_constraints(
        self, backend: Backend, left_1, right_1, left_2, right_2, expected_out
    ):
        c1 = LessEqualConstraint(Constant(left_1), Constant(right_1))
        c2 = LessEqualConstraint(Constant(left_2), Constant(right_2))
        constraint = AndConstraint([c1, c2])
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, np.empty(0))))
        assert np.array_equal(out, np.array([expected_out]))

    # ----- Comparison

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    @pytest.mark.parametrize(
        "left_operand, right_operand, expected_out",
        [(5, 4, 1), (4, 5, 0)],
    )
    def test_less_equal_constraint(
        self, backend: Backend, left_operand, right_operand, expected_out
    ):
        left_operand = Constant(left_operand)
        right_operand = Constant(right_operand)
        constraint = LessEqualConstraint(left_operand, right_operand)
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, np.empty(0))))
        assert np.array_equal(out, np.array([expected_out]))

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    @pytest.mark.parametrize(
        "left_operand, right_operand, expected_out",
        [(4, 5, 0), (5, 5, EPS), (5, 4, 1 + EPS)],
    )
    def test_less_constraint(
        self, backend: Backend, left_operand, right_operand, expected_out
    ):
        left_operand = Constant(left_operand)
        right_operand = Constant(right_operand)
        constraint = LessConstraint(left_operand, right_operand)
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, np.empty(0))))
        assert np.allclose(out, np.array([expected_out]), atol=EPS.numpy())

    @pytest.mark.parametrize(
        "backend",
        [NumpyBackend(), PytorchBackend()],
    )
    @pytest.mark.parametrize(
        "left_operand, right_operand, expected_out",
        [(4, 4, 0), (4 + EPS, 4, EPS), (4, 4 + EPS, EPS)],
    )
    def test_equal_constraint(
        self, backend: Backend, left_operand, right_operand, expected_out
    ):
        left_operand = Constant(left_operand)
        right_operand = Constant(right_operand)
        constraint = EqualConstraint(left_operand, right_operand)
        executor = ConstraintsExecutor(constraint, backend=backend)
        out = to_numpy(executor.execute(to_backend(backend, np.empty(0))))
        assert np.allclose(out, np.array([expected_out]), atol=EPS.numpy())
