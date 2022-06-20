import numpy as np
import pytest

from constrained_attacks.constraints.constraints_executor import (
    EPS,
    NumpyConstraintsExecutor,
)
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


class TestRelationConstraint:
    def test_object_construction(self):
        a = Constant(5)
        b = Constant(4)
        LessEqualConstraint(a, b)
        assert True

    # ------------ Values

    def test_or_features(self):
        x = np.arange(12).reshape(4, 3)
        feature = Feature(1)
        constant = Constant(2)
        constraint = LessEqualConstraint(feature, constant)
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(x)
        expected = np.array([0, 2, 5, 8])
        assert np.array_equal(out, expected)

    def test_math_operator_plus(self):
        x = np.arange(12).reshape(4, 3) / 2
        feature = MathOperation("+", Feature(1), Feature(1))
        constant = Constant(2)
        constraint = LessEqualConstraint(feature, constant)
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(x)
        expected = np.array([0, 2, 5, 8])
        assert np.array_equal(out, expected)

    def test_safe_division(self):
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
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(x)
        assert np.array_equal(out, np.zeros(4))

    # ------------ Constraints

    # ------ Binary

    def test_or_constraints(self):
        a = Constant(5)
        b = Constant(4)
        c1 = LessEqualConstraint(a, b)
        c2 = LessEqualConstraint(b, a)
        constraint = OrConstraint([c1, c2])
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(np.empty(0))
        assert np.array_equal(out, 0)

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
        self, left_1, right_1, left_2, right_2, expected_out
    ):
        c1 = LessEqualConstraint(Constant(left_1), Constant(right_1))
        c2 = LessEqualConstraint(Constant(left_2), Constant(right_2))
        constraint = AndConstraint([c1, c2])
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(np.empty(0))
        assert np.array_equal(out, expected_out)

    # ----- Comparison

    @pytest.mark.parametrize(
        "left_operand, right_operand, expected_out",
        [(5, 4, 1), (4, 5, 0)],
    )
    def test_less_equal_constraint(
        self, left_operand, right_operand, expected_out
    ):
        left_operand = Constant(left_operand)
        right_operand = Constant(right_operand)
        constraint = LessEqualConstraint(left_operand, right_operand)
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(np.empty(0))
        assert np.array_equal(out, expected_out)

    @pytest.mark.parametrize(
        "left_operand, right_operand, expected_out",
        [(4, 5, 0), (5, 5, EPS), (5, 4, 1 + EPS)],
    )
    def test_less_constraint(self, left_operand, right_operand, expected_out):
        left_operand = Constant(left_operand)
        right_operand = Constant(right_operand)
        constraint = LessConstraint(left_operand, right_operand)
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(np.empty(0))
        assert np.allclose(out, expected_out, atol=EPS)

    @pytest.mark.parametrize(
        "left_operand, right_operand, expected_out",
        [(4, 4, 0), (4 + EPS, 4, EPS), (4, 4 + EPS, EPS)],
    )
    def test_equal_constraint(self, left_operand, right_operand, expected_out):
        left_operand = Constant(left_operand)
        right_operand = Constant(right_operand)
        constraint = EqualConstraint(left_operand, right_operand)
        executor = NumpyConstraintsExecutor(constraint)
        out = executor.execute(np.empty(0))
        assert np.allclose(out, expected_out, atol=EPS)
