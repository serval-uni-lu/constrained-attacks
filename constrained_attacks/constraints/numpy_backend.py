from typing import List, Optional, Union

import numpy as np

from constrained_attacks.constraints.backend import Backend
from constrained_attacks.typing import NDANumber


class NumpyBackend(Backend):
    def __init__(self, eps: float = 0.000001) -> None:
        self.eps = np.array(eps)

    def get_eps(self) -> NDANumber:
        return self.eps

    def get_zeros(self, operands: List[NDANumber]) -> NDANumber:
        i = np.argmax([op.ndim for op in operands])
        return np.zeros(operands[i].shape, dtype=operands[i].dtype)

    # Values
    def constant(self, value: Union[int, float]) -> NDANumber:
        return np.array([value])

    def feature(self, x: NDANumber, feature_id: int) -> NDANumber:
        return x[:, feature_id]

    # Math operations

    def math_operation(
        self,
        operator: str,
        left_operand: NDANumber,
        right_operand: NDANumber,
    ) -> NDANumber:
        if operator == "+":
            return left_operand + right_operand
        elif operator == "-":
            return left_operand - right_operand
        elif operator == "*":
            return left_operand * right_operand
        elif operator == "/":
            return left_operand / right_operand
        elif operator == "**":
            return left_operand**right_operand
        elif operator == "%":
            return left_operand % right_operand
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def safe_division(
        self,
        dividend: NDANumber,
        divisor: NDANumber,
        safe_value: NDANumber,
    ) -> NDANumber:
        return np.divide(
            dividend,
            divisor,
            out=np.full_like(dividend, safe_value),
            where=divisor != 0,
        )

    def log(
        self, operand: NDANumber, safe_value: Optional[NDANumber] = None
    ) -> NDANumber:
        if safe_value is not None:
            return np.log(
                operand,
                out=np.full_like(operand, fill_value=safe_value),
                where=(operand > 0),
            )
        return np.log(operand)

    def many_sum(self, operands: List[NDANumber]) -> NDANumber:
        return np.sum(operands, axis=0)

    def less_equal_constraint(
        self, left_operand: NDANumber, right_operand: NDANumber
    ) -> NDANumber:

        zeros = self.get_zeros([left_operand, right_operand])
        substraction = left_operand - right_operand
        bound_zero = np.max(np.stack([zeros, substraction]), axis=0)

        return bound_zero

    def less_constraint(
        self, left_operand: NDANumber, right_operand: NDANumber
    ) -> NDANumber:
        zeros = self.get_zeros([left_operand, right_operand])
        substraction = (left_operand + self.eps) - right_operand
        bound_zero = np.max(np.stack([zeros, substraction]), axis=0)
        return bound_zero

    def equal_constraint(
        self, left_operand: NDANumber, right_operand: NDANumber
    ) -> NDANumber:
        return np.abs(left_operand - right_operand)

    def or_constraint(self, operands: List[NDANumber]) -> NDANumber:
        return np.min(np.stack(operands), axis=0)

    def and_constraint(self, operands: List[NDANumber]) -> NDANumber:
        return np.sum(np.stack(operands), axis=0)

    def count(self, operands: List[NDANumber], inverse: bool) -> NDANumber:
        if inverse:
            return np.sum(
                np.stack([(op != 0).astype(float) for op in operands]),
                axis=0,
            )
        else:
            return np.sum(
                np.stack([(op == 0).astype(float) for op in operands]),
                axis=0,
            )
