from typing import List, Optional, Union

import numpy as np
import torch

from constrained_attacks.constraints.backend import Backend


class PytorchBackend(Backend):
    def __init__(self, eps: float = 0.000001) -> None:
        self.eps = torch.tensor(eps)

    def get_eps(self) -> torch.Tensor:
        return self.eps

    def get_zeros(self, operands: List[torch.Tensor]) -> torch.Tensor:
        i = np.argmax([op.ndim for op in operands])
        return torch.zeros(operands[i].shape, dtype=operands[i].dtype)

    # Values
    def constant(self, value: Union[int, float]) -> torch.Tensor:
        return torch.tensor(np.array([value]))

    def feature(self, x: torch.Tensor, feature_id: int) -> torch.Tensor:
        return x[:, feature_id]

    # Math operations

    def math_operation(
        self,
        operator: str,
        left_operand: torch.Tensor,
        right_operand: torch.Tensor,
    ) -> torch.Tensor:
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
        dividend: torch.Tensor,
        divisor: torch.Tensor,
        safe_value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            divisor != 0,
            torch.div(dividend, divisor),
            safe_value,
        )

    def log(
        self, operand: torch.Tensor, safe_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if safe_value is not None:
            return torch.where(operand > 0, torch.log(operand), safe_value)
        return torch.log(operand)

    def many_sum(self, operands: List[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack(operands), dim=0)

    def less_equal_constraint(
        self, left_operand: torch.Tensor, right_operand: torch.Tensor
    ) -> torch.Tensor:

        zeros = self.get_zeros([left_operand, right_operand])
        substraction = left_operand - right_operand
        bound_zero = torch.max(torch.stack([zeros, substraction]), dim=0)[0]

        return bound_zero

    def less_constraint(
        self, left_operand: torch.Tensor, right_operand: torch.Tensor
    ) -> torch.Tensor:
        zeros = self.get_zeros([left_operand, right_operand])
        substraction = (left_operand + self.eps) - right_operand
        bound_zero = torch.max(torch.stack([zeros, substraction]), dim=0)[0]
        return bound_zero

    def equal_constraint(
        self, left_operand: torch.Tensor, right_operand: torch.Tensor
    ) -> torch.Tensor:
        return torch.abs(left_operand - right_operand)

    def or_constraint(self, operands: List[torch.Tensor]) -> torch.Tensor:
        return torch.min(torch.stack(operands), dim=0).values

    def and_constraint(self, operands: List[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack(operands), dim=0)

    def count(
        self, operands: List[torch.Tensor], inverse: bool
    ) -> torch.Tensor:
        if inverse:
            return torch.sum(
                torch.stack([(op != 0).float() for op in operands]),
                dim=0,
            )
        else:
            return torch.sum(
                torch.stack([(op == 0).float() for op in operands]),
                dim=0,
            )
