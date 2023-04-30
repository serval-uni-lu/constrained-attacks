import typing
from abc import abstractmethod
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import torch

from constrained_attacks.constraints.relation_constraint import (
    AndConstraint,
    BaseRelationConstraint,
    Constant,
    ConstraintsNode,
    Count,
    EqualConstraint,
    Feature,
    LessConstraint,
    LessEqualConstraint,
    Log,
    ManySum,
    MathOperation,
    OrConstraint,
    SafeDivision,
)

EPS: torch.Tensor = torch.tensor(0.000001)


def get_feature_index(
    feature_names: npt.ArrayLike, feature_id: typing.Union[int, str]
) -> int:

    if isinstance(feature_id, int):
        return feature_id

    if isinstance(feature_id, str):
        if feature_names is None:
            raise ValueError(
                f"Feature names not provided. "
                f"Impossible to convert {feature_id} to index"
            )

        feature_names = np.array(feature_names)
        index = np.where(feature_names == feature_id)[0]

        if len(index) <= 0:
            raise IndexError(f"{feature_id} is not in {feature_names}")

        return index[0]

    raise NotImplementedError


class ConstraintsVisitor:
    """Abstract Visitor Class"""

    @abstractmethod
    def visit(self, item: ConstraintsNode) -> Any:
        pass

    @abstractmethod
    def execute(self) -> Any:
        pass


class PytorchConstraintsVisitor(ConstraintsVisitor):

    str_operator_to_result = {
        "+": lambda left, right: left + right,
        "-": lambda left, right: left - right,
        "*": lambda left, right: left * right,
        "/": lambda left, right: left / right,
        "**": lambda left, right: left**right,
        "%": lambda left, right: left % right,
    }

    def __init__(
        self,
        constraint: BaseRelationConstraint,
        feature_names: npt.ArrayLike = None,
    ) -> None:
        self.constraint = constraint
        self.feature_names = feature_names

    @staticmethod
    def get_zeros_torch(
        operands: typing.List["torch.Tensor"],
    ) -> "torch.Tensor":
        i = np.argmax([op.ndim for op in operands])
        return torch.zeros(operands[i].shape, dtype=operands[i].dtype)

    def visit(self, constraint_node: ConstraintsNode) -> "torch.Tensor":

        # ------------ Values
        if isinstance(constraint_node, Constant):

            def process(x):
                return torch.tensor([constraint_node.constant])

            return process

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )

            def process(x):
                return x[:, feature_index]

            return process

        elif isinstance(constraint_node, MathOperation):
            operator = constraint_node.operator
            if not (operator in self.str_operator_to_result):
                raise NotImplementedError

            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            operator_function = self.str_operator_to_result[operator]

            def process(x):
                return operator_function(left_operand(x), right_operand(x))

            return process

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)

            def process(x):
                return torch.where(
                    divisor(x) != 0,
                    torch.div(dividend(x), divisor(x)),
                    fill_value(x),
                )

            return process

        elif isinstance(constraint_node, Log):
            operand = constraint_node.operand.accept(self)
            if constraint_node.safe_value is not None:
                safe_value = constraint_node.safe_value.accept(self)

                def process(x):
                    return torch.where(
                        operand(x) > 0, torch.log(operand(x)), safe_value(x)
                    )

                return process

            def process(x):
                return torch.log(operand(x))

            return process

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]

            def process(x):
                return torch.sum(
                    torch.stack([op(x) for op in operands]), dim=0
                )

            return process

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]

            def process(x):
                return torch.min(
                    torch.stack([op(x) for op in operands]), dim=0
                ).values

            return process

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]

            def process(x):
                return torch.sum(
                    torch.stack([op(x) for op in operands]), dim=0
                )

            return process

        # ------ Comparison TODO: continue here
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            def proccess(x):
                zeros = self.get_zeros_torch(
                    [left_operand(x), right_operand(x)]
                )
                substraction = left_operand(x) - right_operand(x)
                bound_zero = torch.max(
                    torch.stack([zeros, substraction]), dim=0
                )[0]
                return bound_zero

            return proccess

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            def proccess(x):
                zeros = self.get_zeros_torch(
                    [left_operand(x), right_operand(x)]
                )
                substraction = (left_operand(x) + EPS) - right_operand(x)
                bound_zero = torch.max(
                    torch.stack([zeros, substraction]), dim=0
                )[0]
                return bound_zero

            return proccess

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            def proccess(x):
                return torch.abs(left_operand(x) - right_operand(x))

            return proccess

        # ------ Extension

        elif isinstance(constraint_node, Count):
            operands = [e.accept(self) for e in constraint_node.operands]
            if constraint_node.inverse:

                def process(x):
                    return torch.sum(
                        torch.stack([(op(x) != 0).float() for op in operands]),
                        dim=0,
                    )

                return proccess
            else:

                def process(x):
                    return torch.sum(
                        torch.stack([(op(x) == 0).float() for op in operands]),
                        dim=0,
                    )

                return proccess

        else:
            raise NotImplementedError

    def execute(self) -> "torch.Tensor":
        return self.constraint.accept(self)


class PytorchConstraintsParser:
    def __init__(
        self,
        constraint: BaseRelationConstraint,
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.feature_names = feature_names
        self.process = None

    def translate(self) -> Callable[[], "torch.Tensor"]:
        if self.process is None:
            visitor = PytorchConstraintsVisitor(
                self.constraint, self.feature_names
            )
            self.process = visitor.execute()

        return self.process

    def execute(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.translate()(x)
