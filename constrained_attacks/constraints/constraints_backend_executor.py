import typing
from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch

from constrained_attacks.constraints.backend import Backend
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

EPS: npt.NDArray[Any] = np.array(0.000001)


def get_feature_index(
    feature_names: Optional[npt.ArrayLike], feature_id: typing.Union[int, str]
) -> int:
    if isinstance(feature_id, str):
        if feature_names is None:
            raise ValueError(
                f"Feature names not provided. "
                f"Impossible to convert {feature_id} to index"
            )
        else:
            feature_names = np.array(feature_names)
            index = np.where(feature_names == feature_id)[0]
            if len(index) > 0:
                return index[0]
            raise IndexError(f"{feature_id} is not in {feature_names}")
    else:
        return feature_id


class ConstraintsVisitor:
    """Abstract Visitor Class"""

    @abstractmethod
    def visit(self, item: ConstraintsNode) -> Any:
        pass

    @abstractmethod
    def execute(self) -> Any:
        pass


class BackendConstraintsVisitor(ConstraintsVisitor):
    def __init__(
        self,
        constraint: BaseRelationConstraint,
        x: "torch.Tensor",
        backend: Backend,
        feature_names: npt.ArrayLike = None,
    ) -> None:
        self.constraint = constraint
        self.x = x
        self.backend = backend
        self.feature_names = feature_names

    def visit(self, constraint_node: ConstraintsNode) -> Any:

        # ------------ Values
        if isinstance(constraint_node, Constant):
            return self.backend.constant(constraint_node.constant)

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )
            return self.backend.feature(self.x, feature_index)

        elif isinstance(constraint_node, MathOperation):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            operator = constraint_node.operator
            return self.backend.math_operation(
                operator, left_operand, right_operand
            )

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)
            return self.backend.safe_division(dividend, divisor, fill_value)

        elif isinstance(constraint_node, Log):
            operand = constraint_node.operand.accept(self)
            if constraint_node.safe_value is not None:
                safe_value = constraint_node.safe_value.accept(self)
            else:
                safe_value = None

            return self.backend.log(operand, safe_value)

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]

            return self.backend.many_sum(operands)

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return self.backend.or_constraint(operands)

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return self.backend.and_constraint(operands)

        # ------ Comparison
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return self.backend.less_equal_constraint(
                left_operand, right_operand
            )

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self) + EPS
            right_operand = constraint_node.right_operand.accept(self)
            return self.backend.less_constraint(left_operand, right_operand)

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return self.backend.equal_constraint(left_operand, right_operand)

            # ------ Extension

        elif isinstance(constraint_node, Count):
            operands = [e.accept(self) for e in constraint_node.operands]
            inverse = constraint_node.inverse
            self.backend.count(operands, inverse)

        else:
            raise NotImplementedError

    def execute(self) -> "torch.Tensor":
        return self.constraint.accept(self)


class ConstraintsExecutor:
    def __init__(
        self,
        constraint: BaseRelationConstraint,
        backend: Backend,
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.backend = backend
        self.feature_names = feature_names

    def execute(self, x: "torch.Tensor") -> "torch.Tensor":
        visitor = BackendConstraintsVisitor(
            self.constraint, x, self.backend, self.feature_names
        )
        return visitor.execute()
