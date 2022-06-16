from abc import abstractmethod

import numpy as np

from constrained_attacks.constraints.relation_constraint import (
    AndConstraint,
    BaseRelationConstraint,
    Constant,
    ConstraintsNode,
    EqualConstraint,
    Feature,
    LessConstraint,
    LessEqualConstraint,
    MathOperation,
    OrConstraint,
)

EPS = np.array(0.000001)


def get_zeros(operands):
    i = np.argmax([op.ndim for op in operands])
    return np.zeros(operands[i].shape, dtype=operands[i].dtype)


class ConstraintsExecutor:
    """Abstract Vistor Class"""

    @abstractmethod
    def visit(self, item):
        pass

    @abstractmethod
    def execute(self, item):
        pass


str_operator_to_result = {
    "+": lambda left, right: left + right,
    "-": lambda left, right: left - right,
    "*": lambda left, right: left * right,
    "/": lambda left, right: left / right,
}


class NumpyConstraintsVisitor:
    def __init__(self, constraint: BaseRelationConstraint, x: np.ndarray):
        self.constraint = constraint
        self.x = x

    def visit(self, constrain_node: ConstraintsNode):

        # ------------ Values
        if isinstance(constrain_node, Constant):
            return np.array(constrain_node.constant)

        elif isinstance(constrain_node, Feature):
            return self.x[:, constrain_node.feature_id]

        elif isinstance(constrain_node, MathOperation):
            left_operand = constrain_node.left_operand.accept(self)
            right_operand = constrain_node.right_operand.accept(self)
            operator = constrain_node.operator
            if operator in str_operator_to_result:
                return str_operator_to_result[operator](
                    left_operand, right_operand
                )
            else:
                raise NotImplementedError

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constrain_node, OrConstraint):
            operands = [e.accept(self) for e in constrain_node.operands]
            return np.min(operands, axis=0)

        elif isinstance(constrain_node, AndConstraint):
            operands = [e.accept(self) for e in constrain_node.operands]
            return np.sum(operands, axis=0)

        # ------ Comparison
        elif isinstance(constrain_node, LessEqualConstraint):
            left_operand = constrain_node.left_operand.accept(self)
            right_operand = constrain_node.right_operand.accept(self)
            zeros = get_zeros([left_operand, right_operand])
            return np.max([zeros, (left_operand - right_operand)], axis=0)

        elif isinstance(constrain_node, LessConstraint):
            left_operand = constrain_node.left_operand.accept(self) + EPS
            right_operand = constrain_node.right_operand.accept(self)
            zeros = get_zeros([left_operand, right_operand])
            return np.max([zeros, (left_operand - right_operand)], axis=0)

        elif isinstance(constrain_node, EqualConstraint):
            left_operand = constrain_node.left_operand.accept(self)
            right_operand = constrain_node.right_operand.accept(self)
            return np.abs(left_operand - right_operand)

    def execute(self):
        return self.constraint.accept(self)


class NumpyConstraintsExecutor:
    def __init__(self, constraint: BaseRelationConstraint):
        self.constraint = constraint

    def execute(self, x: np.ndarray):
        visitor = NumpyConstraintsVisitor(self.constraint, x)
        return visitor.execute()
