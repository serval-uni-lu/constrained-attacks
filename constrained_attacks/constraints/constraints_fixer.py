from typing import List

import numpy as np

from constrained_attacks.constraints.constraints_executor import (
    NumpyConstraintsExecutor,
    get_feature_index,
)
from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
    EqualConstraint,
    Feature,
)


class ConstraintsFixer:
    def __init__(
        self,
        guard_constraints: List[BaseRelationConstraint],
        fix_constraints: List[EqualConstraint],
        feature_names=None,
    ):
        self.guard_constraints = guard_constraints
        self.fix_constraints = fix_constraints
        self.feature_names = feature_names
        for c in self.fix_constraints:
            if not isinstance(c, EqualConstraint):
                raise ValueError("Fix constraints must be an EqualConstraint.")
            else:
                if not isinstance(c.left_operand, Feature):
                    raise ValueError(
                        "Left operand of fix constraints must be a Feature."
                    )

    def fix(self, x):
        x = x.copy()
        for i in range(len(self.fix_constraints)):
            guard_c = self.guard_constraints[i]
            fix_c = self.fix_constraints[i]

            # Index of inputs that shall be updated
            # according the guard constraints,
            # if none then update all.
            if guard_c is not None:
                executor = NumpyConstraintsExecutor(
                    guard_c, self.feature_names
                )
                to_update = executor.execute(x) > 0
            else:
                to_update = np.ones(x.shape[0]).astype(np.bool)

            # Index of the feature to update.
            # Ignore warning, this is checked in the constructor.
            index = get_feature_index(
                self.feature_names, fix_c.left_operand.feature_id
            )

            # Value to be update.
            # Known warning, not supposed to evaluate without constraint.
            executor = NumpyConstraintsExecutor(
                fix_c.right_operand, self.feature_names
            )
            new_value = executor.execute(x[to_update])

            x[to_update, index] = new_value

            return x
