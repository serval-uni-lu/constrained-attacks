from typing import Any

import numpy as np
import numpy.typing as npt

from constrained_attacks.constraints.constraints import (
    Constraints,
    get_feature_min_max,
)
from constrained_attacks.constraints.constraints_executor import (
    NumpyConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint


class ConstraintChecker:
    def __init__(self, constraints: Constraints, tolerance: float = 0.0):
        self.constraints = constraints
        self.tolerance = tolerance

    def _check_relationship_constraints(self, x_adv: npt.NDArray[Any]):
        constraints_executor = NumpyConstraintsExecutor(
            AndConstraint(self.constraints.relation_constraints),
            feature_names=self.constraints.feature_names,
        )
        out = constraints_executor.execute(x_adv)
        return out <= self.tolerance

    def _check_boundary_constraints(self, x, x_adv):
        xl, xu = get_feature_min_max(self.constraints, x)
        xl_ok, xu_ok = np.min(
            (xl - np.finfo(np.float32).eps) <= x_adv, axis=1
        ), np.min((xu + np.finfo(np.float32).eps) >= x_adv, axis=1)
        return xl_ok * xu_ok

    def _check_type_constraints(self, x_adv):
        int_type_mask = self.constraints.feature_types != "real"
        if int_type_mask.sum() > 0:
            type_ok = np.min(
                (x_adv[:, int_type_mask] == np.round(x_adv[:, int_type_mask])),
                axis=1,
            )
        else:
            type_ok = np.ones(shape=x_adv.shape[:-1], dtype=np.bool)
        return type_ok

    def _check_mutable_constraints(self, x, x_adv):
        immutable_mask = ~self.constraints.mutable_features
        if immutable_mask.sum() > 0:
            mutable_ok = np.min(
                (x[:, immutable_mask] == x_adv[:, immutable_mask]), axis=1
            )
        else:
            mutable_ok = np.ones(shape=x_adv.shape[:-1], dtype=np.bool)
        return mutable_ok

    def check_constraints(self, x, x_adv) -> np.ndarray:
        constraints = np.array(
            [
                self._check_relationship_constraints(x_adv),
                self._check_boundary_constraints(x, x_adv),
                self._check_type_constraints(x_adv),
                self._check_mutable_constraints(x, x_adv),
            ]
        )
        constraints = np.min(constraints, axis=0)
        return constraints
