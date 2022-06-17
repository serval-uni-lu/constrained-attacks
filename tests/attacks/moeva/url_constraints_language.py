from typing import List, Union

import numpy as np
import tensorflow as tf

from constrained_attacks.constraints.constraints_executor import (
    NumpyConstraintsExecutor,
    TensorFlowConstraintsExecutor,
)
from constrained_attacks.constraints.file_constraints import FileConstraints
from constrained_attacks.constraints.relation_constraint import (
    AndConstraint,
    BaseRelationConstraint,
    Constant,
    Feature,
)


class LessEqualConstraintFeature:
    pass


class UrlConstraints(FileConstraints):
    def __init__(self):
        features_path = "./tests/resources/url/features.csv"
        self.tau = 0.000001
        self.tolerance = 0.0
        super().__init__(features_path)

    def fix_features_types(self, x):
        # No implementation yet
        return x

    def get_relation_constraints(self) -> List[BaseRelationConstraint]:
        def apply_if_a_supp_zero_than_b_supp_zero(a: Feature, b: Feature):
            return (Constant(0) <= a) or (Constant(0) < b)

        g1 = Feature(1) <= Feature(0)

        intermediate_sum = Constant(0)
        for i in range(3, 18):
            intermediate_sum = intermediate_sum + Feature(i)
        intermediate_sum = intermediate_sum + (Constant(3) * Feature(19))

        g2 = intermediate_sum <= Feature(0)

        # g3: if x[:, 21] > 0 then x[:,3] > 0
        g3 = apply_if_a_supp_zero_than_b_supp_zero(Feature(21), Feature(3))

        # g4: if x[:, 23] > 0 then x[:,13] > 0
        g4 = apply_if_a_supp_zero_than_b_supp_zero(Feature(23), Feature(13))

        intermediate_sum = (
            Constant(3) * Feature(20)
            + Constant(4) * Feature(21)
            + Constant(2) * Feature(23)
        )
        g5 = intermediate_sum <= Feature(0)

        # g6: if x[:, 19] > 0 then x[:,25] > 0
        g6 = apply_if_a_supp_zero_than_b_supp_zero(Feature(19), Feature(25))

        # g7: if x[:, 19] > 0 then x[:,26] > 0
        # g7 = apply_if_a_supp_zero_than_b_supp_zero(19, 26)

        # g8: if x[:, 2] > 0 then x[:,25] > 0
        g8 = apply_if_a_supp_zero_than_b_supp_zero(Feature(2), Feature(25))

        # g9: if x[:, 2] > 0 then x[:,26] > 0
        # g9 = apply_if_a_supp_zero_than_b_supp_zero(2, 26)

        # g10: if x[:, 28] > 0 then x[:,25] > 0
        g10 = apply_if_a_supp_zero_than_b_supp_zero(Feature(28), Feature(25))

        # g11: if x[:, 31] > 0 then x[:,26] > 0
        g11 = apply_if_a_supp_zero_than_b_supp_zero(Feature(31), Feature(26))

        # x[:,38] <= x[:,37]
        g12 = Feature(38) - Feature(37)

        g13 = (Constant(3) * Feature(20)) <= (Feature(0) + Constant(1))

        g14 = (Constant(4) * Feature(21)) <= (Feature(0) + Constant(1))

        g15 = (Constant(4) * Feature(2)) <= (Feature(0) + Constant(1))
        g16 = (Constant(2) * Feature(23)) <= (Feature(0) + Constant(1))

        return [
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            g8,
            g10,
            g11,
            g12,
            g13,
            g14,
            g15,
            g16,
        ]

    def evaluate_numpy(self, x) -> np.ndarray:
        constraints = self.get_relation_constraints()
        constraints = np.column_stack(
            [NumpyConstraintsExecutor(g).execute(x) for g in constraints]
        )
        constraints[constraints <= self.tolerance] = 0.0
        return constraints

    def evaluate_tf(self, x: tf.Tensor):

        constraints = self.get_relation_constraints()
        constraints = AndConstraint(constraints)
        executor = TensorFlowConstraintsExecutor(constraints)
        return executor.execute(x)

    def evaluate(
        self, x: np.ndarray, use_tensors: bool = False
    ) -> Union[np.ndarray, tf.Tensor]:
        if use_tensors:
            return self.evaluate_tf(x)
        return self.evaluate_numpy(x)

    def get_nb_constraints(self) -> int:
        return 14
