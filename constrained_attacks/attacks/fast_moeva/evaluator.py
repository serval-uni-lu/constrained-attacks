import numpy as np

from constrained_attacks.classifier.classifier import Classifier
from constrained_attacks.constraints.constraints import Constraints
from constrained_attacks.constraints.constraints_executor import (
    NumpyConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint
from constrained_attacks.utils import compute_distance


class Evaluator:
    def __init__(
        self,
        x_clean: np.ndarray,
        y_clean: np.ndarray,
        classifier: Classifier,
        constraints: Constraints,
        fun_distance_preprocess=lambda x: x,
        norm=None,
    ):
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.classifier = classifier
        self.constraints = constraints
        self.fun_distance_preprocess = fun_distance_preprocess
        self.norm = norm

        # Optional parameters
        self.norm = norm

        # Caching
        self.x_clean_distance = self.fun_distance_preprocess(x_clean)

    def _obj_misclassify(
        self, x: np.ndarray, n_inputs_per_clean
    ) -> np.ndarray:
        y_clean = np.repeat(self.y_clean, n_inputs_per_clean, axis=0)
        y_pred = self.classifier.predict_proba(x)[np.arange(len(x)), y_clean]
        return y_pred

    def _obj_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        return compute_distance(x_1, x_2, self.norm)

    def _calculate_constraints(self, x):
        executor = NumpyConstraintsExecutor(
            AndConstraint(self.constraints.relation_constraints),
            feature_names=self.constraints.feature_names,
        )

        return executor.execute(x)

    def evaluate(self, x):
        n_inputs_per_clean = x.shape[0] / self.x_clean.shape[0]
        x_adv = np.repeat(self.x_clean, n_inputs_per_clean, axis=0)
        x_adv[:, self.constraints.mutable_features] = x

        obj_misclassify = self._obj_misclassify(x_adv, n_inputs_per_clean)

        obj_distance = self._obj_distance(
            self.fun_distance_preprocess(x_adv),
            np.repeat(self.x_clean_distance, n_inputs_per_clean, axis=0),
        )

        obj_constraints = self._calculate_constraints(x_adv)

        f = [obj_misclassify, obj_distance, obj_constraints]

        return np.column_stack(f)
