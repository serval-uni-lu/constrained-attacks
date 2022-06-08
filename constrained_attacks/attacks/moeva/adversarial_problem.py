import numpy as np
from pymoo.core.problem import Problem

from constrained_attacks.classifier.classifier import Classifier
from constrained_attacks.constraints.constraints import Constraints
from constrained_attacks.utils import compute_distance

NB_OBJECTIVES = 3


def get_nb_objectives():
    return NB_OBJECTIVES


class AdversarialProblem(Problem):
    def __init__(
        self,
        x_clean: np.ndarray,
        y_clean: int,
        classifier: Classifier,
        constraints: Constraints,
        fun_distance_preprocess=lambda x: x,
        norm=None,
    ):
        # Parameters
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.classifier = classifier
        self.constraints = constraints
        self.fun_distance_preprocess = fun_distance_preprocess
        self.norm = norm

        # Optional parameters
        self.norm = norm

        # Caching
        self.x_clean_distance = self.fun_distance_preprocess(
            x_clean.reshape(1, -1)
        )[0]

        # Computed attributes
        xl, xu = constraints.get_feature_min_max()
        xl, xu = (
            xl[constraints.get_mutable_mask()],
            xu[self.constraints.get_mutable_mask()],
        )

        super().__init__(
            n_var=self.constraints.get_mutable_mask().sum(),
            n_obj=get_nb_objectives(),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def get_x_clean(self):
        return self.x_clean

    def _obj_misclassify(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.classifier.predict_proba(x)[:, self.y_clean]
        return y_pred

    def _obj_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        return compute_distance(x_1, x_2, self.norm)

    def _calculate_constraints(self, x):
        G = self.constraints.evaluate(x)
        G = G * (G > 0).astype(np.float)

        return G

    def _evaluate(self, x, out, *args, **kwargs):

        # print("Evaluate")

        # Sanity check
        if (x - self.xl < 0).sum() > 0:
            print("Lower than lower bound.")

        if (x - self.xu > 0).sum() > 0:
            print("Lower than lower bound.")

        # --- Prepare necessary representation of the samples

        # Retrieve original representation

        x_adv = np.repeat(self.x_clean.reshape(1, -1), x.shape[0], axis=0)
        x_adv[:, self.constraints.get_mutable_mask()] = x

        obj_misclassify = self._obj_misclassify(x_adv)

        obj_distance = self._obj_distance(
            self.fun_distance_preprocess(x_adv), self.x_clean_distance
        )

        all_constraints = self._calculate_constraints(x_adv)
        obj_constraints = all_constraints.sum(axis=1)

        F = [obj_misclassify, obj_distance, obj_constraints]

        # --- Output
        out["F"] = np.column_stack(F)
