import sys

import numpy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from constrained_evasion.constraints.constraints import Constraints
from constrained_evasion.utils import compute_distance

numpy.set_printoptions(threshold=sys.maxsize)


class ObjectiveCalculator:
    def __init__(
        self,
        classifier,
        constraints: Constraints,
        thresholds: dict,
        norm=np.inf,
        fun_distance_preprocess=lambda x: x,
        n_jobs=1,
    ):
        self.classifier = classifier
        self.constraints = constraints
        self.norm = norm
        self.fun_distance_preprocess = fun_distance_preprocess

        if isinstance(thresholds["misclassification"], float):
            thresholds["misclassification"] = np.array(
                [
                    1 - thresholds["misclassification"],
                    thresholds["misclassification"],
                ]
            )
        self.thresholds = thresholds
        self.n_jobs = n_jobs

    def _calculate_objective(self, x_clean, y_clean, x_adv):

        # Constraints
        constraint_violation = 1 - self.constraints.check_constraints(
            x_clean[np.newaxis, :], x_adv
        )

        # Misclassify
        f1 = self.classifier.predict_proba(x_adv)[:, y_clean]

        # Distance
        f2 = compute_distance(
            self.fun_distance_preprocess(x_clean[np.newaxis, :]),
            self.fun_distance_preprocess(x_adv),
            self.norm,
        )

        return np.column_stack([constraint_violation, f1, f2])

    def _objective_respected(self, objective_values, y_clean):
        constraints_respected = objective_values[:, 0] <= 0
        misclassified = (
            objective_values[:, 1]
            < self.thresholds["misclassification"][y_clean]
        )
        distance = objective_values[:, 2] <= self.thresholds["distance"]
        return np.column_stack(
            [
                constraints_respected,
                misclassified,
                distance,
                constraints_respected * misclassified,
                constraints_respected * distance,
                misclassified * distance,
                constraints_respected * misclassified * distance,
            ]
        )

    def _objective_array(self, x_clean, y_clean, x_adv):
        objective_values = self._calculate_objective(x_clean, y_clean, x_adv)
        return self._objective_respected(objective_values, y_clean)

    def success_rate(self, x_clean, y_clean, x_adv):
        return self._objective_array(x_clean, y_clean, x_adv).mean(axis=0)

    def at_least_one(self, x_clean, y_clean, x_adv):
        return np.array(self.success_rate(x_clean, y_clean, x_adv) > 0)

    def success_rate_many(self, x_clean, y_clean, x_adv):
        at_least_one = np.array(
            [
                self.success_rate(x_clean[i], y_clean[i], x_adv_i) > 0
                for i, x_adv_i in tqdm(enumerate(x_adv), total=len(x_adv))
            ]
        )
        return at_least_one.mean(axis=0)

    def success_rate_many_df(self, x_clean, y_clean, x_ad):
        success_rates = self.success_rate_many(x_clean, y_clean, x_ad)

        columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
        success_rate_df = pd.DataFrame(
            success_rates.reshape([1, -1]),
            columns=columns,
        )
        return success_rate_df

    def _get_one_successful(
        self,
        x_clean,
        y_clean,
        x_adv,
        preferred_metrics="misclassification",
        order="asc",
        max_inputs=-1,
    ):

        metrics_to_index = {"misclassification": 1, "distance": 2}

        # Calculate objective and respected values
        objective_values = self._calculate_objective(x_clean, y_clean, x_adv)
        objective_respected = self._objective_respected(
            objective_values, y_clean
        )

        # Sort by the preferred_metrics parameter
        sorted_index = np.argsort(
            objective_values[:, metrics_to_index[preferred_metrics]]
        )

        # Reverse order if parameter set
        if order == "desc":
            sorted_index = sorted_index[::-1]

        # Cross the sorting with the successful attacks
        sorted_index_success = sorted_index[
            objective_respected[sorted_index, -1]
        ]

        # Bound the number of input to return
        if max_inputs > -1:
            sorted_index_success = sorted_index_success[:1]

        success_full_attacks = x_adv[sorted_index_success]

        return success_full_attacks

    def get_successful_attacks(
        self,
        x_clean,
        y_clean,
        x_adv,
        preferred_metrics="misclassification",
        order="asc",
        max_inputs=-1,
        return_index_success=False,
    ):

        successful_attacks = []

        if self.n_jobs == 1:
            for i in tqdm(range(len(x_clean)), total=len(x_clean)):
                successful_attacks.append(
                    self._get_one_successful(
                        x_clean[i],
                        y_clean[i],
                        x_adv[i],
                        preferred_metrics,
                        order,
                        max_inputs,
                    )
                )

        # Parallel run
        else:
            processed_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._get_one_successful)(
                    x_clean[i],
                    y_clean[i],
                    x_adv[i],
                    preferred_metrics,
                    order,
                    max_inputs,
                )
                for i in tqdm(range(len(x_clean)), total=len(x_clean))
            )
            for processed_result in processed_results:
                successful_attacks.append(processed_result)

        if return_index_success:
            index_success = np.array([len(e) >= 1 for e in successful_attacks])
        successful_attacks = np.concatenate(successful_attacks, axis=0)

        if return_index_success:
            return successful_attacks, index_success
        else:
            return successful_attacks
