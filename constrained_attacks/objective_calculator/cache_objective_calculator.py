import sys

import numpy
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from constrained_attacks.constraints.constraints import Constraints
from constrained_attacks.utils import compute_distance

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
        """Calculate the objectives satisfaction according to a model
        and a set of constraints.
        This version is using cache, therefore you should pass the
        parameters recompute=True whenever possible if your input
        change.


        Parameters
        ----------
        classifier : _type_
            The tager classifier.
        constraints : Constraints
            The set of constraints.
        thresholds : dict
            Dictionary containing a float value for the
            "misclassfication" and  "distance" key.
        norm : _type_, optional
            Norm to compute the distance, by default np.inf.
        fun_distance_preprocess : _type_, optional
            function used to preprocess input before the distance metric
            calculation, typically the n-1 first steps of an n step
            classification Pipeline, by default lambdax:x.
        n_jobs : int, optional
            Number of parallel jobs for returning adversarial examples,
            we recommand using the classifier parallel capabilities
            instead, by default 1.
        """
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
        self.objectives_eval = None
        self.objectives_respected = None

    def set_cache_objectives_eval(self, objectives_eval):
        self.objectives_eval = objectives_eval

    def compute_objectives_eval(self, x_clean, y_clean, x_adv):
        constraint_violation = np.array(
            [
                1
                - self.constraints.check_constraints(
                    x_clean[i][np.newaxis, :], x_adv[i]
                )
                for i in range(len(x_clean))
            ]
        )

        classification = self.classifier.predict_proba(
            x_adv.reshape(-1, x_adv.shape[-1])
        )
        classification = classification[
            np.arange(classification.shape[0]),
            np.repeat(y_clean, x_adv.shape[1]),
        ]
        classification = classification.reshape(x_adv.shape[:-1])

        x_clean_distance = self.fun_distance_preprocess(x_clean)
        x_adv_shape = x_adv.shape
        x_adv_distance = self.fun_distance_preprocess(
            x_adv.reshape(-1, x_adv.shape[-1])
        ).reshape((*x_adv_shape[:-1], -1))
        distance = np.array(
            [
                compute_distance(
                    x_clean_distance[i][np.newaxis, :],
                    x_adv_distance[i],
                    self.norm,
                )
                for i in range(len(x_clean))
            ]
        )

        return np.array([constraint_violation, classification, distance])

    def get_objectives_eval(self, x_clean, y_clean, x_adv, recompute=False):
        if self.objectives_eval is None or recompute:
            self.objectives_eval = self.compute_objectives_eval(
                x_clean, y_clean, x_adv
            )
        return self.objectives_eval

    def compute_objectives_respected(self, objectives_eval, y_clean):
        constraints_respected = objectives_eval[0] <= 0
        misclassified = np.array(
            [
                (
                    objectives_eval[1][i]
                    < self.thresholds["misclassification"][y_clean[i]]
                )
                for i in range(len(y_clean))
            ]
        )
        distance = objectives_eval[2] <= self.thresholds["distance"]
        return np.array(
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

    def get_objectives_respected(
        self, x_clean, y_clean, x_adv, recompute=False
    ):
        if self.objectives_respected is None or recompute:
            objectives_eval = self.get_objectives_eval(
                x_clean, y_clean, x_adv, recompute=False
            )
            self.objectives_respected = self.compute_objectives_respected(
                objectives_eval, y_clean
            )
        return self.objectives_respected

    def get_success_rate(self, x_clean, y_clean, x_adv, recompute=False):
        objectives_respected = self.get_objectives_respected(
            x_clean, y_clean, x_adv, recompute
        )
        at_least_one_objectives_respected = np.max(
            objectives_respected, axis=2
        )
        success_rate = np.mean(at_least_one_objectives_respected, axis=1)
        return success_rate

    def _get_one_successful(
        self,
        x_clean,
        y_clean,
        x_adv,
        objective_values,
        objective_respected,
        preferred_metrics="misclassification",
        order="asc",
        max_inputs=-1,
    ):

        metrics_to_index = {"misclassification": 1, "distance": 2}

        # Sort by the preferred_metrics parameter
        sorted_index = np.argsort(
            objective_values[metrics_to_index[preferred_metrics], :]
        )

        # Reverse order if parameter set
        if order == "desc":
            sorted_index = sorted_index[::-1]

        # Cross the sorting with the successful attacks
        sorted_index_success = sorted_index[
            objective_respected[-1][sorted_index]
        ]

        # Bound the number of input to return
        if max_inputs > -1:
            sorted_index_success = sorted_index_success[:max_inputs]

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
        recompute=False,
    ):

        successful_attacks = []

        objectives_values = self.get_objectives_eval(
            x_clean, y_clean, x_adv, recompute=recompute
        )
        objectives_respected = self.get_objectives_respected(
            x_clean, y_clean, x_adv, recompute=recompute
        )

        if self.n_jobs == 1:
            for i in tqdm(range(len(x_clean)), total=len(x_clean)):
                successful_attacks.append(
                    self._get_one_successful(
                        x_clean[i],
                        y_clean[i],
                        x_adv[i],
                        objectives_values[:, i, :],
                        objectives_respected[:, i, :],
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
                    objectives_values[i],
                    objectives_respected[i],
                    preferred_metrics,
                    order,
                    max_inputs,
                )
                for i in tqdm(range(len(x_clean)), total=len(x_clean))
            )
            for processed_result in processed_results:
                successful_attacks.append(processed_result)

        if return_index_success:
            index_success = [
                np.array([i for _ in es])
                for i, es in enumerate(successful_attacks)
            ]

            index_success = np.concatenate(index_success)
        successful_attacks = np.concatenate(successful_attacks, axis=0)

        if return_index_success:
            return successful_attacks, index_success
        else:
            return successful_attacks
