from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from mlc.constraints.constraints import Constraints
from mlc.constraints.constraints_checker import ConstraintChecker

from constrained_attacks.typing import NDBool, NDInt, NDNumber
from constrained_attacks.utils import compute_distance

np.set_printoptions(threshold=sys.maxsize)


@dataclass
class ObjectiveMeasure:
    misclassification: NDNumber
    distance: NDNumber
    constraints: NDNumber

    def __getitem__(self, key: int) -> ObjectiveMeasure:
        return ObjectiveMeasure(*[e[key] for e in self.__dict__.values()])


@dataclass
class ObjectiveRespected:
    misclassification: NDBool
    distance: NDBool
    constraints: NDBool
    m_and_d: NDBool
    m_and_c: NDBool
    d_and_c: NDBool
    mdc: NDBool

    def __getitem__(self, key: int) -> ObjectiveRespected:
        return ObjectiveRespected(*[e[key] for e in self.__dict__.values()])


@dataclass
class Person:
    name: str
    age: int


class ObjectiveCalculator:
    def __init__(
        self,
        classifier: Callable[[NDNumber], NDNumber],
        constraints: Constraints,
        thresholds: Dict[str, float],
        norm: str = "inf",
        fun_distance_preprocess: Callable[[NDNumber], NDNumber] = lambda x: x,
        n_jobs: int = 1,
    ) -> None:
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

        self.thresholds = thresholds.copy()

        # if isinstance(self.thresholds["misclassification"], float):
        #     self.thresholds["misclassification"] = np.array(
        #         [
        #             1 - self.thresholds["misclassification"],
        #             self.thresholds["misclassification"],
        #         ]
        #     )

        if thresholds["misclassification"] is not None:
            raise NotImplementedError(
                "misclassification threshold is not yet implemented in this version."
            )

        if "constraints" not in self.thresholds:
            self.thresholds["constraints"] = 0.0
        self.n_jobs = n_jobs
        self.objectives_eval: Optional[ObjectiveMeasure] = None
        self.objectives_respected: Optional[ObjectiveRespected] = None

    def set_cache_objectives_eval(
        self, objectives_eval: ObjectiveMeasure
    ) -> None:
        self.objectives_eval = objectives_eval

    def compute_objectives_eval(
        self, x_clean: NDNumber, y_clean: NDInt, x_adv: NDNumber
    ) -> ObjectiveMeasure:
        constraints_checker = ConstraintChecker(
            self.constraints, self.thresholds["constraints"]
        )

        constraint_violation = np.array(
            [
                1
                - constraints_checker.check_constraints(
                    x_clean[i][np.newaxis, :], x_adv[i]
                )
                for i in range(len(x_clean))
            ]
        )

        # Misclassification
        y_clean = np.repeat(y_clean, x_adv.shape[1], axis=0)
        classification = self.classifier(x_adv.reshape(-1, x_adv.shape[-1]))

        label_mask = np.zeros(classification.shape)
        label_mask[np.arange(len(y_clean)), y_clean] = 1

        correct_logit = np.max(label_mask * classification, axis=1)
        wrong_logit = np.max((1.0 - label_mask) * classification, axis=1)

        classification = correct_logit - wrong_logit

        classification = classification.reshape(*x_adv.shape[:-1])

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

        return ObjectiveMeasure(classification, distance, constraint_violation)

    def get_objectives_eval(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveMeasure:
        if self.objectives_eval is None or recompute:
            self.objectives_eval = self.compute_objectives_eval(
                x_clean, y_clean, x_adv
            )
        return self.objectives_eval

    def compute_objectives_respected(
        self, objectives_eval: ObjectiveMeasure, y_clean: NDInt
    ) -> ObjectiveRespected:

        constraints_respected = objectives_eval.constraints <= 0

        misclassified = objectives_eval.misclassification <= 0
        # # if y_clean.max() == 1 and self.thresholds.

        # if (y_clean.max() == 1) and (
        #     self.thresholds["misclassification"] is not None
        # ):
        #     y_pred = (
        #         objectives_eval.classification[..., 1]
        #         >= self.thresholds["misclassification"]
        #     ).astype(int)
        # else:
        #     y_pred = np.argmax(objectives_eval.classification, axis=-1)

        # misclassified = y_pred != y_clean[:, np.newaxis]
        distance = objectives_eval.distance <= self.thresholds["distance"]

        return ObjectiveRespected(
            misclassification=misclassified,
            distance=distance,
            constraints=constraints_respected,
            m_and_d=misclassified * distance,
            m_and_c=misclassified * constraints_respected,
            d_and_c=distance * constraints_respected,
            mdc=misclassified * distance * constraints_respected,
        )

    def get_objectives_respected(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveRespected:
        if self.objectives_respected is None or recompute:
            objectives_eval = self.get_objectives_eval(
                x_clean, y_clean, x_adv, recompute
            )
            self.objectives_respected = self.compute_objectives_respected(
                objectives_eval, y_clean
            )
        return self.objectives_respected

    def get_success_rate(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveRespected:
        objectives_respected = self.get_objectives_respected(
            x_clean, y_clean, x_adv, recompute
        )

        at_least_one = ObjectiveRespected(
            misclassification=np.max(
                objectives_respected.misclassification, axis=1
            ),
            distance=np.max(objectives_respected.distance, axis=1),
            constraints=np.max(objectives_respected.constraints, axis=1),
            m_and_c=np.max(objectives_respected.m_and_c, axis=1),
            m_and_d=np.max(objectives_respected.m_and_d, axis=1),
            d_and_c=np.max(objectives_respected.d_and_c, axis=1),
            mdc=np.max(objectives_respected.mdc, axis=1),
        )

        success_rate = [np.mean(e) for e in at_least_one.__dict__.values()]

        return ObjectiveRespected(*success_rate)

    def get_successful_attacks(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        preferred_metrics: str = "misclassification",
        order: str = "asc",
        max_inputs: int = -1,
        recompute: bool = True,
    ) -> NDNumber:

        indexes = self.get_successful_attacks_indexes(
            x_clean,
            y_clean,
            x_adv,
            preferred_metrics,
            order,
            max_inputs,
            recompute=recompute,
        )

        return x_adv[indexes]

    def get_successful_attacks_indexes(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        preferred_metrics: str = "misclassification",
        order: str = "asc",
        max_inputs: int = -1,
        recompute: bool = True,
    ) -> NDInt:
        if max_inputs == -1:
            max_inputs = x_adv.shape[1]

        objectives_measures = self.get_objectives_eval(
            x_clean, y_clean, x_adv, recompute=recompute
        )
        objectives_respected = self.get_objectives_respected(
            x_clean, y_clean, x_adv, recompute=recompute
        )
        objectives_mdc = objectives_respected.mdc

        metric = objectives_measures.__dict__[preferred_metrics]
        if order == "asc":
            metric = -metric

        indinces = select_k_best(
            metric,
            objectives_mdc,
            max_inputs,
        )

        return indinces

    def reset_objectives_respected(self) -> None:
        self.objectives_respected = None

    def reset_objectives_eval(self) -> None:
        self.objectives_eval = None


def select_k_best(metric: NDNumber, filter: NDBool, k: int) -> NDInt:
    # Find the indices of valid elements based on the filter
    valid_indices = np.where(filter)

    # Sort the valid elements along the B dimension based on the metric in ascending order
    sorted_indices = np.argsort(metric[valid_indices], axis=1)

    # Select the top k indices for each A dimension
    top_k_indices = sorted_indices[:, :k]

    # Create an index array based on valid indices
    a_indices = valid_indices[0][:, np.newaxis]
    b_indices = valid_indices[1][top_k_indices]

    # Combine the A and B indices
    indices = np.concatenate((a_indices, b_indices), axis=1)

    return indices
