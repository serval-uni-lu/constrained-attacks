import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from mlc.constraints.constraints import Constraints
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_reference_directions,
    get_termination,
)
from pymoo.operators.mixed_variable_operator import (
    MixedVariableCrossover,
    MixedVariableMutation,
)
from pymoo.optimize import minimize
from tqdm import tqdm

from constrained_attacks.attacks.moeva.history_callback import HistoryCallback
from constrained_attacks.attacks.moeva.operators import InitialStateSampling
from constrained_attacks.utils import cut_in_batch

from .adversarial_problem import NB_OBJECTIVES, AdversarialProblem


def tf_lof_off():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


class Moeva2:
    def __init__(
        self,
        classifier_class,
        constraints: Constraints,
        norm=None,
        fun_distance_preprocess=lambda x: x,
        n_gen=100,
        n_pop=203,
        n_offsprings=100,
        save_history=None,
        seed=None,
        n_jobs=-1,
        verbose=1,
    ) -> None:

        self.classifier_class = classifier_class
        self.constraints = constraints
        self.norm = norm
        self.fun_distance_preprocess = fun_distance_preprocess
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.n_offsprings = n_offsprings

        self.save_history = save_history
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Defaults
        self.alg_class = RNSGA3
        self.problem_class = AdversarialProblem

        # Computed
        self.ref_points = None

    def _check_inputs(self, x: np.ndarray, y) -> None:
        expected_input_length = self.constraints.mutable_features.shape[0]
        if x.shape[1] != expected_input_length:
            raise ValueError(
                f"Mutable mask has shape (n_features,): "
                f"{expected_input_length}, "
                f"x has shape (n_sample, "
                f"n_features): {x.shape}. n_features must be equal."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "minimize_class argument must be an integer or "
                f"an array of shape ({x.shape[0]})"
            )

        if len(x.shape) != 2:
            raise ValueError(
                f"{x.__name__} ({x.shape}) must have 2 dimensions."
            )

    def _create_algorithm(self) -> GeneticAlgorithm:

        type_mask = self.constraints.feature_types[
            self.constraints.mutable_features
        ]

        sampling = InitialStateSampling(type_mask=type_mask)

        # Default parameters for crossover (prob=0.9, eta=30)
        crossover = MixedVariableCrossover(
            type_mask,
            {
                "real": get_crossover(
                    "real_two_point",
                ),
                "int": get_crossover(
                    "int_two_point",
                ),
            },
        )

        # Default parameters for mutation (eta=20)
        mutation = MixedVariableMutation(
            type_mask,
            {
                "real": get_mutation("real_pm", eta=20),
                "int": get_mutation("int_pm", eta=20),
            },
        )

        ref_points = self.ref_points.copy()

        algorithm = self.alg_class(
            pop_per_ref_point=1,
            ref_points=ref_points,
            n_offsprings=self.n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
            return_least_infeasible=True,
        )

        return algorithm

    def _one_generate(self, x, y: int, classifier):
        # Reduce log
        termination = get_termination("n_gen", self.n_gen)

        constraints = self.constraints

        problem = self.problem_class(
            x_clean=x,
            y_clean=y,
            classifier=classifier,
            constraints=constraints,
            fun_distance_preprocess=self.fun_distance_preprocess,
            norm=self.norm,
        )

        algorithm = self._create_algorithm()

        if self.save_history is not None:
            callback = HistoryCallback()
        else:
            callback = None

        # save_history Implemented from library should always be False
        result = minimize(
            problem,
            algorithm,
            termination,
            verbose=0,
            seed=self.seed,
            callback=callback,
            save_history=False,
        )

        x_adv_mutable = np.array(
            [ind.X.astype(np.float64) for ind in result.pop]
        )
        x_adv = np.repeat(x.reshape(1, -1), x_adv_mutable.shape[0], axis=0)
        x_adv[:, self.constraints.mutable_features] = x_adv_mutable

        if self.save_history is not None:
            history = np.array(result.algorithm.callback.data["F"])
            return x_adv, history
        else:
            return x_adv

    def _batch_generate(self, x, y, batch_i):
        tf_lof_off()

        if self.verbose >= 2:
            print(f"Starting batch #{batch_i} with {len(x)} inputs.")
        iterable = enumerate(x)
        if (self.verbose >= 2) and (batch_i == 0):
            iterable = tqdm(iterable, total=len(x))

        classifier = self.classifier_class

        out = [self._one_generate(x[i], y[i], classifier) for i, _ in iterable]
        if self.save_history is not None:
            out = zip(*out)
            out = [np.array(out_0) for out_0 in out]
        else:
            out = np.array(out)

        return out

    # Loop over inputs to generate adversarials using the _one_generate
    # function above
    def generate(self, x: np.ndarray, y, batch_size=None):
        if self.ref_points is None:
            self.ref_points = get_reference_directions(
                "energy", NB_OBJECTIVES, self.n_pop, seed=1
            )

        batches_i = cut_in_batch(
            np.arange(x.shape[0]),
            n_desired_batch=self.n_jobs,
            batch_size=batch_size,
        )

        if isinstance(y, int):
            y = np.repeat(y, x.shape[0])

        self._check_inputs(x, y)

        iterable = enumerate(batches_i)
        if self.verbose >= 1:
            iterable = tqdm(iterable, total=len(batches_i))

        # Sequential Run
        if self.n_jobs == 1:
            print("Sequential run.")
            out = [
                self._batch_generate(x[batch_indexes], y[batch_indexes], i)
                for i, batch_indexes in iterable
            ]

        # Parallel run
        else:
            print("Parallel run.")
            out = Parallel(n_jobs=self.n_jobs)(
                delayed(self._batch_generate)(
                    x[batch_indexes], y[batch_indexes], i
                )
                for i, batch_indexes in iterable
            )

        if self.save_history is not None:
            out = zip(*out)
            out = [np.concatenate(out_0) for out_0 in out]

            x_adv = out[0]
            histories = out[1]
            return x_adv, histories
        else:
            return np.concatenate(out)
