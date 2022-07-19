import math
import random

import numpy as np
from pymoo.algorithms.moo.rnsga3 import AspirationPointSurvival
from pymoo.factory import get_reference_directions
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from tqdm import tqdm

from constrained_attacks.attacks.fast_moeva.evaluator import Evaluator
from constrained_attacks.attacks.fast_moeva.operators import mate, survive
from constrained_attacks.constraints.constraints import (
    Constraints,
    get_feature_min_max,
)

random.sample(range(100), 10)


class Moeva2:
    def __init__(
        self,
        classifier,
        constraints: Constraints,
        norm=None,
        fun_distance_preprocess=lambda x: x,
        n_gen=100,
        n_pop=200,
        n_offsprings=100,
        save_history=None,
        seed=None,
        n_jobs=-1,
        verbose=1,
    ) -> None:

        self.classifier = classifier
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

        # Computed
        self.ref_points = None

    def generate_batch(
        self, x: np.ndarray, y: np.ndarray, return_history=False
    ):
        history_pop_f = []
        n_input, n_var = x.shape
        n_objective = 3
        ref_points = get_reference_directions(
            "energy", n_objective, self.n_pop, seed=1
        )
        pop_per_ref_point = 1
        mu = 0.05

        n_obj = ref_points.shape[1]

        # add the aspiration point lines
        aspiration_ref_dirs = UniformReferenceDirectionFactory(
            n_dim=n_obj, n_points=pop_per_ref_point
        ).do()

        pop_size = (
            ref_points.shape[0] * aspiration_ref_dirs.shape[0]
            + aspiration_ref_dirs.shape[1]
        )

        # Impure function call, create one per sample
        survivals = []
        for _ in range(n_input):
            survival = AspirationPointSurvival(
                ref_points, aspiration_ref_dirs, mu=mu
            )
            survival.filter_infeasible = False
            survivals.append(survival)
        evaluator_obj = Evaluator(
            x,
            y,
            self.classifier,
            self.constraints,
            self.fun_distance_preprocess,
            self.norm,
        )
        mutable_mask = self.constraints.mutable_features
        int_mask = self.constraints.feature_types[mutable_mask] != "real"

        # Initialize

        pop = np.repeat(x[:, mutable_mask], pop_size, axis=0)

        # Evaluate
        pop_f = evaluator_obj.evaluate(pop)
        x_l, x_u = get_feature_min_max(self.constraints, x)
        x_l, x_u = x_l[:, mutable_mask], x_u[:, mutable_mask]

        history_pop_f.append(pop_f.copy())

        # evaluate
        for _ in tqdm(range(self.n_gen)):

            # Mate
            off = mate(
                pop,
                pop_size,
                self.n_offsprings,
                x_l,
                x_u,
                0.9,
                None,
                20,
                int_mask,
            )

            # Evaluate
            off_f = evaluator_obj.evaluate(off)

            # Survive
            pop, pop_f = survive(
                survivals,
                pop,
                pop_f,
                pop_size,
                off,
                off_f,
                self.n_offsprings,
            )
            history_pop_f.append(pop_f.copy())

        x_adv = np.repeat(x, pop_size, axis=0)
        x_adv[:, mutable_mask] = pop
        x_adv = x_adv.reshape((n_input, pop_size, n_var))

        out = (x_adv,)

        if return_history:
            history_pop_f = np.concatenate(history_pop_f)
            history_pop_f = history_pop_f.reshape(
                (self.n_gen + 1, n_input, pop_size, n_objective)
            )
            history_pop_f = np.swapaxes(history_pop_f, 0, 1)
            out = (*out, history_pop_f)

        return out

    def generate(
        self, x: np.ndarray, y: np.ndarray, batch_size=1, return_history=False
    ) -> np.ndarray:

        n_batch = math.ceil(x.shape[0] / batch_size)
        batch_indexes = np.array_split(np.arange(x.shape[0]), n_batch)
        out = []
        for batch_index in batch_indexes:
            out.append(
                self.generate_batch(
                    x[batch_index], y[batch_index], return_history
                ),
            )

        out = zip(*out)
        processed_out = []
        for e in out:
            e_list = list(e)
            processed_out.append(np.concatenate(e_list))

        processed_out = tuple(processed_out)
        if len(processed_out) == 1:
            processed_out = processed_out[0]
        return processed_out
