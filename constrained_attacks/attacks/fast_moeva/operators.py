import math
from typing import List

import numpy as np
from joblib import Parallel, delayed
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.util.misc import random_permuations

from constrained_attacks.attacks.fast_moeva.survival import (
    ReferenceDirectionSurvival,
)


def survive_one(
    survival: ReferenceDirectionSurvival,
    pop,
    pop_f,
    n_pop,
    off,
    off_f,
    n_off,
):
    local_pop_i = np.arange(n_pop + n_off)
    F = np.concatenate(
        [
            pop_f,
            off_f,
        ],
        axis=0,
    )
    # local_pop_i = Population.new("X", local_pop_i)
    local_pop_i_survive = survival.do(local_pop_i, F, n_survive=n_pop)
    # local_pop_i_survive = local_pop_i_survive.get("X")
    local_pop_survive = np.concatenate(
        [
            pop[local_pop_i_survive[local_pop_i_survive < n_pop]],
            off[local_pop_i_survive[local_pop_i_survive >= n_pop] - n_pop],
        ],
        axis=0,
    )
    return local_pop_survive, F[local_pop_i_survive]


def survive(
    survivals: List[ReferenceDirectionSurvival],
    pop,
    pop_f,
    n_pop,
    off,
    off_f,
    n_off,
    n_jobs,
):

    # for i, survival in tqdm(enumerate(survivals), total=len(survivals)):
    batch_size = np.ceil(len(survivals) / n_jobs).astype(int)
    out = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=0)(
        delayed(survive_one)(
            survivals[i],
            pop[i * n_pop : (i + 1) * n_pop],
            pop_f[i * n_pop : (i + 1) * n_pop],
            n_pop,
            off[i * n_off : (i + 1) * n_off],
            off_f[i * n_off : (i + 1) * n_off],
            n_off,
        )
        for i in range(len(survivals))
    )

    for i in range(len(out)):
        pop[i * n_pop : (i + 1) * n_pop] = out[i][0]
        pop_f[i * n_pop : (i + 1) * n_pop] = out[i][1]

    return pop, pop_f


def select(n_input, n_pop, n_offspring):
    out = []
    for i in range(n_input):
        n_select = math.ceil(n_offspring / 2)
        n_parents = 2
        pressure = 1
        n_random = n_select * n_parents * pressure
        n_perms = math.ceil(n_random / n_pop)
        parents = random_permuations(n_perms, n_pop)[:n_random]
        parents = np.reshape(parents, (n_select, -1))
        parents = parents + (i * n_pop)
        out.append(parents)
    return np.concatenate(out)


def crossover(pop, parents, prob):

    x = pop[parents.T].copy()
    crossover_obj = PointCrossover(2)

    do_crossover = np.random.random(len(parents)) < prob

    _x = crossover_obj._do(None, x[:, do_crossover, :])
    x[:, do_crossover, :] = _x

    off = x.reshape(-1, pop.shape[-1])

    return off


def mutation(X, x_l, x_u, prob, eta, int_mask):

    x_l[:, int_mask] = x_l[:, int_mask] - (0.5 - 1e-16)
    x_u[:, int_mask] = x_u[:, int_mask] + (0.5 - 1e-16)

    X = X.astype(float)
    Y = np.full(X.shape, np.inf)
    n_var = X.shape[-1]

    if prob is None:
        prob = 1.0 / n_var

    do_mutation = np.random.random(X.shape) < prob

    Y[:, :] = X

    xl = x_l[do_mutation]
    xu = x_u[do_mutation]

    X = X[do_mutation]

    delta1 = (X - xl) / (xu - xl)
    delta2 = (xu - X) / (xu - xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = np.random.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (xu - xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < xl] = xl[_Y < xl]
    _Y[_Y > xu] = xu[_Y > xu]

    # set the values for output
    Y[do_mutation] = _Y

    # in case out of bounds repair (very unlikely)
    Y[Y < x_l] = x_l[Y < x_l]
    Y[Y > x_u] = x_u[Y > x_u]

    Y[:, int_mask] = np.rint(Y[:, int_mask])

    return Y


def mate(
    pop,
    n_pop,
    n_offspring,
    x_l,
    x_u,
    crossover_proba,
    mutation_proba,
    mutation_eta,
    int_mask,
):

    n_input = int(pop.shape[0] / n_pop)

    # select
    parents = select(n_input, n_pop, n_offspring)

    # crossover
    off = crossover(pop, parents, crossover_proba)

    # Mutation
    x_l_mut = np.repeat(x_l, n_offspring, axis=0)
    x_u_mut = np.repeat(x_u, n_offspring, axis=0)
    off = mutation(
        off, x_l_mut, x_u_mut, mutation_proba, mutation_eta, int_mask
    )

    return off
