import numpy as np
from pymoo.algorithms.moo.nsga3 import (
    HyperplaneNormalization,
    associate_to_niches,
    calc_niche_count,
    niching,
)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class ReferenceDirectionSurvival:
    def __init__(self, ref_dirs):
        self.ref_dirs = ref_dirs
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])

    def do(self, pop, F, n_survive):

        # # attributes to be set after the survival
        # F = pop.get("F")

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(
            F, return_rank=True, n_stop_if_ranked=n_survive
        )
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        hyp_norm.update(F, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        #  consider only the population until we come to the splitting front
        i_l = np.concatenate(fronts)
        pop, rank, F = pop[i_l], rank[i_l], F[i_l]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(
            F, self.ref_dirs, ideal, nadir
        )

        # attributes of a population
        # pop.set('rank', rank,
        #         'niche', niche_of_individuals,
        #         'dist_to_niche', dist_to_niche)

        # set the optimum, first front and closest to all reference directions
        # closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        # self.opt = pop[intersect(fronts[0], closest)]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(
                    len(self.ref_dirs), niche_of_individuals[until_last_front]
                )
                n_remaining = n_survive - len(until_last_front)

            S = niching(
                pop[last_front],
                n_remaining,
                niche_count,
                niche_of_individuals[last_front],
                dist_to_niche[last_front],
            )

            survivors = np.concatenate(
                (until_last_front, last_front[S].tolist())
            )
            pop = pop[survivors]

        return pop
