import numpy as np
from constrained_attacks.typing import NDNumber
import pandas as pd
import torch

def mutate(x_original, x_mutation):

    if x_original.shape[:-1] != x_mutation.shape[:-1]:
        raise ValueError(
            f"X_original has shape: {x_original.shape}, "
            f"X_mutation has shape {x_mutation.shape}. "
            f"Shapes must be equal until index -1."
        )


def compute_distance(x_1, x_2, norm):
    if norm in ["inf", np.inf, "Linf", "linf"]:
        distance = np.linalg.norm(x_1 - x_2, ord=np.inf, axis=-1)
    elif norm in ["2", 2, "L2", "l2"]:
        distance = np.linalg.norm(x_1 - x_2, ord=2, axis=-1)
    else:
        raise NotImplementedError

    return distance


def cut_in_batch(arr, n_desired_batch=1, batch_size=None):

    if batch_size is None:
        n_batch = min(n_desired_batch, len(arr))
    else:
        n_batch = np.ceil(len(arr) / batch_size)
    batches_i = np.array_split(np.arange(arr.shape[0]), n_batch)

    return [arr[batch_i] for batch_i in batches_i]


def fix_types(x_clean: torch.Tensor, x_adv: torch.Tensor, types: pd.Series) -> torch.Tensor:

    x_adv = x_adv.clone()
    int_indices = np.where(types == "int")[0]
    x_adv_ndim = x_adv.ndim

    if x_clean.ndim == 2:
        x_clean = x_clean.unsqueeze(1)
    if x_adv_ndim == 2:
        x_adv = x_adv.unsqueeze(1)

    int_perturbation = x_adv[:, :, int_indices] - x_clean[:, :, int_indices]

    int_perturbation_fixed = torch.fix(int_perturbation)

    x_adv[:, :, int_indices] = x_clean[:, :, int_indices] + int_perturbation_fixed

    if x_adv_ndim == 2:
        x_adv = x_adv[:, 0, :]

    return x_adv


