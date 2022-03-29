import numpy as np


def mutate(X_original, X_mutation):

    if X_original.shape[:-1] != X_mutation.shape[:-1]:
        raise ValueError(
            f"X_original has shape: {X_original.shape}, "
            f"X_mutation has shape {X_mutation.shape}. "
            f"Shapes must be equal until index -1."
        )


def compute_distance(x_1, x_2, norm):
    if norm in ["inf", np.inf]:
        distance = np.linalg.norm(x_1 - x_2, ord=np.inf, axis=1)
    elif norm in ["2", 2]:
        distance = np.linalg.norm(x_1 - x_2, ord=2, axis=1)
    else:
        raise NotImplementedError

    return distance
