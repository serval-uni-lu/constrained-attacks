def mutate(X_original, X_mutation):

    if X_original.shape[:-1] != X_mutation.shape[:-1]:
        raise ValueError(
            f"X_original has shape: {X_original.shape}, "
            f"X_mutation has shape {X_mutation.shape}. "
            f"Shapes must be equal until index -1."
        )
