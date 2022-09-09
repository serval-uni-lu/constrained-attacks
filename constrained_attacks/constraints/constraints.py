import logging
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
)


@dataclass
class Constraints:
    feature_types: npt.NDArray[Any]
    mutable_features: npt.NDArray[Any]
    lower_bounds: npt.NDArray[Any]
    upper_bounds: npt.NDArray[Any]
    relation_constraints: List[BaseRelationConstraint]
    feature_names: List[str] = None


def get_constraints_from_file(
    features_path: str,
    relation_constraints: List[BaseRelationConstraint],
    col_filter=None,
) -> Constraints:
    features = pd.read_csv(features_path)
    return get_constraints_from_metadata(
        features, relation_constraints, col_filter
    )


def get_constraints_from_metadata(
    metadata: pd.DataFrame,
    relation_constraints: List[BaseRelationConstraint],
    col_filter=None,
) -> Constraints:
    features = metadata
    if col_filter is not None:
        features = features[features["feature"].isin(col_filter)]
    feature_type = features["type"].to_numpy()
    mutable_mask = features["mutable"].to_numpy()
    feature_min = features["min"].to_numpy()
    feature_max = features["max"].to_numpy()
    feature_names = features["feature"].to_numpy()
    return Constraints(
        feature_type,
        mutable_mask,
        feature_min,
        feature_max,
        relation_constraints,
        feature_names,
    )


def get_feature_min_max(
    constraints: Constraints, x: npt.NDArray[Any]
) -> Tuple[np.ndarray, np.ndarray]:

    # Todo: Implement a faster method than this recursive solution
    if (x is not None) and (x.ndim == 2):
        out = [get_feature_min_max(constraints, x0) for x0 in x]
        feature_min = np.array([e[0] for e in out])
        feature_max = np.array([e[1] for e in out])
        return feature_min, feature_max

    lower_bounds, upper_bounds = (
        constraints.lower_bounds,
        constraints.upper_bounds,
    )
    # By default min and max are the extreme values
    feature_min = np.full(lower_bounds.shape, np.nan)
    feature_max = np.full(upper_bounds.shape, np.nan)

    # Creating the mask of value that should be provided by input
    min_dynamic = lower_bounds.astype(str) == "dynamic"
    max_dynamic = upper_bounds.astype(str) == "dynamic"

    # Replace de non-dynamic value by the value provided in the definition
    feature_min[~min_dynamic] = lower_bounds[~min_dynamic]
    feature_max[~max_dynamic] = upper_bounds[~max_dynamic]

    # If the dynamic input was provided, replace value for output,
    # else do nothing (keep the extreme values)
    if (x is not None) and (min_dynamic.sum() > 0):
        feature_min[min_dynamic] = x[min_dynamic]
    if (x is not None) and (max_dynamic.sum() > 0):
        feature_max[max_dynamic] = x[max_dynamic]

    # Raise warning if dynamic input waited but not provided
    dynamic_number = min_dynamic.sum() + max_dynamic.sum()
    if dynamic_number > 0 and x is None:
        logging.getLogger().warning(
            f"{dynamic_number} feature min and max are dynamic but no "
            "input were provided."
        )

    return feature_min, feature_max


def fix_feature_types(constraints: Constraints, x_clean, x_adv):
    int_type_mask = constraints.feature_types != "real"
    if int_type_mask.sum() > 0:
        x_adv = x_adv.copy()
        x_adv_int = x_adv[..., int_type_mask]
        x_clean_int = x_clean[..., int_type_mask]
        x_plus_minus = (
            x_adv_int
            - np.repeat(
                x_clean_int[:, np.newaxis, :], x_adv_int.shape[1], axis=1
            )
        ) >= 0

        x_adv_int[x_plus_minus] = np.floor(x_adv_int[x_plus_minus])
        x_adv_int[~x_plus_minus] = np.ceil(x_adv_int[~x_plus_minus])
        x_adv[..., int_type_mask] = x_adv_int
    return x_adv
