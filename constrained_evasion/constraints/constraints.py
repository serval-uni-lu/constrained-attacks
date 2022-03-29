import abc
from typing import Tuple

import numpy as np


class Constraints(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self):
        self.tolerance = 0.0

    def check_constraints_error(self, x: np.ndarray):
        constraints = self.evaluate(x)
        constraints_violated = np.sum(constraints > 0, axis=0)
        if constraints_violated.sum() > 0:
            raise ValueError(
                f"{constraints_violated}\n Constraints not respected "
                f"{constraints_violated.sum()} times."
            )

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray, use_tensors: bool = False) -> np.ndarray:
        """
        Evaluate the distance to constraints satisfaction of x.
        This method should be overridden by the attacker.

        Args:
            x (np.ndarray): An array of shape (n_samples, n_features)
                containing the sample to evaluate.
            use_tensors (bool): Whether to use tensor operations.

        Returns:
            np.ndarray: An array of shape (n_samples, n_constraints)
                representing the distance to constraints
            satisfaction of each sample.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_nb_constraints(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_mutable_mask(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_min_max(
        self, dynamic_input=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def fix_features_types(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_type(self) -> np.ndarray:
        raise NotImplementedError

    def _calc_relationship_constraints(self, x_adv):
        return np.max(self.evaluate(x_adv), axis=-1) <= 0

    def _calc_boundary_constraints(self, x, x_adv):
        xl, xu = zip(*[self.get_feature_min_max(x0) for x0 in x])
        xl_ok, xu_ok = np.min(
            (xl - np.finfo(np.float32).eps) <= x_adv, axis=1
        ), np.min((xu + np.finfo(np.float32).eps) >= x_adv, axis=1)
        return xl_ok * xu_ok

    def _calc_type_constraints(self, x_adv):
        int_type_mask = self.get_feature_type() != "real"
        type_ok = np.min(
            (x_adv[:, int_type_mask] == np.round(x_adv[:, int_type_mask])),
            axis=1,
        )
        return type_ok

    def _calc_mutable_constraints(self, x, x_adv):
        immutable_mask = ~self.get_mutable_mask()
        if immutable_mask.sum() > 0:
            mutable_ok = np.min(
                (x[:, immutable_mask] == x_adv[:, immutable_mask]), axis=1
            )
        else:
            mutable_ok = np.ones(shape=x_adv.shape[:-1], dtype=np.bool)
        return mutable_ok

    def check_constraints(self, x, x_adv) -> np.ndarray:
        constraints = np.array(
            [
                self._calc_relationship_constraints(x_adv),
                self._calc_boundary_constraints(x, x_adv),
                self._calc_type_constraints(x_adv),
                self._calc_mutable_constraints(x, x_adv),
            ]
        )
        # print(np.sum(constraints, axis=1))
        constraints = np.min(constraints, axis=0)
        return constraints
