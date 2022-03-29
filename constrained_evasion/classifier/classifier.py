import abc
import warnings

import numpy as np


class AbstractClassifier(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict_proba(self, x):
        raise NotImplementedError


class Classifier:

    """
    Wrapper for classifier having a predict_proba method.
    Extend and override for other classifiers.
    """

    def __init__(self, classifier, n_jobs=1, verbose=0) -> None:
        if hasattr(classifier, "predict_proba") and callable(
            getattr(classifier, "predict_proba")
        ):
            self._classifier = classifier
        else:
            raise ValueError(
                "The provided model does not have methods 'predict_proba'."
            )
        self.set_n_jobs(n_jobs)
        self.set_verbose(verbose)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self._classifier.predict_proba(x)
        if proba.shape[1] == 1:
            proba = np.concatenate((1 - proba, proba), axis=1)
        return proba

    def set_verbose(self, verbose: int) -> None:
        if hasattr(self._classifier, "set_params") and callable(
            getattr(self._classifier, "set_params")
        ):
            self._classifier.set_params(verbose=verbose)

    def set_n_jobs(self, n_jobs: int) -> None:
        if hasattr(self._classifier, "set_params") and callable(
            getattr(self._classifier, "set_params")
        ):
            self._classifier.set_params(n_jobs=n_jobs)
