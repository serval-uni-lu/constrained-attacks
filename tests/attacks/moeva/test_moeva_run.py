import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from constrained_evasion.attacks import Moeva2
from tests.attacks.moeva.url_constraints import UrlConstraints


def test_run():
    constraints = UrlConstraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")
    model = joblib.load("./tests/resources/url/baseline.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    attack = Moeva2(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        save_history="full",
        seed=42,
        n_jobs=10,
    )
    out = attack.generate(x_clean[:10], y_clean[:10])
    assert len(out) == 2
