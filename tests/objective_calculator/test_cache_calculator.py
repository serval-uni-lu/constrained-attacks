import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from constrained_attacks.objective_calculator import ObjectiveCalculator
from tests.attacks.moeva.url_constraints import get_url_constraints


def test_cache_objective_calculation():
    constraints = get_url_constraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :10
    ]
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")[
        :10
    ]
    model = joblib.load("./tests/resources/url/baseline_rf.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    x_adv = np.repeat(x_clean[:, np.newaxis, :], 5, axis=1)
    objective_calculator = ObjectiveCalculator(
        model_pipeline,
        constraints,
        thresholds={"misclassification": 0.5, "distance": 0.2},
        norm=2,
        fun_distance_preprocess=preprocessing_pipeline.transform,
    )
    success_rate = objective_calculator.get_success_rate(
        x_clean, y_clean, x_adv
    )
    for i in range(7):
        assert 0 <= success_rate[i] and success_rate[i] <= 1.0

    assert success_rate[0] == 1.0
    assert success_rate[2] == 1.0
    assert success_rate[3] == success_rate[1]
    assert success_rate[4] == 1.0
    assert success_rate[5] == success_rate[1]
    assert success_rate[6] == success_rate[1]

    # Computed manually
    assert success_rate[1] == 0.1
