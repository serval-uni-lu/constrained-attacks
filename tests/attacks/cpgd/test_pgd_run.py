import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

from constrained_attacks.attacks.cpgd.cpgd import CPGD
from constrained_attacks.classifier.tensorflow_classifier import (
    TensorflowClassifier,
)
from constrained_attacks.objective_calculator import ObjectiveCalculator
from tests.attacks.moeva.url_constraints import get_url_constraints


def test_run_pgd():

    constraints = get_url_constraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :10
    ]
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")[
        :10
    ]
    model = TensorflowClassifier(
        load_model("./tests/resources/url/baseline_nn.model")
    )
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    attack = CPGD(
        model_pipeline,
        constraints,
        norm=2,
        eps=0.2,
        eps_step=0.1,
        save_history=None,
        seed=None,
        n_jobs=-1,
        verbose=1,
        enable_constraints=False,
    )
    x_adv = attack.generate(x_clean, y_clean)

    objective_calculator = ObjectiveCalculator(
        model_pipeline,
        constraints,
        thresholds={"misclassification": 0.5, "distance": 0.3},
        norm=2,
        fun_distance_preprocess=preprocessing_pipeline.transform,
    )
    success_rate = objective_calculator.get_success_rate(
        x_clean, y_clean, x_adv
    )
    print(success_rate)
    assert len(x_adv) == 10


def test_run_cpgd():

    constraints = get_url_constraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :10
    ]
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")[
        :10
    ]
    model = TensorflowClassifier(
        load_model("./tests/resources/url/baseline_nn.model")
    )
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    attack = CPGD(
        model_pipeline,
        constraints,
        norm=2,
        eps=0.2,
        eps_step=0.1,
        save_history=None,
        seed=None,
        n_jobs=-1,
        verbose=1,
    )
    x_adv = attack.generate(x_clean, y_clean)

    objective_calculator = ObjectiveCalculator(
        model_pipeline,
        constraints,
        thresholds={"misclassification": 0.5, "distance": 0.3},
        norm=2,
        fun_distance_preprocess=preprocessing_pipeline.transform,
    )
    success_rate = objective_calculator.get_success_rate(
        x_clean, y_clean, x_adv
    )
    print(success_rate)
    assert len(x_adv) == 10
