import time

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

from constrained_attacks.attacks.fast_moeva.moeva import FastMoeva2
from constrained_attacks.attacks.moeva.moeva import Moeva2 as SlowMoeva
from constrained_attacks.classifier.tensorflow_classifier import (
    TensorflowClassifier,
)
from tests.attacks.moeva.url_constraints import get_url_constraints


@pytest.mark.parametrize(
    "model",
    [
        (joblib.load("./tests/resources/url/baseline_rf.model")),
        (
            TensorflowClassifier(
                load_model("./tests/resources/url/baseline_nn.model")
            )
        ),
    ],
)
def test_run_perf(model):
    n_jobs = 10

    constraints = get_url_constraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :100
    ]
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")[
        :100
    ]
    model = joblib.load("./tests/resources/url/baseline_rf.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    # FAST
    model_pipeline[1].set_params(**{"n_jobs": n_jobs})
    attack = FastMoeva2(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        n_gen=100,
        save_history="full",
        seed=42,
        n_jobs=1,
    )
    start = time.time()
    attack.generate(x_clean, y_clean, batch_size=1000)
    fast_time = time.time() - start

    # SLOW
    model_pipeline[1].set_params(**{"n_jobs": 1})
    attack2 = SlowMoeva(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        n_gen=100,
        save_history="full",
        seed=42,
        n_jobs=n_jobs,
    )
    start = time.time()
    attack2.generate(x_clean, y_clean)
    slow_time = time.time() - start
    assert fast_time < slow_time
