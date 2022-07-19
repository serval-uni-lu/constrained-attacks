import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

from constrained_attacks.attacks.fast_moeva.moeva import Moeva2
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
def test_run(model):
    constraints = get_url_constraints()
    n_input = 1000
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")

    x_clean = np.repeat(x_clean, np.ceil(n_input / x_clean.shape[0]), axis=0)[
        :n_input
    ]
    y_clean = np.repeat(y_clean, np.ceil(n_input / y_clean.shape[0]), axis=0)[
        :n_input
    ]

    model = joblib.load("./tests/resources/url/baseline_rf.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    n_jobs = 10
    model_pipeline[1].set_params(**{"n_jobs": n_jobs})

    attack = Moeva2(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        n_gen=10,
        save_history="full",
        seed=42,
        n_jobs=1,
    )
    out = attack.generate(x_clean, y_clean, batch_size=n_input)
    assert out.shape == (n_input, attack.n_pop + 3, x_clean.shape[-1])


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
@pytest.mark.parametrize(
    "n_batch",
    [1, 2],
)
def test_run_with_history(model, n_batch):
    constraints = get_url_constraints()
    n_input = 10
    batch_size = n_input / n_batch
    n_gen = 10
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")

    x_clean = np.repeat(x_clean, np.ceil(n_input / x_clean.shape[0]), axis=0)[
        :n_input
    ]
    y_clean = np.repeat(y_clean, np.ceil(n_input / y_clean.shape[0]), axis=0)[
        :n_input
    ]

    model = joblib.load("./tests/resources/url/baseline_rf.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    n_jobs = 10
    model_pipeline[1].set_params(**{"n_jobs": n_jobs})

    attack = Moeva2(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        n_gen=n_gen,
        save_history="full",
        seed=42,
        n_jobs=1,
    )
    x_adv, x_history = attack.generate(
        x_clean, y_clean, batch_size=batch_size, return_history=True
    )
    assert x_adv.shape == (n_input, attack.n_pop + 3, x_clean.shape[-1])
    assert x_history.shape == (n_input, n_gen + 1, attack.n_pop + 3, 3)


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
@pytest.mark.parametrize(
    "n_batch",
    [1, 2],
)
def test_run_without_history(model, n_batch):
    constraints = get_url_constraints()
    n_input = 10
    batch_size = n_input / n_batch
    n_gen = 10
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")

    x_clean = np.repeat(x_clean, np.ceil(n_input / x_clean.shape[0]), axis=0)[
        :n_input
    ]
    y_clean = np.repeat(y_clean, np.ceil(n_input / y_clean.shape[0]), axis=0)[
        :n_input
    ]

    model = joblib.load("./tests/resources/url/baseline_rf.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    n_jobs = 10
    model_pipeline[1].set_params(**{"n_jobs": n_jobs})

    attack = Moeva2(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        n_gen=n_gen,
        save_history="full",
        seed=42,
        n_jobs=1,
    )
    x_adv = attack.generate(
        x_clean, y_clean, batch_size=batch_size, return_history=False
    )
    assert x_adv.shape == (n_input, attack.n_pop + 3, x_clean.shape[-1])
