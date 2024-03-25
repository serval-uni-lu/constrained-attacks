import time

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def test_model_performance():
    model = joblib.load("./tests/resources/url/baseline_rf.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )
    n_pop = 203
    n_inputs = 1000
    n_jobs = 4
    n_gen = 10
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :n_pop
    ]

    def simulate_adv_generation(model_l, x, series_repeat):
        calls = 0
        for _ in tqdm(range(series_repeat)):
            model_l.predict_proba(x + np.random.rand(1))
            calls += x.shape[0]
        return calls

    # First version
    model_pipeline[1].set_params(**{"n_jobs": 1})

    start = time.time()
    repeat = int(n_gen * n_inputs / n_jobs)
    out = Parallel(n_jobs=n_jobs)(
        delayed(simulate_adv_generation)(model_pipeline, x_clean, repeat)
        for i in range(n_jobs)
    )
    print(f"n_calls_1 {out}")
    time1 = time.time() - start

    # New version
    model_pipeline[1].set_params(**{"n_jobs": n_jobs})

    start = time.time()
    print(
        f"n_call_2 {simulate_adv_generation(model_pipeline, np.repeat(x_clean, n_inputs, axis=0), n_gen)}"
    )
    time2 = time.time() - start

    print(time1, time2)
    print(time1 / time2)
    assert time2 < time1
