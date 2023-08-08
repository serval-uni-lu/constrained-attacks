import numpy as np

from constrained_attacks.objective_calculator.cache_objective_calculator import (
    select_k_best,
)


def test_select_k_best():

    metric = np.array(
        [
            [0.19672191, 0.28087056, 0.20876677, 0.7502264, 0.40139525],
            [0.73564829, 0.61114074, 0.42973652, 0.80448866, 0.24802728],
            [0.67235109, 0.30733528, 0.60639451, 0.8768003, 0.78981015],
        ]
    )
    filter_ok = np.array(
        [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
    ).astype(bool)

    out = select_k_best(metric, filter_ok, 1)
    print(out)
    assert np.equal(out[0], np.array([0, 1])).all()
    assert np.equal(out[1], np.array([0, 4])).all()

    out = select_k_best(metric, filter_ok, 2)
    assert np.equal(out[0], np.array([0, 0, 1, 1])).all()
    assert np.equal(out[1], np.array([0, 3, 1, 4])).all()
