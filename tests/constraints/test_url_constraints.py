import numpy as np
import tensorflow as tf

from tests.attacks.moeva.url_constraints import UrlConstraints


def test_tf_constraints():
    constraints = UrlConstraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    x_clean = tf.convert_to_tensor(x_clean)
    c_eval = constraints.evaluate_tf(x_clean)
    assert tf.reduce_all(c_eval == 0)
