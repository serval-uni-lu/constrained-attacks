import numpy as np
import tensorflow as tf

from constrained_attacks.constraints.constraints_executor import (
    TensorFlowConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint
from tests.attacks.moeva.url_constraints_language import get_url_constraints


def test_tf_constraints():
    constraints = get_url_constraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    x_clean = tf.convert_to_tensor(x_clean[:2], dtype=tf.float32)
    executor = TensorFlowConstraintsExecutor(
        AndConstraint(constraints.relation_constraints)
    )
    c_eval = executor.execute(x_clean)
    assert tf.reduce_all(c_eval == 0)
