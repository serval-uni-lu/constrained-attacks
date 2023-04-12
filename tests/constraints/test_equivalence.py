import numpy as np
import tensorflow as tf
import torch

from constrained_attacks.constraints.constraints_executor import (
    NumpyConstraintsExecutor,
    PytorchConstraintsExecutor,
    TensorFlowConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint
from constrained_attacks.datasets import load_dataset


def test_numpy_tf_torch():
    # constraints = get_url_constraints()
    # x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")

    ds = load_dataset("url")
    x_clean, _ = ds.get_x_y()
    constraints = ds.get_constraints()

    x_clean_np = x_clean[:100].to_numpy()
    x_clean_tf = tf.convert_to_tensor(x_clean_np, dtype=tf.float32)
    x_clean_pt = torch.Tensor(x_clean_np)

    constraints = AndConstraint(constraints.relation_constraints)

    c_eval_np = NumpyConstraintsExecutor(constraints).execute(x_clean_np)
    c_eval_tf = TensorFlowConstraintsExecutor(constraints).execute(x_clean_tf)
    c_eval_pt = PytorchConstraintsExecutor(constraints).execute(x_clean_pt)

    assert np.allclose(c_eval_np, c_eval_tf.numpy())
    assert np.allclose(c_eval_np, c_eval_pt.cpu().numpy())
