from typing import List

import tensorflow as tf

EPS = tf.Variable(0.000001)


def check_min_operands_length(expected, operands: List[tf.Tensor]):
    if len(operands) < expected:
        raise ValueError(
            f"Operands length={len(operands)}, "
            f"expected >= {check_min_operands_length}."
        )


def op_or(operands: List[tf.Tensor]) -> tf.Tensor:
    check_min_operands_length(2, operands)

    local_min = operands[0]
    for i in range(1, len(operands)):
        local_min = tf.minimum(local_min, operands[i])

    return local_min


def op_and(operands: List[tf.Tensor]) -> tf.Tensor:
    check_min_operands_length(2, operands)

    local_sum = operands[0]
    for i in range(1, len(operands)):
        local_sum = local_sum + operands[i]

    return local_sum


def op_inf_eq(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:

    zeros = tf.zeros(a.shape, dtype=a.dtype)
    return tf.maximum(zeros, (a - b))


def op_inf(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return op_inf_eq(a + EPS, b)


def op_equal(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.abs(a - b)
