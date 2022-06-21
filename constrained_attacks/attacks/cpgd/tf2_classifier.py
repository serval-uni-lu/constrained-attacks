from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np
from art.estimators.classification import TensorFlowV2Classifier

from constrained_attacks.constraints.constraints import Constraints
from constrained_attacks.constraints.constraints_executor import (
    TensorFlowConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow.compat.v1 as tf
    from art.defences.postprocessor import Postprocessor
    from art.defences.preprocessor import Preprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)


class TF2Classifier(TensorFlowV2Classifier):
    def __init__(
        self,
        model: Callable,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        constraints: Constraints,
        scaler,
        loss_object: Optional["tf.keras.losses.Loss"] = None,
        train_step: Optional[Callable] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union[
            "Preprocessor", List["Preprocessor"], None
        ] = None,
        postprocessing_defences: Union[
            "Postprocessor", List["Postprocessor"], None
        ] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Initialization specific to TensorFlow v2 models.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :param nb_classes: the number of classes in the classification task.
        :param input_shape: shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
            the model and computing gradients of the loss w.r.t. the input.
        :type loss_object: `tf.keras.losses`
        :param train_step: A function that applies a gradient update to the trainable variables with signature
                           train_step(model, images, labels).
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """

        self.constraints = constraints
        self.scaler = scaler

        super().__init__(
            model,
            nb_classes,
            input_shape,
            loss_object,
            train_step,
            channels_first,
            clip_values,
            preprocessing_defences,
            postprocessing_defences,
            preprocessing,
        )

    def unscale_features(self, inputs):
        inputs -= self.scaler.min_
        inputs /= self.scaler.scale_

        return inputs

    def scale_features(self, inputs):
        inputs *= self.scaler.scale_
        inputs += self.scaler.min_

        return inputs

    def constraint_loss(self, inputs):

        executor = TensorFlowConstraintsExecutor(
            AndConstraint(self.constraints.relation_constraints),
            feature_names=self.constraints.feature_names,
        )

        violations = executor.execute(inputs)

        return violations

    def compute_loss(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "tf.Tensor"],
        y: Union[np.ndarray, "tf.Tensor"],
        reduction: str = "none",
        training_mode: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the loss.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of losses of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if self._loss_object is None:  # pragma: no cover
            raise TypeError(
                "The loss function `loss_object` is required for computing losses, but it is not defined."
            )
        prev_reduction = self._loss_object.reduction
        if reduction == "none":
            self._loss_object.reduction = tf.keras.losses.Reduction.NONE
        elif reduction == "mean":
            self._loss_object.reduction = (
                tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )
        elif reduction == "sum":
            self._loss_object.reduction = tf.keras.losses.Reduction.SUM

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)

        if tf.executing_eagerly():
            x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
            predictions = self.model(x_preprocessed_tf, training=training_mode)
            if self._reduce_labels:
                loss = self._loss_object(np.argmax(y, axis=1), predictions)
            else:
                loss = self._loss_object(y, predictions)

            loss_constraints = self.constraint_loss(x_preprocessed_tf)

            loss = loss - loss_constraints

        else:
            raise NotImplementedError("Expecting eager execution.")

        self._loss_object.reduction = prev_reduction
        return loss.numpy()

    def loss_gradient(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "tf.Tensor"],
        y: Union[np.ndarray, "tf.Tensor"],
        training_mode: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, "tf.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if self._loss_object is None:  # pragma: no cover
            raise TypeError(
                "The loss function `loss_object` is required for computing loss gradients, but it has not been "
                "defined."
            )

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                # Apply preprocessing
                if self.all_framework_preprocessing:
                    x_grad = tf.convert_to_tensor(x)
                    tape.watch(x_grad)
                    x_input, y_input = self._apply_preprocessing(
                        x_grad, y=y, fit=False
                    )
                else:
                    x_preprocessed, y_preprocessed = self._apply_preprocessing(
                        x, y=y, fit=False
                    )
                    x_grad = tf.convert_to_tensor(x_preprocessed)
                    tape.watch(x_grad)
                    x_input = x_grad
                    y_input = y_preprocessed

                predictions = self.model(x_input, training=training_mode)

                if self._reduce_labels:
                    loss = self._loss_object(
                        np.argmax(y_input, axis=1), predictions
                    )
                else:
                    loss = self._loss_object(y_input, predictions)

                loss_constraints = self.constraint_loss(x_input)

                loss = loss - loss_constraints

            gradients = tape.gradient(loss, x_grad)

            if isinstance(x, np.ndarray):
                gradients = gradients.numpy()

        else:
            raise NotImplementedError("Expecting eager execution.")

        # Apply preprocessing gradients
        if not self.all_framework_preprocessing:
            gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients
