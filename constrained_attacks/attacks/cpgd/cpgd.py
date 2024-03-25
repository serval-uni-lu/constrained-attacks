import numpy as np
from art.attacks.evasion import ProjectedGradientDescent as PGD
from art.estimators.classification import TensorFlowV2Classifier
from constrained_attacks.constraints.constraints import (
    Constraints,
    fix_feature_types,
    get_feature_min_max,
)

from constrained_attacks.attacks.cpgd.tf2_classifier import TF2Classifier


class CPGD:
    def __init__(
            self,
            classifier,
            constraints: Constraints,
            norm=None,
            eps=0.3,
            eps_step=0.1,
            save_history=None,
            seed=None,
            n_jobs=-1,
            verbose=1,
            enable_constraints=True,
    ) -> None:
        self.classifier = classifier
        self.constraints = constraints
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.save_history = save_history
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.enable_constraints = enable_constraints

    def generate(self, x_clean, y_clean):

        # Convert classifier
        scaler = self.classifier[0]
        tf_model = self.classifier[1].get_internal_classifier()

        xl, xu = get_feature_min_max(self.constraints, x_clean)

        xl, xu = scaler.transform(xl), scaler.transform(xu)
        xu[xu == xl] = xu[xu == xl] + 0.0000000000001
        if self.enable_constraints:
            tf_model_atk = TF2Classifier(
                tf_model,
                constraints=self.constraints,
                scaler=scaler,
                input_shape=x_clean.shape[1:],
                nb_classes=2,
                loss_object=tf_model.loss,
                clip_values=(xl, xu),
            )
        else:
            tf_model_atk = TensorFlowV2Classifier(
                tf_model,
                input_shape=x_clean.shape[1:],
                nb_classes=2,
                loss_object=tf_model.loss,
                clip_values=(xl, xu),
            )

        # Generate inputs
        attack = PGD(
            tf_model_atk,
            eps=self.eps,
            eps_step=self.eps_step,
            targeted=False,
            norm=2,
        )
        x_adv = attack.generate(
            scaler.transform(x_clean),
            y_clean,
            mask=self.constraints.mutable_features,
        )

        x_adv = scaler.inverse_transform(x_adv)

        # Fix datatypes
        x_adv = x_adv[:, np.newaxis, :]
        x_adv = fix_feature_types(self.constraints, x_clean, x_adv)
        return x_adv
