import time
from warnings import warn

import numpy as np
import torch
from mlc.constraints.constraints import Constraints
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.numpy_backend import NumpyBackend
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from mlc.transformers.tab_scaler import TabScaler
from mlc.utils import to_numpy_number
from torchattacks.attack import Attack
from torchattacks.wrappers.multiattack import MultiAttack

from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.attacks.cta.capgd2 import CAPGD2
from constrained_attacks.attacks.cta.cfab import CFAB
from constrained_attacks.attacks.cta.cpgdl2 import CPGDL2
from constrained_attacks.attacks.moeva.moeva import Moeva2
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)


class NoAttack:
    """Utility class to have no attack."""

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


class ConstrainedMultiAttack(MultiAttack):
    def __init__(self, objective_calculator, *args, **kargs):
        super(ConstrainedMultiAttack, self).__init__(*args, **kargs)
        self.objective_calculator = objective_calculator
        self.attack_times = []
        self.robust_accuracies = []
        self.constraints_rate = []
        self.distance_ok = []
        self.mdc = []
        self.stopwatch_data = []

    # Override from upper class
    # Moeva does not use the same model as other attacks
    def check_validity(self):
        if len(self.attacks) < 2:
            warn("More than two attacks should be given.")

        ids = []
        for attack in self.attacks:
            if isinstance(attack, Moeva2):
                if hasattr(attack.model, "__self__"):
                    ids.append(id(attack.model.__self__.wrapper_model))
                else:
                    ids.append(id(attack.model.wrapper_model))
            else:
                ids.append(id(attack.model))
        if len(set(ids)) != 1:
            raise ValueError(
                "At least one of attacks is referencing a different model."
            )

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        def constraints_distance(c, x):
            x_dim = len(x.shape)
            x_shape = x.shape
            x = x.reshape(-1, x_shape[-1])

            ce = ConstraintsExecutor(
                c,
                backend=NumpyBackend(),
                feature_names=self.objective_calculator.constraints.feature_names,
            )
            distance = ce.execute(x)
            distance = distance.reshape(x_shape[: x_dim - 1])
            return distance

        def constraints_percentage(c, x, measure):
            x_dim = len(x.shape)
            distance = constraints_distance(c, x)

            constraint_ok = (
                distance <= self.objective_calculator.thresholds["constraints"]
            )

            if x_dim == 2:
                constraint_ok = constraint_ok[:, np.newaxis, :]

            percentage = constraint_ok
            self.distance_ok.append(np.mean(measure.distance))

            percentage = np.mean(percentage, axis=1)
            percentage = np.mean(percentage, axis=0)

            return percentage

        for attack_i, attack in enumerate([NoAttack()] + self.attacks):
            # Attack the one that failed so far
            # if attack_i > 0:
            #     exit(0)
            start_time = time.time()
            # attack.objective_calculator = self.objective_calculator
            adv_images = attack(images[fails], labels[fails])
            self.attack_times.append(time.time() - start_time)

            # Correct shape
            filter_adv = (
                adv_images.unsqueeze(1)
                if len(adv_images.shape) < 3
                else adv_images
            )
            # Type conversion
            numpy_clean = to_numpy_number(images[fails]).astype(np.float32)
            numpy_adv = to_numpy_number(filter_adv)

            # Indexes of the successful attacks
            (
                success_attack_indices,
                success_adversarials_indices,
            ) = self.objective_calculator.get_successful_attacks_indexes(
                numpy_clean, labels[fails].cpu(), numpy_adv, max_inputs=1
            )

            # If we found adversarials
            if len(success_attack_indices) > 0:
                final_images[fails[success_attack_indices]] = filter_adv[
                    success_attack_indices, success_adversarials_indices, :
                ].squeeze(1)
                mask = torch.ones_like(fails)
                mask[success_attack_indices] = 0
                fails = fails.masked_select(mask.bool())

            multi_atk_records.append(len(fails))

            self.robust_accuracies.append(1 - (len(fails) / batch_size))

            self.mdc.append(
                self.objective_calculator.get_success_rate(
                    numpy_clean,
                    labels[fails].cpu(),
                    numpy_adv,
                    recompute=False,
                )
            )

            log_constraints = True
            if log_constraints:
                numpy_adv_constraints = numpy_adv.reshape(-1, numpy_adv.shape[-1])

                executor = ConstraintsExecutor(
                    AndConstraint(
                        self.objective_calculator.constraints.relation_constraints
                    ),
                    NumpyBackend(),
                    feature_names=self.objective_calculator.constraints.feature_names,
                )
                executor.execute(numpy_adv_constraints)
                constraints_and = executor.get_last_and_operands()
                self.constraints_rate.append(
                    (constraints_and <= 0).astype(int).mean(0)
                )
                if hasattr(attack, "stopwatch"):
                    self.stopwatch_data.append(attack.stopwatch.get_all_total())
                if hasattr(attack, "stopwatch_data"):
                    self.stopwatch_data.append(attack.stopwatch_data)

            print("Over")

            if len(fails) == 0:
                break

        if self.verbose:
            print(self._return_sr_record(multi_atk_records))

        if self._accumulate_multi_atk_records:
            self._update_multi_atk_records(multi_atk_records)

        return final_images


class ConstrainedAutoAttack(Attack):
    r"""
    Extended AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    with constrained examples

    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        constraints_eval: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        n_jobs=-1,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        eps_margin=0.01,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.n_jobs = n_jobs

        if self.constraints_eval.relation_constraints is not None:
            self.objective_calculator = ObjectiveCalculator(
                model_objective,
                constraints=self.constraints_eval,
                thresholds={"distance": eps},
                norm=norm,
                fun_distance_preprocess=self.scaler.transform,
            )
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints_eval.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints_eval.feature_names,
            )
        else:
            self.objective_calculator = None
            self.constraints_executor = None

        if version == "standard":  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=False,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model_objective,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                ],
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        n_restarts=5,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                ],
            )

        elif version == "rand":
            self._autoattack = MultiAttack(
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return int(time.time()) if self.seed is None else self.seed


class ConstrainedAutoAttack2(Attack):
    r"""
    Extended AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    with constrained examples

    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        constraints_eval: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        n_jobs=-1,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        eps_margin=0.01,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.n_jobs = n_jobs

        if self.constraints_eval.relation_constraints is not None:
            self.objective_calculator = ObjectiveCalculator(
                model_objective,
                constraints=self.constraints_eval,
                thresholds={"distance": eps},
                norm=norm,
                fun_distance_preprocess=self.scaler.transform,
            )
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints_eval.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints_eval.feature_names,
            )
        else:
            self.objective_calculator = None
            self.constraints_executor = None

        if version == "standard":  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CPGDL2(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=False,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model_objective,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                ],
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        n_restarts=5,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                ],
            )

        elif version == "rand":
            self._autoattack = MultiAttack(
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return int(time.time()) if self.seed is None else self.seed


class ConstrainedAutoAttack3(Attack):
    r"""
    Extended AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    with constrained examples

    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        constraints_eval: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        n_jobs=-1,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        eps_margin=0.01,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
        steps=10,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.n_jobs = n_jobs
        self.steps = steps

        if self.constraints_eval.relation_constraints is not None:
            self.objective_calculator = ObjectiveCalculator(
                model_objective,
                constraints=self.constraints_eval,
                thresholds={"distance": eps},
                norm=norm,
                fun_distance_preprocess=self.scaler.transform,
            )
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints_eval.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints_eval.feature_names,
            )
        else:
            self.objective_calculator = None
            self.constraints_executor = None

        if version == "standard":  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CPGDL2(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        steps=self.steps,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        steps=self.steps,
                    ),
                    Moeva2(
                        model_objective,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                ],
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        n_restarts=5,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                ],
            )

        elif version == "rand":
            self._autoattack = MultiAttack(
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return int(time.time()) if self.seed is None else self.seed


class ConstrainedAutoAttack4(Attack):
    r"""
    Extended AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    with constrained examples

    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        constraints_eval: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        n_jobs=-1,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        eps_margin=0.01,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
        steps=10,
        n_gen=100,
        n_offsprings=100,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.n_jobs = n_jobs
        self.steps = steps
        self.n_gen = n_gen
        self.n_offsprings = n_offsprings

        if self.constraints_eval.relation_constraints is not None:
            self.objective_calculator = ObjectiveCalculator(
                model_objective,
                constraints=self.constraints_eval,
                thresholds={"distance": eps},
                norm=norm,
                fun_distance_preprocess=self.scaler.transform,
            )
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints_eval.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints_eval.feature_names,
            )
        else:
            self.objective_calculator = None
            self.constraints_executor = None

        if version == "standard":  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD2(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=2,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        steps=self.steps,
                    ),
                    Moeva2(
                        model_objective,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                        n_gen=self.n_gen,
                        n_offsprings=self.n_offsprings,
                    ),
                ],
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        n_restarts=5,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                ],
            )

        elif version == "rand":
            self._autoattack = MultiAttack(
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return int(time.time()) if self.seed is None else self.seed


class ConstrainedAutoAttack5(Attack):
    r"""
    Extended AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    with constrained examples

    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        constraints_eval: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        n_jobs=-1,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        eps_margin=0.01,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
        steps=10,
        n_gen=100,
        n_offsprings=100,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.n_jobs = n_jobs
        self.steps = steps
        self.n_gen = n_gen
        self.n_offsprings = n_offsprings

        # TODO
        self.objective_calculator = ObjectiveCalculator(
            model_objective,
            constraints=self.constraints_eval,
            thresholds={"distance": eps},
            norm=norm,
            fun_distance_preprocess=self.scaler.transform,
        )
        if self.constraints_eval.relation_constraints is not None:
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints_eval.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints_eval.feature_names,
            )
        else:
            self.constraints_executor = None

        if version == "standard":  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD2(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=2,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        best_restart=False,
                        steps=self.steps,
                    ),
                    Moeva2(
                        model_objective,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                        n_gen=self.n_gen,
                        n_offsprings=self.n_offsprings,
                    ),
                ],
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        n_restarts=5,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    Moeva2(
                        model,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                    ),
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                    CFAB(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                    ),
                ],
            )

        elif version == "rand":
            self._autoattack = MultiAttack(
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return int(time.time()) if self.seed is None else self.seed
