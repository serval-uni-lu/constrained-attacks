import time

import numpy as np
import torch
from torchattacks.attack import Attack
from torchattacks.wrappers.multiattack import MultiAttack

from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.attacks.cta.cfab import CFAB
from constrained_attacks.attacks.moeva.moeva import Moeva2
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from mlc.constraints.constraints import Constraints
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from mlc.transformers.tab_scaler import TabScaler
from mlc.utils import to_numpy_number


class ConstrainedMultiAttack(MultiAttack):
    def __init__(self, objective_calculator, *args, **kargs):
        super(ConstrainedMultiAttack, self).__init__(*args, **kargs)
        self.objective_calculator = objective_calculator

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        for _, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            outputs = self.get_logits(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails])

            success_attack_indices = self.objective_calculator.get_successful_attacks_indexes(
                to_numpy_number(images[fails]).astype(np.float32), labels[fails], to_numpy_number(adv_images.unsqueeze(1)), )

            final_images[success_attack_indices] = adv_images[success_attack_indices]

            fails = torch.masked_select(fails, corrects)
            multi_atk_records.append(len(fails))

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

    def __init__(self, constraints: Constraints, constraints_eval: Constraints, scaler: TabScaler, model,
                 model_objective,
                 fix_equality_constraints_end: bool = True, fix_equality_constraints_iter: bool = True, eps_margin=0.01,
                 norm='Linf', eps=8 / 255, version='standard', n_classes=10, seed=None, verbose=False):
        super().__init__('AutoAttack', model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ['default']
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter

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

        if version == 'standard':  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(self.objective_calculator, [
                CAPGD(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(),
                      verbose=verbose, loss='ce', n_restarts=1,
                      fix_equality_constraints_end=fix_equality_constraints_end,
                      fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin, ),
                CFAB(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(
                ), verbose=verbose, multi_targeted=False, n_classes=n_classes, n_restarts=1,
                     fix_equality_constraints_end=fix_equality_constraints_end,
                     fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin),
                Moeva2(model, constraints=constraints, eps=eps, norm=norm, seed=self.get_seed(),
                       verbose=verbose, fun_distance_preprocess=scaler.transform),
            ])

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == 'plus':
            self._autoattack = ConstrainedMultiAttack(self.objective_calculator, [
                CAPGD(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(),
                      verbose=verbose, loss='ce',
                      fix_equality_constraints_end=fix_equality_constraints_end,
                      fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin, n_restarts=5),
                CFAB(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(
                ), verbose=verbose, n_classes=n_classes, n_restarts=1,
                     fix_equality_constraints_end=fix_equality_constraints_end,
                     fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin),
                Moeva2(model, constraints=constraints, eps=eps, norm=norm, seed=self.get_seed(),
                       verbose=verbose, fun_distance_preprocess=scaler.transform),
                CAPGD(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(),
                      verbose=verbose, loss='ce', n_restarts=1,
                      fix_equality_constraints_end=fix_equality_constraints_end,
                      fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin, ),
                CFAB(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(
                ), verbose=verbose, multi_targeted=True, n_classes=n_classes, n_restarts=1,
                     fix_equality_constraints_end=fix_equality_constraints_end,
                     fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin),
            ])

        elif version == 'rand':
            self._autoattack = MultiAttack([
                CAPGD(constraints, scaler, model, model_objective, eps=eps, norm=norm, seed=self.get_seed(),
                      verbose=verbose, loss='ce',
                      fix_equality_constraints_end=fix_equality_constraints_end,
                      fix_equality_constraints_iter=fix_equality_constraints_iter, eps_margin=eps_margin, eot_iter=20,
                      n_restarts=1),
            ])

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
        return time.time() if self.seed is None else self.seed
