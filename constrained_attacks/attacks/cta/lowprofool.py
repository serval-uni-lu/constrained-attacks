# Misc
# [1] Ballet, V., Renard, X., Aigrain, J., Laugel, T., Frossard, P., & Detyniecki, M. (2019). Imperceptible Adversarial Attacks on Tabular Data. NeurIPS 2019 Workshop on Robust AI in Financial Services: Data, Fairness, Explainability, Trustworthiness, and Privacy (Robust AI in FS 2019)
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from mlc.constraints.constraints import Constraints
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from mlc.models.model import Model
from mlc.transformers.tab_scaler import TabScaler
from torchattacks.attack import Attack

from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from constrained_attacks.utils import (
    fix_equality_constraints,
    fix_immutable,
    fix_types,
)
from joblib import Parallel, delayed
from tqdm import tqdm

class MaskModule(nn.Module):

    def __init__(self, x_clean, mutable_mask, model, scaler):
        super().__init__()
        self.model = model
        self.mutable_mask = mutable_mask
        self.x_clean = x_clean
        self.scaler = scaler

        self.base = self.scaler.transform(x_clean.reshape(1, -1))[0]
        self.base.requires_grad = True
        # self.base[self.mutable_mask] = 0
        self.mutable_idx = torch.where(self.mutable_mask)[0]
        self.apply_softmax = False

    def forward(self, x):
        base = self.base.clone()
        base[self.mutable_idx] = x
        base = self.scaler.inverse_transform(base.reshape(1, -1))
        out = self.model(base)

        if self.apply_softmax:
            out = torch.softmax(out, 1)
        out = out[0]
        return out

    def set_apply_softmax(self, apply_softmax):
        self.apply_softmax = apply_softmax


class LowProFool(Attack):

    def __init__(
        self,
        constraints: Constraints,
        scaler: TabScaler,
        model: Model,
        weights,
        eps: float = 8 / 255,
        steps: int = 20000,
        alpha: float = 0.001,
        lambda_: float = 8.5,
        seed: int = 0,
        model_name= None,
        fix_equality_constraints_end: bool = True,
        verbose: bool = False,
        **kwargs
    ) -> None:
        super().__init__("LowProFool", model)
        self.constraints = constraints
        self.scaler = scaler
        self.steps = steps
        self.verbose = verbose
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.alpha = alpha
        self.lambda_ = lambda_
        self.eps = eps
        self.seed = seed
        self.model_name = model_name

        # Add categorical features to immutable features as in [1]
        mutable_mask = np.logical_and(
            self.constraints.mutable_features,
            np.logical_not(self.constraints.feature_types == "cat"),
        )

        self.weights = weights[mutable_mask]

        self.mutable_mask = scaler.transform_mask(
            torch.tensor(mutable_mask, dtype=torch.float)
        ).to(self.device).bool()

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Overridden.
        """
        # self._check_inputs(images)

        x = images
        x_in = images.clone()
        x = self.scaler.transform(x)

        x = x.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv = self.perturb(x, labels, cheap=True)

        # print(f"SHAPE {adv.shape}")
        x = self.scaler.inverse_transform(x)
        adv = self.scaler.inverse_transform(adv)

        adv = fix_types(x_in, adv, self.constraints.feature_types)
        adv = fix_immutable(x_in, adv, self.constraints.mutable_features)

        # print(f"Length constraints {len(self.constraints.relation_constraints)}")
        if self.fix_equality_constraints_end:
            adv = fix_equality_constraints(self.constraints, adv)

        return adv

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if hasattr(self, "scaler") and self.scaler is not None:
            inputs = self.scaler.inverse_transform(inputs)
        return super().get_logits(inputs, labels, *args, **kwargs)

    def perturb_one(
        self,
        x_in,
        y_in,
    ):
        x_clean = x_in.clone()
        model = MaskModule(x_in, self.mutable_mask, self.model, self.scaler)
        if self.model_name == "tabnet":
            model.set_apply_softmax(True)
        x_in = x_in[self.mutable_mask]
        bounds = [np.zeros(x_in.shape), np.ones(x_in.shape)]

        out = lowProFool(
            x_in,
            y_in,
            model,
            self.weights,
            bounds,
            self.steps,
            self.alpha,
            self.lambda_,
        )
        x_adv = x_clean.clone()
        # print(x_adv.shape)
        # print(self.mutable_mask.shape)
        # print(out[3].shape)

        x_adv[self.mutable_mask] = out[3]
        return x_adv

    def perturb(self, x_in, y_in, cheap=True):
        torch.random.manual_seed(self.seed)
        # x_out = []
        # for i in range(len(x_in)):
        #     print(f"Example {i}")
        #     x_out.append(self.perturb_one(x_in[i], y_in[i]))
        
        x_out = Parallel(n_jobs=20)(delayed(self.perturb_one)(x_in[i], y_in[i]) for i in tqdm(range(len(x_in))))
        # print(f"LEn {len(x_out)/}")
        
        return torch.stack(x_out)


# Clipping function
def clip(current, low_bound, up_bound):
    assert len(current) == len(up_bound) and len(low_bound) == len(up_bound)
    low_bound = torch.FloatTensor(low_bound)
    up_bound = torch.FloatTensor(up_bound)
    clipped = torch.max(torch.min(current, up_bound), low_bound)
    return clipped


def lowProFool(
    x, y_clean, model, weights, bounds, maxiters, alpha, lambda_
):
    """
    Generates an adversarial examples x' from an original sample x

    :param x: tabular sample
    :param target_pred: Target
    :param model: neural network
    :param weights: feature importance vector associated with the dataset at hand
    :param bounds: bounds of the datasets with respect to each feature
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
    :return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
    """

    r = Variable(
        torch.FloatTensor(1e-4 * np.ones(x.numpy().shape)), requires_grad=True
    )
    v = torch.FloatTensor(np.array(weights))

    output = model.forward(x + r)

    # print(torch.min(output).cpu().detach().numpy())
    # if torch.min(output).cpu().detach().numpy() < 0:
    #     model.set_apply_softmax(True)
    #     output = model.forward(x + r)
    orig_pred = output.max(0, keepdim=True)[1].cpu().numpy()
    # target_pred = np.abs(1 - orig_pred)
    # orig_pred = 1 - target_pred

    target_pred = 1 - y_clean
    target = [0.0, 1.0] if target_pred == 1 else [1.0, 0.0]
    target = Variable(torch.tensor(target, requires_grad=False))
    # print(target)
    # print(output)
    # exit(0)

    lambda_ = torch.tensor([lambda_])

    bce = nn.BCELoss()
    l1 = lambda v, r: torch.sum(torch.abs(v * r))  # L1 norm
    l2 = lambda v, r: torch.sqrt(torch.sum(torch.mul(v * r, v * r)))  # L2 norm

    best_norm_weighted = np.inf
    best_pert_x = x

    loop_i, loop_change_class = 0, 0
    while loop_i < maxiters:

        # zero_gradients(r)
        if x.grad is not None:
            x.grad.zero_()

        # Computing loss
        # if loop_i % 100 == 0:
        #     print(f"step {loop_i}/{maxiters}, output {output}")
        # print(f"v {v}")
        # print(f"r {r}")
        # print(f"v.shape {v.shape}")
        # print(f"r.shape {r.shape}")
        loss_1 = bce(output, target)
        loss_2 = l2(v, r)
        loss = loss_1 + lambda_ * loss_2
        
        # print(f"loss_1 {loss_1}")
        # print(f"loss_2 {loss_2}")
        # print(f"loss {loss}")
        # Get the gradient
        loss.backward(retain_graph=True)
        grad_r = r.grad.data.cpu().numpy().copy()
        # print(f"grad_r {grad_r}")

        # Guide perturbation to the negative of the gradient
        ri = -grad_r
        # print(f"ri {ri}")


        # limit huge step
        ri *= alpha
        # print(f"ri {ri}")

        # Adds new perturbation to total perturbation
        r = r.clone().detach().cpu().numpy() + ri
        # print(f"r {r}")


        # For later computation
        r_norm_weighted = np.sum(np.abs(r * weights))
        # print(f"r_norm_weighted {r_norm_weighted}")

        # Ready to feed the model
        r = Variable(torch.FloatTensor(r), requires_grad=True)
        # print(f"r {r}")

        # Compute adversarial example
        xprime = x + r

        # Clip to stay in legitimate bounds
        xprime = clip(xprime, bounds[0], bounds[1])
        # print(f"Step {loop_i} {xprime}")

        # Classify adversarial example
        output = model.forward(xprime)
        # print(output)
        output_pred = output.max(0, keepdim=True)[1].cpu().numpy()

        # Keep the best adverse at each iterations
        if output_pred != orig_pred and r_norm_weighted < best_norm_weighted:
            best_norm_weighted = r_norm_weighted
            best_pert_x = xprime

        if output_pred == orig_pred:
            loop_change_class += 1

        loop_i += 1

    # Clip at the end no matter what
    best_pert_x = clip(best_pert_x, bounds[0], bounds[1])
    output = model.forward(best_pert_x)
    output_pred = output.max(0, keepdim=True)[1].cpu().numpy()

    return (
        orig_pred,
        output_pred,
        best_pert_x.clone().detach().cpu().numpy(),
        loop_change_class,
    )


# Forked from https://github.com/LTS4/DeepFool
def deepfool(x_old, net, maxiters, alpha, bounds, weights=[], overshoot=0.002):
    """
    :param image: tabular sample
    :param net: network
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param bounds: bounds of the datasets with respect to each feature
    :param weights: feature importance vector associated with the dataset at hand
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    input_shape = x_old.numpy().shape
    x = x_old.clone()
    x = Variable(x, requires_grad=True)

    output = net.forward(x)
    orig_pred = output.max(0, keepdim=True)[
        1
    ]  # get the index of the max log-probability

    origin = Variable(torch.tensor([orig_pred], requires_grad=False))

    I = []
    if orig_pred == 0:
        I = [0, 1]
    else:
        I = [1, 0]

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    k_i = origin

    loop_i = 0
    while torch.eq(k_i, origin) and loop_i < maxiters:

        # Origin class
        output[I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        # Target class
        if x.grad is not None:
            x.grad.zero_()
        # zero_gradients(x)
        output[I[1]].backward(retain_graph=True)
        cur_grad = x.grad.data.numpy().copy()

        # set new w and new f
        w = cur_grad - grad_orig
        f = (output[I[1]] - output[I[0]]).data.numpy()

        pert = abs(f) / np.linalg.norm(w.flatten())

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)

        if len(weights) > 0:
            r_i /= np.array(weights)

        # limit huge step
        r_i = alpha * r_i / np.linalg.norm(r_i)

        r_tot = np.float32(r_tot + r_i)

        pert_x = x_old + (1 + overshoot) * torch.from_numpy(r_tot)

        if len(bounds) > 0:
            pert_x = clip(pert_x, bounds[0], bounds[1])

        x = Variable(pert_x, requires_grad=True)

        output = net.forward(x)

        k_i = torch.tensor(np.argmax(output.data.cpu().numpy().flatten()))

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot
    pert_x = clip(pert_x, bounds[0], bounds[1])

    return orig_pred, k_i, pert_x.cpu(), loop_i
