# Misc
# [1] Ballet, V., Renard, X., Aigrain, J., Laugel, T., Frossard, P., & Detyniecki, M. (2019). Imperceptible Adversarial Attacks on Tabular Data. NeurIPS 2019 Workshop on Robust AI in Financial Services: Data, Fairness, Explainability, Trustworthiness, and Privacy (Robust AI in FS 2019)
import numpy as np
from joblib import Parallel, delayed

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
# from torch.autograd.gradcheck import zero_gradients
from functools import partial
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
from mlc.metrics.metric_factory import create_metric


from trickster.search import a_star_search
from trickster.optim import CategoricalLpProblemContext
from trickster.optim import run_experiment
from trickster.domain.categorical import FeatureExpansionSpec
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from constrained_attacks.utils import (
    fix_equality_constraints,
    fix_immutable,
    fix_types,
)
from mlc.metrics.compute import compute_metric



class TricksterModel(object):
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric

    def score(self,X, y):
        self.last_adv = X
        # print(X.shape)
        # print(y.shape)
        pred = np.argmax(self.predict_proba(X), 1)
        acc = (y == pred).mean()
        # print("SCORE")
        return acc 

    def decision_function(self, x):

        output = self.model.predict_proba(x)
        return output

    def predict_proba(self, x):
        # print(f"Prediction for {len(x)} examples")
        return self.model.predict_proba(x)

    def grad(self, x, target_class=None):
        x = torch.Tensor(x)
        x.requires_grad = True
        output = self.model.get_logits(x.unsqueeze(0),with_grad=True)
        objective = torch.cat(
            [torch.zeros((output.shape[0], 1)), 
            torch.ones((output.shape[0], 1))],1)
        if target_class == 0:
            objective = 1-objective
        output.backward(objective)
        return x.grad.cpu().numpy()
    
    def parameters(self):
        return self.model.parameters()

    @property
    def training(self):
        return self.model.training

    def eval(self):
        self.model.eval()


class UCS(Attack):

    def __init__(
        self,
        constraints: Constraints,
        scaler: TabScaler,
        model: Model,
        steps: int = 20000,
        seed: int = 0,
        fix_equality_constraints_end: bool = True,
        verbose: bool = False,
        fun_distance_preprocess=lambda x: x,
        n_bins=150,
        model_name=None,
        **kwargs,
    ) -> None:
        super().__init__("UCS", model)
        self.constraints = constraints
        self.scaler = scaler
        self.steps = steps
        self.verbose = verbose
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.seed = seed
        self.fun_distance_preprocess = fun_distance_preprocess
        self.n_bins = n_bins
        # Add categorical features to immutable features as in [1]
        self.mutable_mask = self.constraints.mutable_features
        self.feature_types = self.constraints.feature_types
        self.feature_min = np.array(self.constraints.lower_bounds).astype(np.float_)
        self.feature_max = np.array(self.constraints.upper_bounds).astype(np.float_)

        # model_predict_probal = model.predict_proba
        # model = model.wrapper_model

        # if model_name == "tabnet":
        #     model = nn.Sequential(
        #         model,
        #         nn.Softmax(
        #             dim=1
        #         ),  # Softmax activation along the dimension 1 (assuming batch is dimension 0)
        #     )
        
        # model.predict_proba = model_predict_probal
        # self.model = model
        self.model = TricksterModel(model, create_metric("accuracy"))

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Overridden.
        """
        # self._check_inputs(images)

        x = images
        x_in = images.clone()
        # x = self.scaler.transform(x)

        x = x.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv = self.perturb(x, labels, cheap=True)

        # print(f"SHAPE {adv.shape}")
        # x = self.scaler.inverse_transform(x)
        # adv = self.scaler.inverse_transform(adv)

        adv = fix_types(x_in, adv, self.constraints.feature_types)
        adv = fix_immutable(x_in, adv, self.constraints.mutable_features)

        # print(f"Length constraints {len(self.constraints.relation_constraints)}")

        print(adv.shape)
        print(x.shape)
        if self.fix_equality_constraints_end:
            adv = fix_equality_constraints(self.constraints, adv)

        # print("CHANGES 1")
        # print(torch.max(torch.abs(x_in - adv)))
        # print("L2")
        # print(torch.cdist(x_in, adv, p=2.0))
        # print(np.linalg.norm(self.fun_distance_preprocess(x_in.detach().numpy()) - self.fun_distance_preprocess(adv[:,0,:].detach().numpy()), ord=2, axis=1).max())

        # print("Are closes")
        # print(np.allclose(x_in.detach().numpy(), adv[:,0,:].detach().numpy()))
        # print(np.abs(x_in.detach().numpy()- adv[:,0,:].detach().numpy()).max())


        return adv

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if hasattr(self, "scaler") and self.scaler is not None:
            inputs = self.scaler.inverse_transform(inputs)
        return super().get_logits(inputs, labels, *args, **kwargs)


    def perturb_one(self, x_in, y_in, cheap=True):
        

        bin_level = 150
        epsilon = 1

        expansion_specs, transformable_feature_idxs = get_expansions_specs(
            self.mutable_mask,
            self.feature_types,
            self.feature_min,
            self.feature_max,
            bin_level,
        )

        problem_ctx = CategoricalLpProblemContext(
            clf=self.model,
            target_class=0,
            target_confidence=0.5,
            lp_space="2",
            expansion_specs=expansion_specs,
            epsilon=epsilon,
            fun_distance_preprocess= get_preprocess(self.fun_distance_preprocess)
        )

        result = run_experiment(
            data=(x_in, y_in),
            problem_ctx=problem_ctx,
            transformable_feature_idxs=transformable_feature_idxs,
            # logger=logger,
            reduce_classifier=False
        )
        # print(result)
        # print(result["search_results"].shape)

        x_adv = self.model.last_adv[0]

        # print(f"Supposely best {self.model.predict_proba([x_adv])}")
        # print(result["search_results"]["x_adv_features"])
        # print(result["search_results"]["x_adv_features"].iloc[0].shape)
        x_adv = result["search_results"]["x_adv_features"].iloc[0]
        if x_adv is None:
            x_adv = x_in[0]

        # print(f"The best {self.model.predict_proba([x_adv])}")

        # print(result[x_adv_features].shape)

        return torch.tensor(x_adv)


    def perturb(self, x_in, y_in, cheap=True):
        x_in = x_in.numpy()
        y_in = y_in.numpy()

        torch.random.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        # x_out = []
        # for i in range(len(x_in)):
        #     print(f"Example {i}")
        #     x_out.append(self.perturb_one(x_in[i], y_in[i]))
        
        x_out = Parallel(n_jobs=20)(delayed(self.perturb_one)(
            x_in[i].reshape(1, -1), y_in[i].reshape(-1,1)
        ) for i in tqdm(range(len(x_in))))
        
        # print("CHANGES")
        # print(np.abs(np.stack(x_out) - x_in).max())

        return torch.stack(x_out)


def get_preprocess(fun_l):
    def preprocess_x(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            return fun_l(x)[0]
        else:
            return fun_l(x)

    return preprocess_x

    
def get_expansions_specs(
    mutable, feature_types, feature_min, feature_max, n_bins
):
    # print(f"mutable {mutable}")
    # print(f"feature_types {feature_types}")
    # print(f"feature_min {feature_min}")
    # print(f"feature_max {feature_max}")
    # print(f"n_bins {n_bins}")
    steps = (feature_max - feature_min) / n_bins
    steps[feature_types == "int"] = np.floor(steps[feature_types == "int"])
    steps[feature_types == "int"] = np.max([steps[feature_types == "int"], np.ones(len(steps[feature_types == "int"]))])
    expansions = []
    feature_idx = []
    for i, _ in enumerate(mutable):
        if mutable[i]:
            feature_idx.append(i)
            if feature_types[i] == "cat":
                expansions.append(
                    FeatureExpansionSpec(
                        [i],
                        partial(
                            expand_categorical, upper_bound=feature_max[i]
                        ),
                    )
                )
            else:
                expansions.append(
                    FeatureExpansionSpec(
                        [i],
                        partial(
                            expand_continous,
                            step_size=steps[i],
                            lower_bound=feature_min[i],
                            upper_bound=feature_max[i],
                        ),
                    )
                )

    return expansions, feature_idx


def expand_continous_increase(sample, feat_idxs, step_size, upper_bound):
    sub_sample = sample[feat_idxs]
    children = []

    x_new = sub_sample + step_size
    if x_new <= upper_bound:
        child = np.array(sample)
        child[feat_idxs] = x_new
        children.append(child)

    # print(F"increase {sample[feat_idxs]} -> {[c[feat_idxs] for c in children]}")
    return children


def expand_continous_decrease(sample, feat_idxs, step_size, lower_bound):
    sub_sample = sample[feat_idxs]
    children = []

    x_new = sub_sample - step_size
    if x_new >= lower_bound:
        child = np.array(sample)
        child[feat_idxs] = x_new
        children.append(child)
    # print(F"{feat_idxs} descrease {sample[feat_idxs]} -> {[c[feat_idxs] for c in children]}")
    return children


def expand_continous(sample, feat_idxs, step_size, lower_bound, upper_bound):
    children = []
    children.extend(
        expand_continous_increase(sample, feat_idxs, step_size, upper_bound)
    )
    children.extend(
        expand_continous_decrease(sample, feat_idxs, step_size, lower_bound)
    )
    return children


def expand_categorical(sample, feat_idxs, upper_bound):
    sub_sample = sample[feat_idxs]
    children = []

    for e in np.arange(upper_bound + 1).astype(np.int_):
        if e != sub_sample:
            child = np.array(sample)
            child[feat_idxs] = e
            children.append(child)

    # print(F"cat {sample[feat_idxs]} -> {[c[feat_idxs] for c in children]}")
    return children
