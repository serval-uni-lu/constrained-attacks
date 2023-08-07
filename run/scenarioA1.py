"""
Scenario A1: Whitebox attacks with knowledge domain constraints
Source and target models are the same; CPGD, CAPGD, CMAPGD, CFAB, and CAA evaluation
"""

import os
import sys

sys.path.append(".")
sys.path.append("../ml-commons")
from comet import XP

import torch
import numpy as np
import copy
from argparse import ArgumentParser

from mlc.datasets.dataset_factory import load_dataset
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model_factory import load_model
from mlc.transformers.tab_scaler import TabScaler

from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import (
    AndConstraint
)
from constrained_attacks.attacks.cta.cpgdl2 import CPGDL2
from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.attacks.cta.cfab import CFAB
from constrained_attacks.attacks.cta.caa import ConstrainedAutoAttack
from constrained_attacks.attacks.moeva.moeva import Moeva2

from mlc.dataloaders.fast_dataloader import FastTensorDataLoader
from typing import List
import time

def run_experiment(model, dataset, scaler, x, y, args, save_examples: int = 1, xp_path="./data", filter_class=None, n_jobs=1):
    experiment = XP({**args,"filter_class":filter_class}, project_name="scenario_A1")

    save_path = os.path.join(xp_path, experiment.get_name())
    os.makedirs(save_path, exist_ok=True)

    attack_name = args.get("attack_name", "pgdl2")
    ATTACKS = {"pgdl2": (CPGDL2, {}), "apgd": (CAPGD, {}), "fab": (CFAB, {}),
               "moeva": (Moeva2, {"fun_distance_preprocess": scaler.transform,"n_jobs":n_jobs,
                                  "thresholds": {"distance": args.get("max_eps")}}),
               "caa": (ConstrainedAutoAttack, {"constraints_eval": copy.deepcopy(dataset.get_constraints()), })}

    attack_class = ATTACKS.get(attack_name, (CPGDL2, {}))

    # In scneario A1, the attacker is not aware of the constraints or the mutable features
    constraints = copy.deepcopy(dataset.get_constraints())
    attack_args = {"eps": args.get("max_eps"), "norm": "L2", **attack_class[1]}

    attack = attack_class[0](constraints=constraints, scaler=scaler, model=model.wrapper_model,
                             fix_equality_constraints_end=False, fix_equality_constraints_iter=False,
                             model_objective=model.predict_proba, **attack_args)

    device = model.device

    dataloader = FastTensorDataLoader(torch.Tensor(x.values).to(device),
        torch.tensor(y, dtype=torch.long).to(device), batch_size=args.get("batch_size"))

    for batch_idx, batch in enumerate(dataloader):
        metric = create_metric("auc")
        startt = time.time()
        adv_x = attack(batch[0], batch[1]).detach()
        endt = time.time()
        experiment.log_metric("attack_duration", endt-startt, step=batch_idx)

        filter_x, filter_y, filter_adv = batch[0], batch[1], adv_x

        if(filter_class is not None):
            filter = batch[1] ==filter_class
            filter_x,filter_y,filter_adv = batch[0][filter], batch[1][filter],adv_x[filter]

        test_auc = compute_metric(
            model,
            metric,
            batch[0],
            batch[1],
        )
        experiment.log_metric("clean_auc", test_auc, step=batch_idx)

        adv_auc = compute_metric(
            model,
            metric,
            adv_x,
            batch[1],
        )
        experiment.log_metric("adv_auc", adv_auc, step=batch_idx)

        eval_constraints = copy.deepcopy(dataset.get_constraints())
        constraints_executor = ConstraintsExecutor(
            AndConstraint(eval_constraints.relation_constraints),
            PytorchBackend(),
            feature_names=eval_constraints.feature_names,
        )
        constraints_val = constraints_executor.execute(adv_x)
        constraints_ok = (constraints_val <= 0).float().mean()
        experiment.log_metric("adv_constraints", constraints_ok, step=batch_idx)

        objective_calculator = ObjectiveCalculator(
            classifier=model.predict_proba,
            constraints=eval_constraints,
            thresholds={
                "distance": args.get("max_eps"),
                "constraints": 0.01,
            },
            norm="L2",
            fun_distance_preprocess=scaler.transform,
        )
        filter_adv = filter_adv.unsqueeze(1) if len(filter_adv.shape)<3 else filter_adv
        success_rate = objective_calculator.get_success_rate(
            filter_x.detach().numpy(),
            filter_y,
            filter_adv.unsqueeze(1).detach().numpy(),
        )

        experiment.log_metrics(vars(success_rate), step=batch_idx)

        if save_examples:
            adv_name = "adv_{}.pt".format(batch_idx)
            adv_path = os.path.join(save_path, adv_name)
            torch.save(adv_x.detach().cpu(),adv_path)
            experiment.log_asset(adv_name, adv_path)


def run(dataset_name: str, model_name: str, attacks_name: List[str] = None, max_eps: float = 0.1, subset: int = 1,
        batch_size: int = 1024, save_examples: int = 1, device: str = "cuda", custom_path: str = "",
        filter_class:int=None, n_jobs:int=-1):
    # Load data

    dataset = load_dataset(dataset_name)
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()
    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]

    if subset > 0:
        x_test = x_test[:subset]
        y_test = y_test[:subset]

    metadata = dataset.get_metadata(only_x=True)

    # Scaler
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )

    # Load model
    model_class = load_model(model_name)
    weight_path = f"../models/constrained/{dataset_name}_{model_name}.model" if custom_path == "" else custom_path

    if not os.path.exists(weight_path):
        print("{} not found. Skipping".format(weight_path))
        return

    force_device = device if device!="" else None
    model = model_class.load_class(weight_path, x_metadata=metadata, scaler=scaler, force_device=force_device)
    print("--------- Start of verification ---------")
    # Verify model

    metric = create_metric("auc")
    auc = compute_metric(
        model,
        metric,
        x_test.values,
        y_test,
    )
    print("Test AUC: ", auc)

    # Constraints

    constraints = dataset.get_constraints()
    constraints_executor = ConstraintsExecutor(
        AndConstraint(constraints.relation_constraints),
        PytorchBackend(),
        feature_names=constraints.feature_names,
    )
    constraints_val = constraints_executor.execute(torch.Tensor(x_test.values))
    constraints_ok = (constraints_val <= 0.01).float().mean()
    print(f"Constraints ok: {constraints_ok * 100:.2f}%")
    assert constraints_ok == 1

    print("--------- End of verification ---------")

    for attack_name in attacks_name:
        args = {"dataset_name": dataset_name, "model_name": model_name, "attack_name": attack_name, "subset": subset,
                "batch_size": batch_size, "max_eps": max_eps, "weight_path": weight_path}

        run_experiment(model, dataset, scaler, x_test, y_test, args, save_examples, filter_class=filter_class, n_jobs=n_jobs)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training with Hyper-parameter optimization"
    )
    parser.add_argument("--dataset_name", type=str, default="lcld_v2_iid",
                        )
    parser.add_argument("--model_name", type=str, default="tabtransformer",
                        )
    parser.add_argument("--custom_path", type=str, default="",
                        )
    parser.add_argument("--attacks_name", type=str, default="pgdl2",
                        )
    parser.add_argument("--device", type=str, default="",
                        )
    parser.add_argument("--max_eps", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_examples", type=int, default=1)
    parser.add_argument("--filter_class", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    run(dataset_name=args.dataset_name, model_name=args.model_name, attacks_name=args.attacks_name.split("+"),
        subset=args.subset, custom_path=args.custom_path, filter_class=args.filter_class, n_jobs=args.n_jobs,
        batch_size=args.batch_size, save_examples=args.save_examples, max_eps=args.max_eps, device=args.device)