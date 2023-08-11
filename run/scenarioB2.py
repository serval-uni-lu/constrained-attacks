"""
Scenario B2: Query based attacks without knowing domain constraints
Source and target models are the same; MoeVa only 
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
from run.scenarioA1 import run_experiment

def run(dataset_name: str, model_name: str, attacks_name: List[str] = None, max_eps: float = 0.1, subset: int = 1,
        batch_size: int = 1024, save_examples: int = 1, device: str = "cuda", custom_path: str = "",
        filter_class:int=None, n_jobs:int=-1, n_offsprings:int=100,n_gen:int=100):
    # Load data

    dataset = load_dataset(dataset_name)
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()
    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]

    if subset > 0 and subset < len(y_test):
        _,x_test,_, y_test = train_test_split(x_test, y_test, test_size = subset, stratify=y_test, random_state=42)
        class_imbalance = np.unique(y_test, return_counts=True)
        print("class imbalance",class_imbalance)

    else:
        subset =  len(y_test)


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
    assert constraints_ok > 0.9

    if constraints_ok < 1:
        x_test = x_test.iloc[(constraints_val <= 0.01).nonzero(as_tuple=True)[0].numpy()]
        y_test = y_test[(constraints_val <= 0.01).nonzero(as_tuple=True)[0].numpy()]
    print("--------- End of verification ---------")

    # In scneario B1, the attacker not aware of the constraints
    constraints = copy.deepcopy(dataset.get_constraints())
    constraints.relation_constraints = None
    constraints_eval = copy.deepcopy(dataset.get_constraints())
    
    attacks = {
               "moeva": (Moeva2, {"fun_distance_preprocess": scaler.transform,
               "n_jobs":n_jobs,"n_gen":args.get("n_gen",100),
               "n_offsprings": args.get("n_offsprings",100),"thresholds": {"distance": args.get("max_eps")}})
        
    }

    for attack_name in attacks_name:
        args = {"dataset_name": dataset_name, "model_name": model_name, "attack_name": attack_name, "subset": subset,
                "batch_size": batch_size, "max_eps": max_eps, "weight_path": weight_path, "n_gen":n_gen, "n_offsprings":n_offsprings}

        run_experiment(model, model, dataset, scaler, x_test, y_test, args, save_examples, filter_class=filter_class, n_jobs=n_jobs,
        constraints = constraints, ATTACKS=attacks,project_name="scenario_B2_v1", constraints_eval=constraints_eval)


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
    parser.add_argument("--attacks_name", type=str, default="moeva",
                        )
    parser.add_argument("--device", type=str, default="",
                        )
    parser.add_argument("--max_eps", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_examples", type=int, default=1)
    parser.add_argument("--filter_class", type=int, default=1)
    parser.add_argument("--n_offsprings", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    run(dataset_name=args.dataset_name, model_name=args.model_name, attacks_name=args.attacks_name.split("+"),
        subset=args.subset, custom_path=args.custom_path, filter_class=args.filter_class, n_jobs=args.n_jobs,
        batch_size=args.batch_size, save_examples=args.save_examples, max_eps=args.max_eps, device=args.device,
        n_gen=args.n_gen, n_offsprings=args.n_offsprings)
