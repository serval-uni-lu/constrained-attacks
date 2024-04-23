import os
import sys
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from constrained_attacks.typing import NDInt
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)

sys.path.append(".")
from mlc.logging import XP

import torch
import optuna
from optuna.trial import TrialState
from mlc.datasets.dataset_factory import load_dataset
from mlc.models.model_factory import load_model
from argparse import ArgumentParser, Namespace
from mlc.transformers.tab_scaler import TabScaler
from mlc.metrics.compute import compute_metric, compute_metrics
from mlc.metrics.metric_factory import create_metric
from mlc.dataloaders import get_custom_dataloader
import pandas as pd
import itertools

# Torch config to avoid crash on HPC
torch.multiprocessing.set_sharing_strategy("file_system")

CUSTOM_DATALOADERS = ["default", "subset", "madry", "dist"]


def run(
    dataset_name: str,
    model_name: str,
    custom_dataloader: str,
    constraints: int,
    filter_class: int,
    subset: int,
    eps: float,
    attacks: List[str],
) -> None:
    print("Subset optimization for {} on {}".format(model_name, dataset_name))

    dataset = load_dataset(dataset_name)
    metadata = dataset.get_metadata(only_x=True)

    x, y = dataset.get_x_y()
    splits = dataset.get_splits()

    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]

    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )

    model_class = load_model(model_name)

    weight_path = f"../models/mlc/best_models/{model_name}_{dataset_name}_{custom_dataloader}.model"
    steps = 10
    n_gen = 100
    n_offsprings = 100
    seed = 0
    x_1_path = get_adv_path(
        dataset_name,
        model_name,
        custom_dataloader,
        constraints,
        attacks[0],
        subset,
        eps,
        steps,
        n_gen,
        n_offsprings,
        seed,
    )
    x_2_path = get_adv_path(
        dataset_name,
        model_name,
        custom_dataloader,
        constraints,
        attacks[1],
        subset,
        eps,
        steps,
        n_gen,
        n_offsprings,
        seed,
    )
    if not os.path.exists(weight_path):
        print(
            f"{dataset_name}, {model_name}, {custom_dataloader}: {weight_path} Not found!"
        )
        return {}
    if not os.path.exists(x_1_path):
        print(
            f"{dataset_name}, {model_name}, {custom_dataloader}: {x_1_path} Not found!"
        )
        return {}
    if not os.path.exists(x_2_path):
        print(
            f"{dataset_name}, {model_name}, {custom_dataloader}: {x_2_path} Not found!"
        )
        return {}

    model = model_class.load_class(
        weight_path,
        x_metadata=metadata,
        scaler=scaler,
        force_device="cpu",
    )

    x_clean, y_clean = get_x_attack(
        x_test,
        y_test,
        dataset.get_constraints(),
        model,
        filter_class=filter_class,
        filter_correct=False,
        subset=subset,
    )

    # By construction the cache only contains adverarial that respected constraints and distance,
    # Hence we just have to check for the model prediction

    y_clean_success_idx = np.where(model.predict(x_clean.values) != y_clean)[0]

   

    x_1 = torch.load(x_1_path)
    x_2 = torch.load(x_2_path)

    y_1_success_idx = np.where(model.predict(x_1) != y_clean)[0]
    y_2_success_idx = np.where(model.predict(x_2) != y_clean)[0]

    print(len(y_clean_success_idx))
    print(len(y_1_success_idx))
    print(len(y_2_success_idx))

    # print(np.isin(y_clean_success_idx, y_1_success_idx).mean())
    assert np.isin(y_clean_success_idx, y_1_success_idx).all()
    assert np.isin(y_clean_success_idx, y_2_success_idx).all()

    y1_not_clean = np.setdiff1d(
        y_1_success_idx, y_clean_success_idx
    )
    y2_not_clean = np.setdiff1d(
        y_2_success_idx, y_clean_success_idx
        
    )

    intersection = np.intersect1d(
        y1_not_clean, y2_not_clean
    )

    out = {
        "dataset": dataset_name,
        "model": model_name,
        "training": custom_dataloader,
        attacks[0]: len(y1_not_clean),
        "intersection": len(intersection),
        attacks[1]: len(y2_not_clean),
    }

    return out


def get_adv_path(
    dataset_name: str,
    model_name: str,
    model_training: str,
    constraints: int,
    attack_name: str,
    subset: int,
    eps: float,
    steps: int,
    n_gen: int,
    n_offsprings: int,
    seed: int,
) -> str:
    adv_name = f"{dataset_name}_{model_name}_{model_training}_{constraints}_{attack_name}_{subset}_{eps}_{steps}_{n_gen}_{n_offsprings}_{seed}.pt"
    os.makedirs("./cache", exist_ok=True)
    adv_path = os.path.join("./cache", adv_name)
    return adv_path


def get_x_attack(
    x: pd.DataFrame,
    y: NDInt,
    constraints,
    model,
    filter_class=None,
    filter_correct=True,
    subset=0,
) -> Tuple[pd.DataFrame, NDInt]:
    if filter_class is not None:
        filter_l = y == filter_class
        x, y = x[filter_l], y[filter_l]

    if filter_correct:
        y_pred = model.predict(x.values)
        filter_l = y_pred == y
        x, y = x[filter_l], y[filter_l]

    constraints_executor = ConstraintsExecutor(
        AndConstraint(constraints.relation_constraints),
        PytorchBackend(),
        feature_names=constraints.feature_names,
    )
    constraints_val = constraints_executor.execute(torch.Tensor(x.values))
    constraints_ok_filter = constraints_val <= 1e-9
    constraints_ok_mean = constraints_ok_filter.float().mean()
    print(f"Constraints ok: {constraints_ok_mean * 100:.2f}%")

    if constraints_ok_mean < 1:
        filter_l = constraints_ok_filter.nonzero(as_tuple=True)[0].numpy()
        x, y = x.iloc[filter_l], y[filter_l]

    if subset > 0 and subset < len(y):
        _, x, _, y = train_test_split(
            x, y, test_size=subset, stratify=y, random_state=42
        )
        class_imbalance = np.unique(y, return_counts=True)
        print("class imbalance", class_imbalance)

    return x, y


if __name__ == "__main__":
    datasets = ["lcld_v2_iid"]
    models = ["tabtransformer", "stg", "tabnet", "torchrln", "vime"]
    # models = ["tabtransformer"]
    dataloaders = ["default"]
    constraints = 1
    filter_class = 1
    attacks = ["pgdl2", "apgd"]
    metric_list = []
    for dataset, model, dataloader in itertools.product(
        datasets, models, dataloaders
    ):
        subset = 1000 if dataset != "malware" else 100
        eps = 0.5 if dataset != "malware" else 5

        metric_dict = run(
            dataset,
            model,
            dataloader,
            constraints,
            filter_class,
            subset,
            eps,
            attacks,
        )
        metric_list.append(metric_dict)
    df = pd.DataFrame(metric_list)
    df.to_csv(f"intersection_apgd_{attacks[0]}_{attacks[1]}.csv")
