"""
Scenario A1: Whitebox attacks with knowledge domain constraints
Source and target models are the same; CPGD, CAPGD, CMAPGD, CFAB, and CAA evaluation
"""

import os
import sys

import pandas as pd

sys.path.append(".")
sys.path.append("../ml-commons")
import copy
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from mlc.dataloaders.fast_dataloader import FastTensorDataLoader
from mlc.datasets.dataset_factory import load_dataset
from mlc.metrics.compute import (
    compute_metric,
    compute_metrics,
    compute_metrics_from_scores,
)
from mlc.metrics.metric_factory import create_metric
from mlc.models.model import Model
from mlc.models.model_factory import load_model
from mlc.transformers.tab_scaler import TabScaler
from sklearn.model_selection import train_test_split

from comet import XP
from constrained_attacks.attacks.moeva.moeva import Moeva2
from constrained_attacks.ensemble import Ensemble
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from constrained_attacks.typing import NDInt


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


def load_model_and_weights(
    dataset_name: str,
    model_name: str,
    custom_path: str,
    metadata: pd.DataFrame,
    scaler: TabScaler,
    device: str,
) -> Tuple[Model, str]:
    print(model_name)
    print(custom_path)

    splits_model_name = model_name.split(":")
    splits_custom_path = custom_path.split(":")
    if len(splits_model_name) > 1:
        models = [
            load_model_and_weights(
                dataset_name, m, p, metadata, scaler, device
            )[0]
            for m, p in zip(splits_model_name, splits_custom_path)
        ]
        return Ensemble(models), custom_path

    # Load model
    model_class = load_model(model_name)
    weight_path = (
        f"../models/constrained/{dataset_name}_{model_name}.model"
        if custom_path == ""
        else custom_path
    )

    if not os.path.exists(weight_path):
        print("{} not found. Skipping".format(weight_path))
        return

    force_device = device if device != "" else None
    model = model_class.load_class(
        weight_path,
        x_metadata=metadata,
        scaler=scaler,
        force_device=force_device,
    )

    return model, weight_path


def parse_target(model_name_target, custom_path_target, dataset_name):
    list_model_name_target = model_name_target.split(":")
    if custom_path_target != "":
        list_custom_path_target = custom_path_target.split(":")
    else:
        list_custom_path_target = [
            f"../models/constrained/{dataset_name}_{model_name_target}.model"
            for model_name_target in list_model_name_target
        ]
    return list_model_name_target, list_custom_path_target


def run(
    dataset_name: str,
    model_name: str,
    subset: int = 1,
    device: str = "cpu",
    custom_path: str = "",
    filter_class: int = None,
):
    # Load data

    dataset = load_dataset(dataset_name)
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()
    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]

    metadata = dataset.get_metadata(only_x=True)

    # Scaler
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )

    # Load
    model, weight_path = load_model_and_weights(
        dataset_name, model_name, custom_path, metadata, scaler, device
    )

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

    # Constraints check

    constraints = dataset.get_constraints()
    constraints_executor = ConstraintsExecutor(
        AndConstraint(constraints.relation_constraints),
        PytorchBackend(),
        feature_names=constraints.feature_names,
    )
    constraints_val = constraints_executor.execute(torch.Tensor(x_test.values))
    constraints_ok = (constraints_val <= 1e-9).float().mean()
    print(f"Constraints ok: {constraints_ok * 100:.2f}%")
    assert constraints_ok > 0.9

    print("--------- End of verification ---------")

    x_test_attack, y_test_attack = get_x_attack(
        x_test,
        y_test,
        dataset.get_constraints(),
        model,
        filter_class=filter_class,
        filter_correct=False,
        subset=subset,
    )
    metrics = [
        create_metric(e)
        for e in ["auc", "accuracy", "precision", "recall", "avg_score"]
    ]
    for xl, yl in zip([x_test], [y_test]):
        m_i = compute_metrics(model, metrics, xl, yl)
        print(m_i)

    for xl, yl in zip([x_test_attack], [y_test_attack]):
        m_i = compute_metrics(
            model,
            [create_metric(e) for e in ["accuracy", "avg_score"]],
            xl,
            yl,
        )
        print(m_i)

    print("Score")
    y_score = model.predict_proba(x_test)
    print(y_score.shape)

    # t_min = 0
    # t_max = 1
    # t = (t_min + t_max) / 2
    # for i in range(10):
    #     y_s_l = (y_score[:, 1] > t).astype(float)
    #     y_s_l = np.column_stack([1 - y_s_l, y_s_l])
    #     print(y_s_l.shape)
    #     recall = compute_metrics_from_scores(
    #         create_metric("recall"), y_test, y_s_l
    #     )
    #     if recall > 0.7:
    #         t_min = t
    #     else:
    #         t_max = t
    #     print(t, recall)
    #     t = (t_min + t_max) / 2

    # y_s_l = (y_score[:, 1] > t).astype(float)
    # y_s_l = np.column_stack([1 - y_s_l, y_s_l])
    # print(y_s_l.shape)
    # metrics_l = compute_metrics_from_scores(
    #     metrics, y_test, y_s_l
    # )
    # print(metrics_l)

    # model.epochs = 20
    # model.fit(
    #     x.iloc[splits["train"]],
    #     y[splits["train"]],
    #     x.iloc[splits["val"]],
    #     y[splits["val"]],
    # )
    # for xl, yl in zip([x_test], [y_test]):
    #     m_i = compute_metrics(model, metrics, xl, yl)
    #     print(m_i)

    # for xl, yl in zip([x_test_attack], [y_test_attack]):
    #     m_i = compute_metrics(
    #         model,
    #         [create_metric(e) for e in ["accuracy", "avg_score"]],
    #         xl,
    #         yl,
    #     )
    #     print(m_i)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training with Hyper-parameter optimization"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lcld_v2_iid",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tabtransformer",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        default="",
    )

    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--filter_class", type=int, default=None)

    args = parser.parse_args()

    run(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        subset=args.subset,
        custom_path=args.custom_path,
        filter_class=args.filter_class,
    )
