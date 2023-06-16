import os
import numpy as np

from mlc.datasets.dataset_factory import load_dataset
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model_factory import load_model
from mlc.transformers.tab_scaler import TabScaler
import torch
import torch.nn as nn

from constrained_attacks.attacks.cta.capgd import CAPGD
from mlc.constraints.constraints import Constraints
from mlc.transformers.tab_scaler import TabScaler
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint

from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)

from constrained_attacks.utils import fix_types


def run() -> None:

    dataset = load_dataset("url")
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()

    x_train = x.iloc[splits["train"]]
    y_train = y[splits["train"]]
    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]
    x_val = x.iloc[splits["val"]]
    y_val = y[splits["val"]]
    metadata = dataset.get_metadata(only_x=True)
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(x.values, x_type=metadata["type"])

    # Load model
    model_class = load_model("torchrln")
    save_path = "../mlc/data/models/url_torchrln.model"
    model = model_class.load_class(save_path, x_metadata=metadata)
    metric = create_metric("auc")
    auc = compute_metric(
        model,
        metric,
        scaler.transform(x_test.values),
        np.array([1 - y_test, y_test]).T,
    )
    print("Test AUC: ", auc)

    metric = create_metric("accuracy")
    acc = compute_metric(
        model, metric, scaler.transform(x_test.values), y_test
    )
    print("Test acc: ", acc)

    # Constraints

    constraints = dataset.get_constraints()
    constraints_executor = ConstraintsExecutor(
        AndConstraint(constraints.relation_constraints),
        PytorchBackend(),
        feature_names=constraints.feature_names,
    )
    constraints_val = constraints_executor.execute(torch.Tensor(x_test.values))
    constraints_ok = (constraints_val <= 0).float().mean()
    print(f"Constraints ok: {constraints_ok*100:.2f}%")

    # Attack

    model_attack = nn.Sequential(
        model.model.cpu(),
    )

    constraints_attack = dataset.get_constraints()
    # constraints_attack.relation_constraints = None
    EPS = 8/255
    attack = CAPGD(
        constraints_attack,
        scaler,
        model_attack,
        lambda x: model.predict_proba(scaler.transform(x)),
        verbose=True,
        steps=10,
        n_restarts=1,
        eps=EPS - EPS/100,
        loss="ce"
    )

    adv = attack(
        torch.Tensor(x_test.values),
        torch.tensor(y_test, dtype=torch.long),
    )
    adv = scaler.inverse_transform(adv)
    # adv = fix_types(torch.Tensor(x_test.values), adv, metadata["type"])

    constraints_val = constraints_executor.execute(adv)
    constraints_ok = (constraints_val <= 0).float().mean()
    print(f"Constraints ok: {constraints_ok*100:.2f}%")

    model.device = "cpu"
    model.to_device()
    
    objective_calculator = ObjectiveCalculator(
        classifier=lambda x: model.predict_proba(scaler.transform(x)),
        constraints=constraints,
        thresholds={
            # "misclassification": 0.5,
            "distance": EPS,
            "constraints": 0.0,
        },
        fun_distance_preprocess=scaler.transform,
    )
    print("Before fix")
    print(
        objective_calculator.get_success_rate(
            x_test.values, y_test, adv.unsqueeze(1).detach().numpy()
        )
    )
    print("After fix")
    print(
        objective_calculator.get_success_rate(
            x_test.values,
            y_test,
            fix_types(torch.Tensor(x_test.values), adv, metadata["type"])
            .unsqueeze(1)
            .detach()
            .numpy(),
            recompute=True,
        )
    )


if __name__ == "__main__":
    run()
