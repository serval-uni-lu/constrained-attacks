import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mlc.constraints.constraints import Constraints
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from mlc.datasets.dataset_factory import load_dataset
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model_factory import load_model
from mlc.transformers.tab_scaler import TabScaler
from sklearn.metrics import matthews_corrcoef

from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.attacks.cta.cfab import CFAB
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from constrained_attacks.utils import compute_distance, fix_types


def run() -> None:

    dataset = load_dataset("lcld_201317_ds_time")
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()

    # x_train = x.iloc[splits["train"]]
    # y_train = y[splits["train"]]
    # x_test = x.iloc[splits["test"][:1000]]
    # y_test = y[splits["test"][:1000]]
    # x_val = x.iloc[splits["val"]]
    # y_val = y[splits["val"]]

    x_train = x.iloc[:350000]
    y_train = y[:350000]
    x_test = x.iloc[400000:]
    y_test = y[400000:]
    x_val = x.iloc[350000:400000]
    y_val = y[350000:400000]

    metadata = dataset.get_metadata(only_x=True)
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(x.values, x_type=metadata["type"])

    # Load model
    model_class = load_model("torchrln")
    save_path = "../mlc/data/models/lcld_torchrln.model"
    model = model_class.load_class(
        save_path, x_metadata=metadata, scaler=scaler
    )
    metric = create_metric("auc")
    auc = compute_metric(
        model,
        metric,
        x_test.values,
        np.array([1 - y_test, y_test]).T,
    )
    print("Test AUC: ", auc)

    metric = create_metric("accuracy")
    acc = compute_metric(model, metric, x_test.values, y_test)
    print("Test acc: ", acc)

    metric = create_metric("mcc")
    mcc = compute_metric(model, metric, x_test.values, y_test)
    print("Test mcc: ", mcc)

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
    EPS = 1
    attack = CFAB(
        constraints_attack,
        scaler,
        model_attack,
        # lambda x: model.predict_proba(scaler.transform(x)),
        verbose=True,
        steps=10,
        n_restarts=1,
        eps=EPS - EPS / 100,
        n_classes=2,
        norm="L2",
    )

    adv = []
    for i, batch_i in enumerate(np.array_split(np.arange(x_test.shape[0]), 5)):
        x_l = x_test.values[batch_i]
        adv_l = attack(
            torch.Tensor(x_l),
            torch.tensor(model.predict(x_l), dtype=torch.long),
        )
        adv_l = scaler.inverse_transform(adv_l)
        adv.append(adv_l)
        print(f"Batch {i} done.")

    adv = torch.cat(adv, dim=0)

    distance = compute_distance(
        scaler.transform(x_test.values),
        scaler.transform(adv.detach().numpy()),
        "2",
    )
    plt.hist(distance)
    plt.savefig("distance_hist2.png")
    plt.clf()
    plt.plot(np.sort(distance.flatten()))
    plt.savefig("distance_curve2.png")
    plt.clf()

    distance_avg = []
    acc_avg = []
    y_pred = model.predict(x_test.values)
    y_ok = y_pred == y_test
    mcc_avg = []
    for i, batch_i in enumerate(
        np.array_split(np.arange(x_test.shape[0]), 20)
    ):
        distance_avg.append(distance[batch_i].mean())
        acc_avg.append(y_ok[batch_i].mean())
        mcc_avg.append(matthews_corrcoef(y_test[batch_i], y_pred[batch_i]))

    acc_avg = np.array(acc_avg)
    distance_avg = np.array(distance_avg)
    mcc_avg = np.array(mcc_avg)

    # min max scale distance
    distance_avg = (distance_avg - distance_avg.min()) / (
        distance_avg.max() - distance_avg.min()
    )
    # min max scale acc
    acc_avg = (acc_avg - acc_avg.min()) / (acc_avg.max() - acc_avg.min())
    # min max scale mcc
    mcc_avg = (mcc_avg - mcc_avg.min()) / (mcc_avg.max() - mcc_avg.min())

    plt.plot(distance_avg, label="distance")
    plt.plot(acc_avg, label="acc")
    plt.plot(mcc_avg, label="mcc")
    plt.legend()
    plt.savefig("distance_acc_curve3.png")
    print("SAVED")

    # adv = fix_types(torch.Tensor(x_test.values), adv, metadata["type"])

    constraints_val = constraints_executor.execute(adv)
    constraints_ok = (constraints_val <= 0).float().mean()
    print(f"Constraints ok adv: {constraints_ok*100:.2f}%")

    model.device = "cpu"
    model.to_device()

    objective_calculator = ObjectiveCalculator(
        classifier=model.predict_proba,
        constraints=constraints,
        thresholds={
            # "misclassification": 0.5,
            "distance": EPS,
            "constraints": 0.01,
        },
        fun_distance_preprocess=scaler.transform,
    )
    print("Clean")
    print(
        objective_calculator.get_success_rate(
            x_test.values, y_test, x_test.values[:, np.newaxis, :]
        )
    )

    print("Before fix")
    print(
        objective_calculator.get_success_rate(
            x_test.values, y_test, adv.unsqueeze(1).detach().numpy()
        )
    )

    distance = objective_calculator.get_objectives_eval(
        x_test.values, y_test, adv.unsqueeze(1).detach().numpy()
    ).distance
    plt.hist(distance)
    plt.savefig("distance_hist.png")
    plt.clf()
    plt.plot(np.sort(distance.flatten()))
    plt.savefig("distance_curve.png")

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
    torch.set_warn_always(True)
    run()
