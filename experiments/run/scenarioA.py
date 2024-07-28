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
import time
from argparse import ArgumentParser
from typing import List, Tuple
from dataclasses import dataclass, fields
import dataclasses
import numpy as np
import torch
from mlc.constraints.constraints_backend_executor import ConstraintsExecutor
from mlc.constraints.pytorch_backend import PytorchBackend
from mlc.constraints.relation_constraint import AndConstraint
from mlc.dataloaders.fast_dataloader import FastTensorDataLoader
from mlc.datasets.dataset_factory import load_dataset
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model import Model
from mlc.models.model_factory import load_model
from mlc.transformers.tab_scaler import TabScaler
from sklearn.model_selection import train_test_split
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveRespected,
)

from comet import LocalXp as XP

# from comet import LocalXp as XP

from constrained_attacks.attacks.cta.caa import (
    ConstrainedAutoAttack,
    ConstrainedAutoAttack2,
    ConstrainedAutoAttack3,
    ConstrainedAutoAttack4,
    ConstrainedMultiAttack,
    ConstrainedAutoAttack5,
)
from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.attacks.cta.capgd2 import CAPGD2
from constrained_attacks.attacks.cta.cfab import CFAB
from constrained_attacks.attacks.cta.cpgdl2 import CPGDL2
from constrained_attacks.attacks.moeva.moeva import Moeva2
from constrained_attacks.attacks.cta.lowprofool import LowProFool
from constrained_attacks.attacks.cta.ucs import UCS

from constrained_attacks.ensemble import Ensemble
from constrained_attacks.objective_calculator.cache_objective_calculator import (
    ObjectiveCalculator,
)
from constrained_attacks.typing import NDInt


def get_weights(dataset):
    x, y = dataset.get_x_y()
    weights = abs(x.corrwith(pd.Series(y)))
    weights[weights.isna()] = 0
    weights = weights / np.linalg.norm(weights)

    return weights.values


def run_experiment(
    model,
    model_eval,
    dataset,
    scaler,
    x,
    y,
    args,
    save_examples: int = 1,
    xp_path="./data/advs",
    filter_class=None,
    n_jobs=1,
    ATTACKS=None,
    constraints=None,
    project_name="scenario_A1_v11",
    constraints_eval=None,
    override_adv=None,
    seed: int = 0,
    steps: int = 10,
    save_adv: int = 0,
    x_opposite=None,
):
    experiment = XP(
        {**args, "filter_class": filter_class, "seed": seed, "steps": steps},
        project_name=project_name,
    )

    save_path = os.path.join(xp_path, experiment.get_name())
    os.makedirs(save_path, exist_ok=True)

    attack_name = args.get("attack_name", "pgdl2")

    if constraints is None:
        constraints = dataset.get_constraints()

    if constraints_eval is None:
        constraints_eval = copy.deepcopy(constraints)

    if ATTACKS is None:
        ATTACKS = {
            "pgdl2ijcai": (
                CPGDL2,
                {
                    "steps": steps,
                    "adaptive_eps": True,
                    "random_start": False,
                    "fix_constraints_ijcai": True,
                },
            ),
            "pgdl2org": (
                CPGDL2,
                {"steps": steps, "adaptive_eps": True, "random_start": False},
            ),
            "pgdl2rsae": (
                CPGDL2,
                {"steps": steps, "adaptive_eps": True, "random_start": True},
            ),
            "pgdl2nrsnae": (
                CPGDL2,
                {"steps": steps, "adaptive_eps": False, "random_start": False},
            ),
            "pgdl2": (
                CPGDL2,
                {"steps": steps, "adaptive_eps": False, "random_start": True},
            ),
            "apgd": (CAPGD, {"steps": steps}),
            "apgd2": (CAPGD2, {"steps": steps, "n_restarts": 2}),
            "apgd2-nrep": (
                CAPGD2,
                {
                    "steps": steps,
                    "n_restarts": 2,
                    "fix_equality_constraints_iter": False,
                },
            ),
            "apgd2-nini": (
                CAPGD2,
                {"steps": steps, "n_restarts": 1, "init_start": False},
            ),
            "apgd2-nran": (
                CAPGD2,
                {"steps": steps, "n_restarts": 1, "random_start": False},
            ),
            "apgd2-nbes": (
                CAPGD2,
                {"steps": steps, "n_restarts": 2, "best_restart": False},
            ),
            "apgd2-nada": (
                CAPGD2,
                {"steps": steps, "n_restarts": 2, "adaptive_eps": False},
            ),
            "apgd3": (
                CAPGD2,
                {"steps": steps, "n_restarts": 2, "best_restart": False},
            ),
            "apgd3-nrep": (
                CAPGD2,
                {
                    "steps": steps,
                    "n_restarts": 2,
                    "best_restart": False,
                    "fix_equality_constraints_iter": False,
                },
            ),
            "apgd3-nini": (
                CAPGD2,
                {
                    "steps": steps,
                    "n_restarts": 1,
                    "best_restart": False,
                    "init_start": False,
                },
            ),
            "apgd3-nran": (
                CAPGD2,
                {
                    "steps": steps,
                    "n_restarts": 1,
                    "best_restart": False,
                    "random_start": False,
                },
            ),
            "apgd3-nada": (
                CAPGD2,
                {
                    "steps": steps,
                    "n_restarts": 2,
                    "best_restart": False,
                    "adaptive_eps": False,
                },
            ),
            "fab": (CFAB, {}),
            "moeva": (
                Moeva2,
                {
                    "fun_distance_preprocess": scaler.transform,
                    "n_jobs": n_jobs,
                    "thresholds": {"distance": args.get("max_eps")},
                    "n_gen": args.get("n_gen"),
                    "n_offsprings": args.get("n_offsprings"),
                },
            ),
            "caa": (
                ConstrainedAutoAttack,
                {
                    "constraints_eval": constraints_eval,
                    "n_jobs": n_jobs,
                    "steps": steps,
                },
            ),
            "caa2": (
                ConstrainedAutoAttack2,
                {
                    "constraints_eval": constraints_eval,
                    "n_jobs": n_jobs,
                    "steps": steps,
                },
            ),
            "caa3": (
                ConstrainedAutoAttack3,
                {
                    "constraints_eval": constraints_eval,
                    "n_jobs": n_jobs,
                    "steps": steps,
                },
            ),
            "caa4": (
                ConstrainedAutoAttack4,
                {
                    "constraints_eval": constraints_eval,
                    "n_jobs": n_jobs,
                    "steps": steps,
                    "n_gen": args.get("n_gen"),
                    "n_offsprings": args.get("n_offsprings"),
                },
            ),
            "caa5": (
                ConstrainedAutoAttack5,
                {
                    "constraints_eval": constraints_eval,
                    "n_jobs": n_jobs,
                    "steps": steps,
                    "n_gen": args.get("n_gen"),
                    "n_offsprings": args.get("n_offsprings"),
                },
            ),
            "lowprofool": (
                LowProFool,
                {
                    "weights": get_weights(dataset),
                    "steps": steps,
                    "model_name": model.name,
                },
            ),
            "ucs": (
                UCS,
                {
                    "fun_distance_preprocess": scaler.transform,
                    "model_name": model.name,
                },
            ),
            "bfs": (
                UCS,
                {
                    "fun_distance_preprocess": scaler.transform,
                    "model_name": model.name,
                    "epsilon": 1000000,
                },
            ),
        }

    attack_class = ATTACKS.get(attack_name, (CPGDL2, {}))

    # In scneario A1, the attacker is aware of the constraints or the mutable features
    constraints = copy.deepcopy(constraints)
    attack_args = {
        "eps": args.get("max_eps"),
        "norm": "L2",
        "seed": seed,
        **attack_class[1],
    }

    model_attack = (
        model.wrapper_model
        if (not attack_name in ["moeva", "ucs", "bfs"])
        else model
    )

    attack = attack_class[0](
        constraints=constraints,
        scaler=scaler,
        model=model_attack,
        fix_equality_constraints_end=True,
        # fix_equality_constraints_iter=True,
        model_objective=model.predict_proba,
        **attack_args,
    )

    eval_constraints = copy.deepcopy(dataset.get_constraints())
    objective_calculator = ObjectiveCalculator(
        classifier=model_eval.predict_proba,
        constraints=eval_constraints,
        thresholds={
            "distance": args.get("max_eps"),
            # "constraints": 0.001,
        },
        norm="L2",
        fun_distance_preprocess=scaler.transform,
    )
    attack = ConstrainedMultiAttack(
        objective_calculator=objective_calculator, attacks=[attack]
    )

    device = model.device

    dataloader = FastTensorDataLoader(
        torch.Tensor(x.values).to(device),
        torch.tensor(y, dtype=torch.long).to(device),
        batch_size=args.get("batch_size"),
    )

    out = []
    for batch_idx, batch in enumerate(dataloader):
        metric = create_metric("auc")
        startt = time.time()

        if override_adv is not None:
            adv_x = override_adv[batch_idx]
        else:
            adv_x = attack(batch[0], batch[1]).detach()

        out.append(adv_x)

        if adv_x.shape == batch[0].shape:
            incorrect_index = torch.isnan(adv_x).any(1)
            adv_x[incorrect_index] = batch[0][incorrect_index]

        endt = time.time()
        experiment.log_metric("attack_duration", endt - startt, step=batch_idx)
        # for e in range(20):
        #     print(attack.attacks[0].__name__)

        auto_attack_metrics = attack
        if isinstance(attack.attacks[0], ConstrainedAutoAttack3) or isinstance(
            attack.attacks[0], ConstrainedAutoAttack4
        ):
            auto_attack_metrics = attack.attacks[0]._autoattack
            experiment.log_metric(
                "attack_constraints_rate_steps_inner",
                auto_attack_metrics.constraints_rate,
                step=batch_idx,
            )

        experiment.log_metric(
            "attack_duration_steps_sum",
            np.sum(auto_attack_metrics.attack_times),
            step=batch_idx,
        )
        experiment.log_metric(
            "attack_duration_steps",
            auto_attack_metrics.attack_times,
            step=batch_idx,
        )
        experiment.log_metric(
            "attack_acc_steps",
            auto_attack_metrics.robust_accuracies,
            step=batch_idx,
        )
        experiment.log_metric(
            "attack_constraints_rate_steps",
            attack.constraints_rate,
            step=batch_idx,
        )
        experiment.log_metric(
            "attack_distance_ok_rate", attack.distance_ok, step=batch_idx
        )
        for i, e in enumerate(auto_attack_metrics.mdc):
            ele = dataclasses.asdict(e)
            for key in ele:
                experiment.log_metric(f"{i}_{key}", ele[key])

        experiment.log_metric(
            "constraints_success_rate", auto_attack_metrics.constraints_rate
        )

        filter_x, filter_y, filter_adv = batch[0], batch[1], adv_x

        if filter_class is not None:
            filter = batch[1] == filter_class
            filter_x, filter_y, filter_adv = (
                batch[0][filter],
                batch[1][filter],
                adv_x[filter],
            )
        else:
            # we can't compute AUC if we filter one class
            test_auc = compute_metric(
                model_eval,
                metric,
                batch[0],
                batch[1],
            )
            experiment.log_metric("clean_auc", test_auc, step=batch_idx)

        acc = compute_metric(
            model_eval, create_metric("accuracy"), batch[0], batch[1]
        )
        experiment.log_metric("clean_acc", acc, step=batch_idx)

        # constraints_executor = ConstraintsExecutor(
        #     AndConstraint(eval_constraints.relation_constraints),
        #     PytorchBackend(),
        #     feature_names=eval_constraints.feature_names,
        # )
        # constraints_val = constraints_executor.execute(adv_x)
        # constraints_ok = (constraints_val <= 0).float().mean()
        # experiment.log_metric(
        #     "adv_constraints", constraints_ok, step=batch_idx
        # )

        filter_adv = (
            filter_adv.unsqueeze(1)
            if len(filter_adv.shape) < 3
            else filter_adv
        )
        adv_x = adv_x.unsqueeze(1) if len(adv_x.shape) < 3 else adv_x
        if len(filter_adv.shape) == 3:
            # for example for Moeva, we need first to extract the successful examples
            (
                success_attack_indices,
                success_adversarials_indices,
            ) = objective_calculator.get_successful_attacks_indexes(
                filter_x.detach().cpu().numpy(),
                filter_y.cpu(),
                filter_adv.detach().cpu().numpy(),
                max_inputs=1,
            )

            filtered_ = filter_x.detach().clone()
            filtered_[success_attack_indices] = filter_adv[
                success_attack_indices, success_adversarials_indices, :
            ]
            success_rate = objective_calculator.get_success_rate(
                filtered_.cpu().numpy(),
                filter_y.cpu(),
                filter_adv.detach().cpu().numpy(),
            )

            if filter_class is None:
                adv_all = batch[0].detach().clone()
                (
                    success_attack_indices_all,
                    success_adversarials_indices_all,
                ) = objective_calculator.get_successful_attacks_indexes(
                    batch[0].detach().cpu().numpy(),
                    batch[1].detach().cpu().numpy(),
                    adv_x.detach().cpu().numpy(),
                    max_inputs=1,
                )
                adv_all[success_attack_indices_all] = adv_x[
                    success_attack_indices_all,
                    success_adversarials_indices_all,
                    :,
                ]
                adv_auc = compute_metric(
                    model_eval,
                    metric,
                    adv_all,
                    batch[1],
                )
                experiment.log_metric("adv_auc", adv_auc, step=batch_idx)
        else:
            success_rate = objective_calculator.get_success_rate(
                filter_x.detach().numpy(),
                filter_y,
                filter_adv.detach().numpy(),
            )

            if filter_class is None:
                adv_auc = compute_metric(
                    model_eval,
                    metric,
                    adv_x,
                    batch[1],
                )
                experiment.log_metric("adv_auc", adv_auc, step=batch_idx)

        experiment.log_metrics(**vars(success_rate), step=batch_idx)

        if save_examples:
            x_adv_df = pd.DataFrame(adv_x[:, 0, :], columns=x.columns)
            path = "./tmp/log_asset.csv"
            for name, e in [
                ("x_test", x),
                ("x_opposite", x_opposite),
                ("x_adv", x_adv_df),
            ]:
                e.to_csv(path, index=False)
                experiment.log_asset(path, name)

            adv_name = "adv_{}.pt".format(batch_idx)
            adv_path = os.path.join(save_path, adv_name)
            torch.save(adv_x.detach().cpu(), adv_path)
            experiment.log_asset(adv_path, adv_name)

        experiment.flush()

    return out


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
    print("###############")
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


def write_cache(path, x_adv):
    x_adv = torch.cat(x_adv, dim=0)
    torch.save(x_adv, path)


def load_cache(path, batch_size):
    x_adv = torch.load(path)

    x_adv_out = []

    n_batch = len(x_adv) // batch_size + 1

    for i in range(n_batch):
        x_adv_out.append(x_adv[i * batch_size : (i + 1) * batch_size])

    return x_adv_out


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
    evaluate_constraints: bool,
) -> str:
    adv_name = f"{dataset_name}_{model_name}_{model_training}_{constraints}_{attack_name}_{subset}_{eps}_{steps}_{n_gen}_{n_offsprings}_{seed}.pt"
    # Evaluate constraints was not plan, backward compatibility
    if not evaluate_constraints:
        adv_name = f"{dataset_name}_{model_name}_{model_training}_{constraints}_{attack_name}_{subset}_{eps}_{steps}_{n_gen}_{n_offsprings}_{seed}_no-constraints-eval.pt"

    os.makedirs("./cache", exist_ok=True)
    adv_path = os.path.join("./cache", adv_name)
    return adv_path


def path_to_training(path):
    training = [
        "ctgan_madry",
        "cutmix_madry",
        "goggle_madry",
        "wgan_madry",
        "tablegan_madry",
        "tvae_madry",
        "ctgan",
        "cutmix",
        "goggle",
        "wgan",
        "tablegan",
        "tvae" "default",
        "madry",
        "subset",
        "dist",
    ]
    for t in training:
        if t in path:
            return t


def run(
    dataset_name: str,
    model_name: str,
    attacks_name: List[str] = None,
    max_eps: float = 0.1,
    subset: int = 1,
    batch_size: int = 1024,
    save_examples: int = 1,
    device: str = "cuda",
    custom_path: str = "",
    filter_class: int = None,
    n_jobs: int = -1,
    project_name="scenario",
    constraints_access=True,
    n_gen=100,
    n_offsprings=100,
    model_name_target=None,
    custom_path_target=None,
    seed: int = 0,
    steps: int = 10,
    load_adv: bool = False,
    save_adv: bool = False,
    evaluate_constraints: bool = True,
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
    if constraints_ok <= 0.9:
        for i, e in enumerate(constraints.relation_constraints):
            constraints_executor = ConstraintsExecutor(
                AndConstraint([e, e]),
                PytorchBackend(),
                feature_names=constraints.feature_names,
            )
            constraints_val = constraints_executor.execute(
                torch.Tensor(x_test.values)
            )
            constraints_ok = (constraints_val <= 1e-9).float().mean()
            print(
                f"Constraints {i} ok: {constraints_ok * 100:.2f}%, {constraints_val.max()}"
            )
            i_print = np.argmax(constraints_val.numpy())
            print(x_test["header_FileAlignment"].iloc[i_print])

    assert constraints_ok > 0.9

    print("--------- End of verification ---------")

    if save_examples > 0:
        x_opposite, _ = get_x_attack(
            x_test,
            y_test,
            dataset.get_constraints(),
            model,
            filter_class=1 - filter_class,
            filter_correct=False,
            subset=subset,
        )
    else:
        x_opposite = None

    x_test, y_test = get_x_attack(
        x_test,
        y_test,
        dataset.get_constraints(),
        model,
        filter_class=filter_class,
        filter_correct=False,
        subset=subset,
    )
    constraints = copy.deepcopy(dataset.get_constraints())
    constraints_eval = copy.deepcopy(dataset.get_constraints())
    if not constraints_access:
        constraints.relation_constraints = None

    if not evaluate_constraints:
        constraints_eval.relation_constraints = None

    if model_name_target is not None:
        list_model_name_target, list_custom_path_target = parse_target(
            model_name_target, custom_path_target, dataset_name
        )
    else:
        list_model_name_target = [model_name]
        list_custom_path_target = [weight_path]

    for attack_name in attacks_name:
        last_adv = None
        if load_adv:
            last_adv = load_cache(
                get_adv_path(
                    dataset_name,
                    model_name,
                    path_to_training(weight_path),
                    int(constraints_access),
                    attack_name,
                    subset,
                    max_eps,
                    steps,
                    n_gen,
                    n_offsprings,
                    seed,
                    evaluate_constraints,
                ),
                batch_size,
            )
        if os.path.exists(
            get_adv_path(
                    dataset_name,
                    model_name,
                    path_to_training(weight_path),
                    int(constraints_access),
                    attack_name,
                    subset,
                    max_eps,
                    steps,
                    n_gen,
                    n_offsprings,
                    seed,
                    evaluate_constraints
            )
        ):
            pass
            # exit(0)
        for target_idx, (
            model_name_target_l,
            custom_path_target_l,
        ) in enumerate(zip(list_model_name_target, list_custom_path_target)):
            args = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "attack_name": attack_name,
                "subset": subset,
                "batch_size": batch_size,
                "max_eps": max_eps,
                "weight_path": weight_path,
                "constraints_access": constraints_access,
                "n_gen": n_gen,
                "n_offsprings": n_offsprings,
                "weight_path_source": weight_path,
                "weight_path_target": custom_path_target_l,
            }
            print(model_name_target_l)

            model_target, weight_path_target = load_model_and_weights(
                dataset_name,
                model_name_target_l,
                custom_path_target_l,
                metadata,
                scaler,
                device,
            )

            # try:
            last_adv = run_experiment(
                model,
                model_target,
                dataset,
                scaler,
                x_test,
                y_test,
                args,
                save_examples,
                filter_class=filter_class,
                n_jobs=n_jobs,
                constraints=constraints,
                project_name=project_name,
                constraints_eval=constraints_eval,
                override_adv=last_adv,
                seed=seed,
                steps=steps,
                x_opposite=x_opposite,
            )

            if save_adv:
                write_cache(
                    get_adv_path(
                        dataset_name,
                        model_name,
                        path_to_training(weight_path),
                        int(constraints_access),
                        attack_name,
                        subset,
                        max_eps,
                        steps,
                        n_gen,
                        n_offsprings,
                        seed,
                        evaluate_constraints,
                    ),
                    last_adv,
                )


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
    parser.add_argument(
        "--attacks_name",
        type=str,
        default="pgdl2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
    )
    parser.add_argument("--max_eps", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_examples", type=int, default=1)
    parser.add_argument("--filter_class", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--project_name", type=str, default="scenario")
    parser.add_argument("--constraints_access", action="store_true")
    parser.add_argument(
        "--no-constraints_access",
        dest="constraints_access",
        action="store_false",
    )
    parser.set_defaults(constraints_access=True)
    parser.add_argument("--n_gen", type=int, default=100)
    parser.add_argument("--n_offsprings", type=int, default=100)
    parser.add_argument("--model_name_target", type=str, default=None)
    parser.add_argument("--custom_path_target", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save_adv", type=int, default=0)
    parser.add_argument("--load_adv", type=int, default=0)
    parser.add_argument(
        "--constraints_evaluation", action="store_true", default=True
    )
    parser.add_argument(
        "--no-constraints_evaluation",
        dest="constraints_evaluation",
        action="store_false",
    )

    args = parser.parse_args()

    run(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        attacks_name=args.attacks_name.split("+"),
        subset=args.subset,
        custom_path=args.custom_path,
        filter_class=args.filter_class,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        save_examples=args.save_examples,
        max_eps=args.max_eps,
        device=args.device,
        project_name=args.project_name,
        constraints_access=args.constraints_access,
        n_gen=args.n_gen,
        n_offsprings=args.n_offsprings,
        model_name_target=args.model_name_target,
        custom_path_target=args.custom_path_target,
        seed=args.seed,
        steps=args.steps,
        load_adv=args.load_adv != 0,
        save_adv=args.save_adv != 0,
        evaluate_constraints=args.constraints_evaluation,
    )
