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
from tqdm import tqdm
# Torch config to avoid crash on HPC
torch.multiprocessing.set_sharing_strategy("file_system")

CUSTOM_DATALOADERS = ["default", "subset", "madry", "dist"]


def compare_two_adv(
    x1_not_clean,
    x2_not_clean,
):
    intersection = np.intersect1d(x1_not_clean, x2_not_clean)
    return len(x1_not_clean), len(x2_not_clean), len(intersection)


def compare_three_adv(
    a,
    b,
    c
):
    ab = np.intersect1d(a, b)
    ac = np.intersect1d(a, c)
    bc = np.intersect1d(b, c)
    abc = np.intersect1d(ab, c)
    return len(a), len(b), len(c), len(ab), len(ac), len(bc), len(abc)


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

    if not os.path.exists(weight_path):
        print(
            f"{dataset_name}, {model_name}, {custom_dataloader}: {weight_path} Not found!"
        )
        return {}

    attacks = attacks.copy()
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

    x_adv_clean_idx = np.where(model.predict(x_clean.values) != y_clean)[0]

    advs_idx = []
    for attack in attacks:
        if (dataset_name == "malware") and (model_name in ["vime", "tabtransformer", "tabnet"]):
            advs_idx.append(np.array([]))
            continue

        adv_idx_seed = []
        for seed in range(5):

            if (attack in ["ucs"]) and seed > 0:
                continue

            a_path =get_adv_path(
                dataset_name,
                model_name,
                custom_dataloader,
                constraints,
                attack,
                subset,
                eps,
                steps,
                n_gen,
                n_offsprings,
                seed,
            )
            if not os.path.exists(a_path):
                raise FileNotFoundError(
                    f"{dataset_name}, {model_name}, {custom_dataloader}: {a_path} Not found!"
                )
            else:
                adv = torch.load(a_path)
                # By construction the cache only contains adverarial that respected constraints and distance,
                # Hence we just have to check for the model prediction
                # print(f"HELLO {attack}")
                adv_idx = np.where(model.predict(adv) != y_clean)[0]
                # print(adv_idx)
                adv_idx_seed.append(adv_idx)

        if len(adv_idx_seed) > 0:
            adv_idx = np.unique(np.concatenate(adv_idx_seed))
            # print(len(adv_idx))
            assert np.isin(x_adv_clean_idx, adv_idx).all()
            adv_idx_not_clean = np.setdiff1d(adv_idx, x_adv_clean_idx)
            advs_idx.append(adv_idx_not_clean)

        else:
            print(f"Empty for {attack}")

    out_template = {
        "dataset": dataset_name,
        "model": model_name,
        "training": custom_dataloader,
    }
    # print(f"FUCCCCCK {len(advs_idx)}")

    out = []
    out2 = []
    for attack_i1, attack_i2 in itertools.combinations_with_replacement(range(len(attacks)), 2):
        out_local = out_template.copy()
        out_local["attack_1_name"] = attacks[attack_i1]
        out_local["attack_2_name"] = attacks[attack_i2]
        value_1, value_2, intersection = compare_two_adv(
            advs_idx[attack_i1], advs_idx[attack_i2]
        )
        out_local["attack_1_count"] = value_1
        out_local["attack_2_count"] = value_2
        out_local["intersection"] = intersection
        out.append(out_local)

    for attack_i1, attack_i2, attack_i3 in itertools.combinations_with_replacement(range(len(attacks)), 3):
        out_local = out_template.copy()
        out_local["attack_1_name"] = attacks[attack_i1]
        out_local["attack_2_name"] = attacks[attack_i2]
        out_local["attack_3_name"] = attacks[attack_i3]
        a, b, c, ab, ac, bc, abc = compare_three_adv(
            advs_idx[attack_i1],
            advs_idx[attack_i2],
            advs_idx[attack_i3]
        )
        computed = {
            "a": a,
            "b": b,
            "c": c,
            "ab": ab,
            "ac": ac,
            "bc": bc,
            "abc": abc
        }
        out_local = {**out_local, **computed}
        out2.append(out_local)
    
    print(len(out))

    return out, out2

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
    prefix = "cache"
    # if (dataset == "malware") and (attack_name == "moeva"):
    #     prefix = "cache.kdd24"
    if (attack_name == "pgdl2ijcai") or (attack_name == "lowprofool") or (attack_name == "moeva") or (attack_name == "ucs"):
        prefix = "cache_bis/cache.bk20240425-ecai24"
        prefix = "cache"
        prefix = "/scratch/users/tsimonetto/cache_not_folder/cache.bk20240425-ecai24"
    # os.makedirs("./cache", exist_ok=True)
    adv_path = os.path.join(f"{prefix}", adv_name)
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
    datasets = ["url", "lcld_v2_iid", "wids",  "ctu_13_neris"]
    models = ["tabtransformer", "stg", "tabnet", "torchrln", "vime"]
    # models = ["tabtransformer"]
    dataloaders = ["default"]
    constraints = 1
    filter_class = 1
    attacks = ["apgd3","apgd3-nrep", "apgd3-nini", "apgd3-nran", "apgd3-nada"]
    # attacks = ["apgd3","pgdl2ijcai", "lowprofool",]
    # attacks = ["apgd3","moeva", "ucs",]
    metric_list = []
    metric_list3 = []
    for dataset, model, dataloader in tqdm(itertools.product(
        datasets, models, dataloaders
    ), total=len(datasets) * len( models)* len( dataloaders)):
        subset = 1000 if dataset != "malware" else 100
        eps = 0.5 if dataset != "malware" else "5.0"

        metric_dict, metric_dict3 = run(
            dataset,
            model,
            dataloader,
            constraints,
            filter_class,
            subset,
            eps,
            attacks,
        )
        metric_list.extend(metric_dict)
        metric_list3.extend(metric_dict3)
    df = pd.DataFrame(metric_list)
    suffix = "cagpd3_ablation"
    # suffix = "gradient"
    # suffix = "search"
    df.to_csv(f"intersection_{suffix}.csv")
    df = pd.DataFrame(metric_list3)
    df.to_csv(f"intersection3_{suffix}.csv")
