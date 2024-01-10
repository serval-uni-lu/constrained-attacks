import json
import re
from typing import Any, Dict, List


import numpy as np
import pandas as pd


from datetime import datetime
from .beautify_data import (
    data_order,
    ordered_model_names,
)


def generate_time_name():
    now = datetime.now()
    time_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    return time_name


def path_to_model_training(path: str) -> str:
    # print(path)
    if isinstance(path, float) and np.isnan(path):
        return "Unknown"
    if "madry" in path:
        return "madry"
    if "default" in path:
        return "default"
    if "subset" in path:
        return "subset"
    if "dist" in path:
        return "dist"
    return "Unknown"


def parse_json_data(json_data: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(json_data["experiments"])
    return df


def load_json_data(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


def add_no_attack(df: pd.DataFrame) -> pd.DataFrame:
    filter_augment = (df["attack_name"] == "pgdl2") & df["scenario_name"].isin(
        ["AB"]
    )

    df_augment = df[filter_augment].copy()
    df_augment["attack_name"] = "no_attack"
    df_augment["robust_acc"] = df_augment["clean_acc"]
    df_augment["mdc"] = 1 - df_augment["clean_acc"].astype(float)
    return pd.concat([df_augment, df])


def add_weight_path_target(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["weight_path_target"].isna(), "weight_path_target"] = df.loc[
        df["weight_path_target"].isna(), "weight_path"
    ]
    return df


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    df = add_no_attack(df)
    # df = add_weight_path_target(df)
    return df


def parse_attack_duration(df: pd.DataFrame) -> pd.DataFrame:
    filter_attack_duration = ~(df["attack_duration_steps_sum"].isnull())
    df.loc[filter_attack_duration, "attack_duration"] = df.loc[
        filter_attack_duration, "attack_duration_steps_sum"
    ]
    df["attack_duration"].astype(float)
    return df


def parse_model_arch(df: pd.DataFrame) -> pd.DataFrame:
    model_names = ordered_model_names.keys()
    pattern = "|".join(map(re.escape, model_names))
    df["source_model_arch"] = df["weight_path"].str.extract(
        f"({pattern})", flags=re.IGNORECASE
    )
    df["target_model_arch"] = df["weight_path_target"].str.extract(
        f"({pattern})", flags=re.IGNORECASE
    )
    return df


def parse_model_training(df: pd.DataFrame) -> pd.DataFrame:
    df["source_model_training"] = df["weight_path"].apply(
        path_to_model_training
    )
    df["target_model_training"] = df["weight_path_target"].apply(
        path_to_model_training
    )
    return df


def parse_is_constrained(df: pd.DataFrame) -> pd.DataFrame:
    df["is_constrained"] = df["constraints_access"].map(
        {"true": True, "false": False, False: False, True: True}
    )
    return df


def parse_n_iter(df: pd.DataFrame) -> pd.DataFrame:
    df["n_iter"] = 0
    df.loc[df["attack_name"] == "moeva", "n_iter"] = df.loc[
        df["attack_name"] == "moeva", "n_gen"
    ]
    df.loc[
        df["attack_name"].isin(["caa3", "apgd", "pgdl2"]), "n_iter"
    ] = df.loc[df["attack_name"].isin(["caa3", "apgd", "pgdl2"]), "steps"]
    return df


def parse_robust_acc(df: pd.DataFrame) -> pd.DataFrame:
    df["robust_acc"] = 1 - df["mdc"].astype(float)
    return df


COL_RENAME = {
    "scenario_name": "scenario",
    "dataset_name": "dataset",
    "attack_name": "attack",
    "max_eps": "eps",
}


def col_rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COL_RENAME)


def add_order(df: pd.DataFrame) -> pd.DataFrame:
    for col, names in data_order.items():
        names = list(names.keys())
        mapping = {name: i for i, name in enumerate(names)}
        # print(col)
        # print(df[col].unique())
        df.loc[~df[col].isna(), f"{col}_order"] = df.loc[
            ~df[col].isna(), col
        ].apply(lambda x: mapping[x])

    for col in ["eps", "n_iter"]:
        df[f"{col}_order"] = df[col]

    return df


def parse_number_type(df: pd.DataFrame) -> pd.DataFrame:
    for e in ["n_offsprings", "n_gen", "steps", "seed", "n_iter"]:
        df[e] = df[e].replace(np.nan, "-1").astype(int)
    df["max_eps"] = df["max_eps"].astype(float)
    return df


def parse_data(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_model_arch(df)
    df = parse_model_training(df)
    df = parse_is_constrained(df)
    df = parse_n_iter(df)
    df = parse_robust_acc(df)
    df = parse_attack_duration(df)
    df = parse_number_type(df)
    df = col_rename(df)
    # print(df["dataset"].unique())
    # print(df.shape)
    df = add_order(df)
    df.to_csv("data_tmp.csv", index=False)
    return df


COL_FILTER = [
    "scenario",
    "dataset",
    "source_model_arch",
    "target_model_arch",
    "source_model_training",
    "target_model_training",
    "is_constrained",
    "attack",
    "eps",
    "n_iter",
    "n_offsprings",
    "seed",
    "attack_duration",
    "clean_acc",
    "robust_acc",
]


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[COL_FILTER]


def filter_source_equal_target(df: pd.DataFrame) -> pd.DataFrame:
    is_scenario_c = df["scenario"].isin(["C", "D", "E"])

    is_target_equal_source = df["target_model_arch"] == df["source_model_arch"]

    df = df[~(is_scenario_c & is_target_equal_source)]
    return df


def filter_d_e(df: pd.DataFrame) -> pd.DataFrame:
    is_scenario_d_e = df["scenario"].isin(["D", "E"])
    is_target_dist_subset = df["target_model_training"].isin(
        ["dist", "subset"]
    )
    df = df[~(is_scenario_d_e & is_target_dist_subset)]
    return df


def filter_values(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_source_equal_target(df)
    return df


def temp_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["source_model_arch"] != "saint")
        & (df["target_model_arch"] != "saint")
    ]
    return df


def filter_just_train(df: pd.DataFrame) -> pd.DataFrame:
    is_scenario_ab = df["scenario"] == "AB"
    is_dist_subset = df["source_model_training"].isin(["dist", "subset"])
    df = df[~(is_scenario_ab & is_dist_subset)]
    return df


def filter_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["robust_acc"].isna()]


def filter_not_default(df: pd.DataFrame) -> pd.DataFrame:
    filter_eps = ~((df["scenario"] == "AB_EPS") & (df["eps"] == 0.5))
    filter_iter_a = ~((df["scenario"] == "A_STEPS") & (df["n_iter"] == 10))
    filter_iter_b = ~(
        (df["scenario"] == "B_STEPS")
        & (df["n_iter"] == 100)
        & (df["n_offsprings"] == 100)
    )
    print(df["n_iter"].value_counts())

    for e in [filter_eps, filter_iter_a, filter_iter_b]:
        print((~e).sum())
    df = df[filter_eps & filter_iter_a & filter_iter_b]
    return df


def correction_n_iter(df: pd.DataFrame) -> pd.DataFrame:
    missing = df["n_iter"] == -1
    # Some n_iter are missing
    missing_sum = missing.sum()
    if missing_sum != 324:
        raise ValueError(f"n_iter is missing {missing_sum} values.")

    df.loc[missing, "n_iter"] = 10

    print(df["is_constrained"].value_counts())

    return df


def run() -> None:
    path = "data/xp_results/data_2024_01_10_22_01_33.json"
    json_data = load_json_data(path)
    df = parse_json_data(json_data)
    df = augment_data(df)
    df = parse_data(df)
    # print(df.shape)
    df = filter_columns(df)
    df = filter_values(df)
    df = filter_source_equal_target(df)
    df = filter_just_train(df)
    df = filter_d_e(df)
    df = filter_nan(df)
    df = correction_n_iter(df)
    df = filter_not_default(df)
    df = temp_filter(df)
    df = add_order(df)
    df.to_csv("data_tmp.csv", index=False)

    df = df.sort_values(by=["scenario", "dataset"])
    # print(df[["scenario", "dataset"]].value_counts(sort=False))
    # print(df[df["dataset"].isna()].shape)
    # print(df.shape)

    count = ["scenario", "dataset"]
    # df = df[df["scenario"] == "A_STEPS"]
    # df = df[df["n_iter"] == 0]
    print(df[count].sort_values(by=count).value_counts(sort=False))

    # print(df[df["n_iter"] == -1]["scenario"].value_counts())

    # print(df[df["n_iter"] == 0][count].value_counts())

    # DEBUG

    # df = df[df["dataset"] == "wids"]
    # print(df.shape)
    # df = df[df["is_constrained"] == True]
    # print(df["is_constrained"].value_counts())
    # df = df[df["target_model_arch"] == "tabnet"]
    # check = [
    #     "seed",
    #     "n_iter",
    #     "attack",
    #     "target_model_training",
    #     "is_constrained",
    # ]
    # check = [
    #     "seed",
    #     "n_iter",
    #     "attack",
    #     "target_model_training",
    # ]
    # df = df[df["scenario"] == "AB_EPS"]

    # print(df.shape)

    # print(df)

    # for model in ["tabnet", "stg", "torchrln", "tabtransformer", "vime"]:
    #     for seed in [0, 1, 2, 3, 4]:
    #         for n_iter in [20, 50, 100]:
    #             for attack in ["apgd", "pgdl2", "caa3"]:
    #                 for target_model_training in ["default", "madry"]:
    #                     for is_constrained in [True, False]:
    #                         n = df[
    #                             (df["seed"] == seed)
    #                             & (df["n_iter"] == n_iter)
    #                             & (df["attack"] == attack)
    #                             & (
    #                                 df["target_model_training"]
    #                                 == target_model_training
    #                             )
    #                             & (df["is_constrained"] == is_constrained)
    #                             & (df["target_model_arch"] == model)
    #                         ]
    #                         if n.shape[0] != 1:
    #                             print(
    #                                 attack,
    #                                 seed,
    #                                 model,
    #                                 target_model_training,
    #                                 is_constrained,
    #                                 n_iter,
    #                             )
    # print(df["eps"].value_counts())
    # print(df["attack"].value_counts())
    # print(df["target_model_training"].value_counts())
    # print(df["is_constrained"].value_counts())
    # print(df["target_model_arch"].value_counts())
    # print(df["seed"].value_counts())

    # print(df.head())
    # for model in ["tabnet", "stg", "torchrln", "tabtransformer", "vime"]:
    #     for seed in [0, 1, 2, 3, 4]:
    #         for eps in [0.25, 1.0]:
    #             for attack in ["apgd", "pgdl2", "caa3", "moeva"]:
    #                 for target_model_training in ["default", "madry"]:
    #                     for is_constrained in [True, False]:
    #                         n = df[
    #                             (df["seed"] == seed)
    #                             & (df["eps"] == eps)
    #                             & (df["attack"] == attack)
    #                             & (
    #                                 df["target_model_training"]
    #                                 == target_model_training
    #                             )
    #                             & (df["is_constrained"] == is_constrained)
    #                             & (df["target_model_arch"] == model)
    #                         ]
    #                         if n.shape[0] != 1:
    #                             print(
    #                                 attack,
    #                                 seed,
    #                                 model,
    #                                 target_model_training,
    #                                 is_constrained,
    #                                 eps,
    #                                 n.shape[0],
    #                             )


if __name__ == "__main__":
    run()
