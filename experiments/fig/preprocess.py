import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from mlc.logging.comet_config import (
    COMET_APIKEY,
    COMET_PROJECT,
    COMET_WORKSPACE,
)
from tqdm import tqdm

from datetime import datetime
from .beautify_data import data_order, ordered_model_names, ordered_model_training_names


def generate_time_name():
    now = datetime.now()
    time_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    return time_name


model_names = {
    "vime": "VIME",
    "deepfm": "DeepFM",
    "torchrln": "RLN",
    "tabtransformer": "TabTransformer",
}

other_names = {
    "model_name": "Model",
}

dataset_names = {
    "url": "URL",
    "lcld_v2_iid": "LCLD",
    "ctu_13_neris": "CTU",
}


GROUP_BY = [
    "dataset_name",
    "attack_name",
    "model_name_target",
    "Model Source",
    "Model Target",
    "model_name",
    "scenario_name",
    "n_gen",
    "n_offsprings",
    "constraints_access",
]


def sort_attack_name(attack: str) -> int:
    # print(f"attack {attack}")
    for i, e in enumerate(
        ["Standard", "CPGD", "CAPGD", "MOEVA", "CAA", "CAA2", "CAA3"]
    ):
        # print(e)
        if e == attack:
            # print(i)
            return i


def attack_to_name(attack: str) -> str:
    attack = attack.upper()
    attack = attack.replace("PGDL2", "CPGD")
    attack = attack.replace("APGD", "CAPGD")
    return attack


def path_to_name(path: str) -> str:
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


def get_xp_data(xp, scenario_name):
    return {
        "scenario_name": scenario_name,
        **{e["name"]: e["valueCurrent"] for e in xp.get_parameters_summary()},
        **{e["name"]: e["valueCurrent"] for e in xp.get_metrics_summary()},
    }


def get_data(scenarios, scenario_names, path="xp.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    out = []
    for i, scenario in enumerate(scenarios):
        scenario_name = scenario_names[i]
        experiments = comet_ml.api.API(COMET_APIKEY).get(
            COMET_WORKSPACE, scenario
        )

        # Rewrite that for loop in joblib parallel
        out.extend(
            Parallel(n_jobs=-1)(
                delayed(get_xp_data)(experiments[i], scenario_name)
                for i in tqdm(range(len(experiments)))
            )
        )

    with open(path, "w") as f:
        json.dump(out, f)

    return out


def process_scenario(df, scenario_name, only_attack="CAA"):
    df = df.copy()
    if only_attack is not None:
        if df["attack_name"].unique().shape[0] > 1:
            df = df[df["attack_name"] == only_attack]

    df = df[df["Model Target"] != "Subset"]
    df = df[df["Model Target"] != "Distribution"]

    if scenario_name == "B":
        df = df[(df["n_gen"] == 100) & (df["n_offsprings"] == 100)]

    if scenario_name == "C":
        df = df[df["model_name"] != df["model_name_target"]]
        df = df[df["Model Source"] != "Robust"]
        idx_minimized_c = df.groupby(
            ["local_name", "model_name_target", "attack_name"]
        )["robust_acc"].idxmin()
        df = df.loc[idx_minimized_c]

    if scenario_name in ["D", "E"]:
        df = df[df["model_name"] != df["model_name_target"]]
        idx_minimized_c = df.groupby(["local_name", "model_name_target"])[
            "robust_acc"
        ].idxmin()
        df = df.loc[idx_minimized_c]

    if len(df) == 0:
        return None

    df_augment = df[
        (df["attack_name"] == df["attack_name"].unique()[0])
        & (df["scenario_name"] == df["scenario_name"].unique()[0])
    ].copy()
    df_augment["attack_name"] = "Standard"
    df_augment["robust_acc"] = df_augment["clean_acc"]
    if scenario_name in ["A", "B"]:
        df_augment["local_name"] = df_augment["Model Source"]
    else:
        df_augment["local_name"] = df_augment["Model Target"]

    df = pd.concat([df_augment, df])
    df["sort"] = df["local_name"].apply(lambda x: attack_plot_order[x])
    df = df.sort_values(by=["sort"])

    if scenario_name in ["C", "D", "E"]:
        df["model_name_graph"] = df["model_name_target"]
    else:
        df["model_name_graph"] = df["model_name"]

    return df


def process_scenario2(df, scenario_name, only_attack="CAA"):
    df = df.copy()
    if only_attack is not None:
        if df["attack_name"].unique().shape[0] > 1:
            df = df[df["attack_name"] == only_attack]

    df = df[df["Model Target"] != "Subset"]
    df = df[df["Model Target"] != "Distribution"]
    # df = add_local_name(df, scenario_name)

    if scenario_name == "B":
        df = df[(df["n_gen"] == 100) & (df["n_offsprings"] == 100)]

    if scenario_name in ["D", "E"]:
        df = df[df["Model Source"].isin(["Subset", "Distribution"])]

    if scenario_name in ["C", "D", "E"]:
        df = df[df["model_name"] != df["model_name_target"]]

        # group = df.groupby(
        #     ["local_name", "model_name", "model_name_target", "attack_name"]
        # )["robust_acc"]
        # idx_minimized_c = group.idxmin()

        # df = df.loc[idx_minimized_c]
        # df[["robust_acc_min", "robust_acc_mean", "robust_acc_max"]] = group.agg(
        #     ["min", "mean", "max"]
        # ).reset_index()[["min", "mean", "max"]]

    if len(df) == 0:
        return None

    df["sort"] = df["local_name"].apply(lambda x: attack_plot_order[x])
    df = df.sort_values(by=["sort"])

    if scenario_name in ["C", "D", "E"]:
        df["model_name_graph"] = df["model_name_target"]
    else:
        df["model_name_graph"] = df["model_name"]

    return df


def download_data(scenarios: Dict[str, List[str]], path: str):
    out = []
    for scenario_name, project_names in scenarios.items():
        for p in project_names:
            current = {}
            current["scenario_name"] = scenario_name
            current["project_name"] = p
            experiments = comet_ml.api.API(COMET_APIKEY).get(
                COMET_WORKSPACE, p
            )
            xp_list = []
            xp_list.extend(
                Parallel(n_jobs=-1)(
                    delayed(get_xp_data)(experiments[i], scenario_name)
                    for i in tqdm(range(len(experiments)))
                )
            )
            current["experiments"] = xp_list
            out.append(current)

    with open(path, "w") as f:
        json.dump(out, f)


TO_GET = {
    "AB": ["scenario-ab-url-v3"],
    "C": [],
    "D": [],
    "E": [],
}


def get_all_data():
    scenario_names = [
        "A1",
        "A2",
        # "A1_time",
        # "A2_time",
        "B1",
        "B2",
        "C1",
        "C2",
        "D1",
        "D2",
        "E1",
        "E2",
    ]
    out = get_data(
        [
            "scenario-a1v18",
            "scenario-a2v18",
            # "scenario-a1-time",
            # "scenario-a2-time",
            "scenario-b1v11",
            "scenario-b2v11",
            "scenario-c1v11",
            "scenario-c2v11",
            "scenario-d1v11",
            "scenario-dv11",
            "scenario-e1v11",
            "scenario-e2v11",
        ],
        scenario_names,
    )
    df = pd.DataFrame(out)
    return df


def preprocess_old(df):
    df = df.copy()

    # Remove unfinished experiments
    df = df[~df["mdc"].isnull()]

    # Replace caa by caa3

    df = df[df["attack_name"] != "CAA"]
    df = df[df["attack_name"] != "CAA2"]
    df["attack_name"] = df["attack_name"].map(
        lambda x: "caa" if x == "caa3" else x
    )

    # Retrieve time
    filter_attack_duration = ~(df["attack_duration_steps_sum"].isnull())
    df.loc[filter_attack_duration, "attack_duration"] = df.loc[
        filter_attack_duration, "attack_duration_steps_sum"
    ]

    # Parse types
    for e in ["mdc", "clean_acc", "n_gen", "n_offsprings", "attack_duration"]:
        df[e] = df[e].astype(float)

    df["constraints_access"] = df["constraints_access"].map(
        {"true": True, "false": False}
    )

    # Robust accuracy
    df["robust_acc"] = 1 - df["mdc"]

    # Parse model type

    model_names = ["vime", "deepfm", "torchrln", "tabtransformer"]
    pattern = "|".join(map(re.escape, model_names))
    df["model_name_target"] = df["weight_path_target"].str.extract(
        f"({pattern})", flags=re.IGNORECASE
    )
    df["attack_duration"].astype(float)

    # Rename key
    df["Scenario"] = df["scenario_name"]

    # Set model target and source (Type)

    df.loc[df["weight_path_target"].isna(), "weight_path_target"] = df.loc[
        df["weight_path_target"].isna(), "weight_path"
    ]
    df["Model Source"] = df["weight_path"].apply(path_to_name)
    df["Model Target"] = df["weight_path_target"].apply(path_to_name)

    # Beautify attack names

    df["attack_name"] = df["attack_name"].apply(attack_to_name)

    # Sort dataframe
    df["attack_name_sort"] = df["attack_name"].apply(sort_attack_name)
    df = df.sort_values(
        by=["scenario_name", "Model Source", "attack_name_sort"]
    )

    # Remove DeepFm
    df = df[df["model_name"] != "deepfm"]
    df = df[df["model_name_target"] != "deepfm"]

    return df


key_to_sort = {
    "Dataset": False,
    "Training": False,
    "Model": True,
    "Scenario": True,
    "Attack": True,
    "Constraints": False,
}


custom_sort = {
    "Attack": sort_attack_name,
}


def parse_json_data(json_data: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(json_data["experiments"])
    return df


def load_json_data(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


def add_no_attack(df: pd.DataFrame) -> pd.DataFrame:
    df_augment = df[df["attack_name"] == "pgdl2"].copy()
    df_augment["attack_name"] = "no_attack"
    df_augment["robust_acc"] = df_augment["clean_acc"]
    df_augment["mdc"] = 1 - df_augment["clean_acc"].astype(float)
    print("HERE")
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


def parse_model_arch(df: pd.DataFrame) -> pd.DataFrame:
    df["source_model_arch"] = df["weight_path"].apply(path_to_name)
    df["target_model_arch"] = df["weight_path_target"].apply(path_to_name)

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
    df["source_model_training"] = df["weight_path"].apply(path_to_name)
    df["target_model_training"] = df["weight_path_target"].apply(path_to_name)
    return df


def parse_is_constrained(df: pd.DataFrame) -> pd.DataFrame:
    df["is_constrained"] = df["constraints_access"].map(
        {"true": True, "false": False}
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
        print(col)
        df.loc[~df[col].isna(), f"{col}_order"] = df.loc[~df[col].isna(), col].apply(
            lambda x: mapping[x]
        )
    return df


def parse_data(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_model_arch(df)
    df = parse_model_training(df)
    df = parse_is_constrained(df)
    df = parse_n_iter(df)
    df = parse_robust_acc(df)
    df = col_rename(df)
    print(df["dataset"].unique())
    print(df.shape)
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


def run() -> None:
    path = "data/xp_results/data_2024_01_03_14_18_12.json"
    json_data = load_json_data(path)
    df = parse_json_data(json_data)
    df = augment_data(df)
    df = parse_data(df)
    print(df.shape)
    df = filter_columns(df)
    df = add_order(df)
    df.to_csv("data_tmp.csv", index=False)
    print(df.shape)


if __name__ == "__main__":
    run()
