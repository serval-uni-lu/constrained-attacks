import json
import os
import re
from pathlib import Path

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

from constrained_attacks.graphics import (
    DPI,
    FONT_SCALE,
    _color_palette,
    _get_filename,
    _setup_legend,
    barplot,
    lineplot,
)

A1_PATH = "A1_all_20230905_2.csv"
A2_PATH = "A2_all_20230905_2.csv"

model_names = {
    "vime": "VIME",
    "deepfm": "DeepFM",
    "torchrln": "RLN",
    "tabtransformer": "TabTrans.",
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
    "steps",
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
        return "Robust"
    if "default" in path:
        return "Standard"
    if "subset" in path:
        return "Subset"
    if "dist" in path:
        return "Distribution"
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


def get_all_data():
    scenario_names = [
        "A1",
        "A2",
    ]
    out = get_data(
        [
            "scenario-a1v18-steps-2",
            "scenario-a2v18-steps-2",
        ],
        scenario_names,
        path="xp_steps.json",
    )
    df = pd.DataFrame(out)
    return df


def table_steps(df, name):
    df = df.copy()

    df = df[df["scenario_name"].isin(["A1", "A2", "B1", "B2"])]
    df = (
        df.groupby(GROUP_BY)["robust_acc"]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )

    df["robust_acc"] = df["mean"]

    df["mean_std"] = (
        "$"
        + df["mean"].map("{:.3f}".format)
        + "$"
        + "\\tiny{$\\pm "
        + (1.96 * df["sem"]).map("{:.3f}".format)
        + "$}"
    )

    df["Model"] = df["model_name_target"].map(model_names)
    df["Dataset"] = df["dataset_name"].map(dataset_names)
    df["Training"] = df["Model Target"]
    df["Cstr"] = df["constraints_access"].map({True: "Yes", False: "No"})
    df["Stp"] = df["steps"]
    df["Attack"] = df["attack_name"]

    df_all = df.copy()
    for attack in df["Attack"].unique():
        df = df_all[df_all["Attack"] == attack]
        name_l = f"{name}_{attack}"
        pivot = df.pivot(
            columns=["Dataset", "Model"],
            index=["Training", "Cstr", "Stp"],
            values=["mean_std"],
        )

        def order_series(list_x):
            ignore = ["Dataset", "Model"]
            ignore_list = [df[e].unique() for e in ignore]

            for e in ignore_list:
                if list_x[0] in e:
                    return list_x

            return list_x.map(sort_attack_name)

        pivot = pivot.sort_index(
            axis=1,
            ascending=[
                True,
                False,
                True,
            ],
            key=order_series,
        )
        pivot = pivot.sort_index(axis=0, ascending=[False, False, True])

        pivot.to_csv(_get_filename(f"{name_l}") + ".csv", float_format="%.3f")
        pivot.to_latex(
            _get_filename(f"{name_l}") + ".tex",
            float_format="%.3f",
            column_format="lll|" + "l" * len(pivot.columns),
            escape=False,
            multicolumn_format="c",
            multicolumn=True,
            multirow=True,
            caption=f"Robust accuracy with different \#step for {attack} attack.",
        )


def plot_all(df):
    table_steps(df, "steps")


def preprocess(df):
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

    for e in ["steps"]:
        df[e] = df[e].astype(int)

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


def new_run() -> None:

    df = get_all_data()
    df = preprocess(df)
    plot_all(df)
    # print(df)


if __name__ == "__main__":
    new_run()
