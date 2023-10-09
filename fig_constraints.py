import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import comet_ml
from mlc.logging.comet_config import (
    COMET_APIKEY,
    COMET_PROJECT,
    COMET_WORKSPACE,
)
from joblib import Parallel, delayed
from tqdm import tqdm

from constrained_attacks.graphics import (
    barplot,
    FONT_SCALE,
    _color_palette,
    _setup_legend,
    _get_filename,
    DPI,
    lineplot,
)
import matplotlib.pyplot as plt
import seaborn as sns

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
    "max_eps",
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
    ]
    out = get_data(
        [
            "scenario-a1-constraints",
        ],
        scenario_names,
        path="xp_constraints.json",
    )
    df = pd.DataFrame(out)
    return df


def table_constraints(df, name):
    df = df.copy()

    df = df[df["attack_name"] != "CAA"]

    df["attack_constraints_rate_steps"] = df[
        "attack_constraints_rate_steps"
    ].map(lambda x: json.loads(x)[-1])

    df_split = df["attack_constraints_rate_steps"].apply(
        lambda x: pd.Series(x)
    )

    df_split.columns = [f"$\\phi_{{{i+1}}}$" for i in range(df_split.shape[1])]

    out = []

    for e in df_split.columns:
        df_l = df.copy()
        df_l["Constraint"] = e
        df_l["constraints_rate"] = df_split[e].map("{:.3f}".format)
        out.append(df_l)

    df = pd.concat(out)
    df["Attack"] = df["attack_name"]
    df["Constraints Type"] = "TODO"

    index = ["Constraint", "Constraints Type"]
    pivot = df.pivot_table(
        index=index,
        columns=["Attack"],
        values=["constraints_rate"],
    )

    def order_series(list_x):
        return list_x.map(sort_attack_name)

    pivot = pivot.sort_index(
        axis=1,
        ascending=[
            True,
            True,
        ],
        key=order_series,
    )

    pivot.to_csv(_get_filename(f"{name}") + ".csv", float_format="%.3f")
    pivot.to_latex(
        _get_filename(f"{name}") + ".tex",
        float_format="%.3f",
        column_format="l" * len(index) + "|" + "l" * len(pivot.columns),
        escape=False,
        multicolumn_format="c",
        multicolumn=True,
        multirow=True,
        caption=f"TABLE",
    )


def plot_all(df):
    # table_eps(df, "eps")
    table_constraints(df, "constraints")
    return None


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
