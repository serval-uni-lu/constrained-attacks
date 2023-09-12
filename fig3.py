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


def barplot(
    data,
    name,
    x,
    y,
    y_label="",
    hue=None,
    x_label="",
    fig_size=(6, 4),
    legend_pos="best",
    overlay_x=None,
    x_lim=None,
    y_lim=None,
    rotate_ticks=0,
    default_acc=None,
    robust_acc=None,
    **kwargs,
):
    plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=FONT_SCALE)

    palette = _color_palette(data, hue, overlay_x)

    if overlay_x and not hue:
        graph = sns.barplot(x=x, y=y, hue=hue, data=data, color=palette[0])
        graph = sns.barplot(
            x=overlay_x, y=y, hue=hue, data=data, color=palette[1]
        )

        if legend_pos:
            topbar = plt.Rectangle(
                (0, 0), 1, 1, fc=palette[0], edgecolor="none"
            )
            bottombar = plt.Rectangle(
                (0, 0), 1, 1, fc=palette[1], edgecolor="none"
            )
            plt.legend(
                [bottombar, topbar],
                [overlay_x, x],
                loc=legend_pos,
                prop={"size": FONT_SCALE * 12},
            )
    else:
        graph = sns.barplot(
            x=x, y=y, hue=hue, data=data, palette=palette, **kwargs
        )
        if default_acc is not None:
            graph.axhline(
                default_acc,
                linestyle="solid",
                color="red",
                linewidth=2,
                label="Standard",
            )
        if robust_acc is not None:
            graph.axhline(
                robust_acc,
                linestyle="dashed",
                color="red",
                linewidth=2,
                label="Robust",
            )
        _setup_legend(data, legend_pos, hue)

    plt.ylabel(y_label)
    plt.xlabel(x_label)

    if x_lim is not None and len(x_lim) == 2:
        plt.xlim(x_lim)

    if y_lim is not None and len(y_lim) == 2:
        plt.xlim(y_lim)

    plt.xticks(rotation=rotate_ticks)

    plt.savefig(_get_filename(name), dpi=DPI, bbox_inches="tight")
    plt.close("all")


def sort_attack_name(attack: str) -> int:
    print(f"attack {attack}")
    for i, e in enumerate(["CPGD", "CAPGD", "CAA"]):
        print(e)
        if e == attack:
            print(i)
            return i


def attack_to_name(attack: str) -> str:
    attack = attack.upper()
    attack = attack.replace("PGDL2", "CPGD")
    attack = attack.replace("APGD", "CAPGD")
    return attack


def path_to_name(path: str) -> str:
    print(path)
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


def one_figure(df: pd.DataFrame, name: str):
    df = df.sort_values(by=["scenario", "attack_name_sort"])
    default_acc = df[df["Model"] == "Standard"]["clean_acc"].values[0]
    robust_acc = df[df["Model"] == "Robust"]["clean_acc"].values[0]
    print(df["model_name"].values[0])
    barplot(
        df,
        name,
        x="name",
        y="mdc",
        y_label=df["model_name"].values[0],
        hue="scenario",
        legend_pos="upper left",
        rotate_ticks=67.5,
        default_acc=default_acc,
        robust_acc=robust_acc,
    )


def run():

    df_a1 = pd.read_csv(A1_PATH)
    df_a2 = pd.read_csv(A2_PATH)

    df_a1["mdc"] = 1 - df_a1["mdc"]
    df_a2["mdc"] = 1 - df_a2["mdc"]

    df_a1["scenario"] = "A1"
    df_a2["scenario"] = "A2"

    df = pd.concat([df_a1, df_a2])

    df["Model"] = df["weight_path"].apply(path_to_name)
    df["attack_name"] = df["attack_name"].apply(attack_to_name)

    df["name"] = df["attack_name"] + " + " + df["Model"]
    df = df[~df["mdc"].isnull()]
    df["attack_name_sort"] = df["attack_name"].apply(sort_attack_name)
    df = df.sort_values(by=["attack_name_sort"])

    for ds in df["dataset_name"].unique():
        df_l = df[(df["dataset_name"] == ds)]
        one_figure(df_l, f"{ds}_all")
        for model in df["model_name"].unique():
            df_l = df[(df["dataset_name"] == ds) & (df["model_name"] == model)]
            one_figure(df_l, f"{ds}_{model}")

    print("For debug")


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


def a_1_plot(df, name):

    df = df.copy()
    df = df[((df["scenario_name"] == "A1") | (df["scenario_name"] == "A2"))]

    df_augment = df[df["attack_name"] == "CPGD"].copy()
    df_augment["attack_name"] = "Standard"
    df_augment["robust_acc"] = df_augment["clean_acc"]

    df = pd.concat([df_augment, df])

    lineplot(
        df,
        name,
        x="attack_name",
        y="robust_acc",
        hue="scenario_name",
        style="Model Source",
    )


attack_plot_order = {
    "Standard": 0,
    "Robust": 1,
    "Robust 2": 2,
    "Standard 2": 3,
    "Robust 1": 4,
    "Standard 1": 5,
}


def add_local_name(df, scenario_name):
    df = df.copy()
    if scenario_name in ["A", "B"]:
        df["local_name"] = df["Model Source"]
    else:
        df["local_name"] = df["Model Target"]
    df["local_name"] = (
        df["local_name"]
        + " "
        + df["constraints_access"].map(lambda x: 1 if x else 2).map(str)
    )
    return df


def process_scenario(df, scenario_name):
    df = df.copy()
    if df["attack_name"].unique().shape[0] > 1:
        df = df[df["attack_name"] == "CAA"]

    df = df[df["Model Target"] != "Subset"]
    df = df[df["Model Target"] != "Distribution"]
    df = add_local_name(df, scenario_name)

    if scenario_name == "B":
        df = df[(df["n_gen"] == 100) & (df["n_offsprings"] == 100)]

    if scenario_name == "C":
        df = df[df["model_name"] != df["model_name_target"]]
        df = df[df["Model Source"] != "Robust"]
        idx_minimized_c = df.groupby(["local_name", "model_name_target"])[
            "robust_acc"
        ].idxmin()
        df = df.loc[idx_minimized_c]

    if scenario_name in ["D", "E"]:
        df = df[df["model_name"] == df["model_name_target"]]

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

    if scenario_name == "C":
        df["model_name_graph"] = df["model_name_target"]
    else:
        df["model_name_graph"] = df["model_name"]

    return df


def attack_plot(df, scenario_name, name):
    df = df.copy()
    df = df[df["scenario_name"].str.contains(scenario_name)]

    df = process_scenario(df, scenario_name)

    if df is None:
        return
    df_plot = df.pivot("local_name", "model_name_graph", "robust_acc")
    df_plot = df_plot.sort_values(
        by=["local_name"], key=lambda x: x.map(attack_plot_order)
    )
    sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="viridis")
    plt.savefig(_get_filename(name), dpi=DPI, bbox_inches="tight")
    plt.clf()


def threat_model_plot(df_all, name):

    df_all = df_all.copy()
    df_new = []
    for scenario_name in ["A", "B", "C", "D", "E"]:
        df = df_all[df_all["scenario_name"].str.contains(scenario_name)]

        df = process_scenario(df, scenario_name)
        df_new.append(df)
    df_new = pd.concat(df_new)

    # df_clean_idx = df["local_name"].isin(["Standard", "Robust"])
    # df_clean = df[df_clean_idx].drop_duplicates()
    # df = pd.concat([df_clean, df[~df_clean_idx]])

    for e in ["Standard", "Robust"]:
        df = df_new[df_new["local_name"].str.contains(e)].copy()

        print(df)
        df.loc[df["attack_name"] == "Standard", "scenario_name"] = ""
        df["graph_name"] = e + " " + df["scenario_name"]
        df["clean_equal"] = df["model_name_graph"] + df["graph_name"]
        df = df.drop_duplicates(subset=["clean_equal"])


        df_plot = df.pivot("graph_name", "model_name_graph", "robust_acc")
        df_plot = df_plot.sort_values(
            by=["graph_name"],
            # key=lambda x: x.map(attack_plot_order)
        )
        sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="viridis")
        plt.savefig(_get_filename(f"{name}_{e}"), dpi=DPI)
        plt.clf()



def get_all_data():
    scenario_names = [
        "A1",
        "A2",
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
            "scenario-a1v16",
            "scenario-a2v16",
            "scenario-b1v10",
            "scenario-b2v10",
            "scenario-c1v10",
            "scenario-c2v10",
            "scenario-d1v10",
            "scenario-dv10",
            "scenario-e1v10",
            "scenario-e2v10",
        ],
        scenario_names,
    )
    df = pd.DataFrame(out)
    return df


def preprocess(df):
    df = df.copy()
    df["Model Source"] = df["weight_path"].apply(path_to_name)
    df["Model Target"] = df["weight_path_target"].apply(path_to_name)
    df["attack_name"] = df["attack_name"].apply(attack_to_name)

    df = df[~df["mdc"].isnull()]
    df["attack_name_sort"] = df["attack_name"].apply(sort_attack_name)
    df = df.sort_values(
        by=["scenario_name", "Model Source", "attack_name_sort"]
    )

    for e in ["mdc", "clean_acc", "n_gen", "n_offsprings"]:
        df[e] = df[e].astype(float)
    df["robust_acc"] = 1 - df["mdc"]
    df["constraints_access"] = df["constraints_access"].map(
        {"true": True, "false": False}
    )
    model_names = ["vime", "deepfm", "torchrln", "tabtransformer"]
    pattern = "|".join(map(re.escape, model_names))
    df["model_name_target"] = df["weight_path_target"].str.extract(
        f"({pattern})", flags=re.IGNORECASE
    )
    return df


def plot_all(df):
    for ds in df["dataset_name"].unique():
        # for model in df["model_name"].unique():
        #     df_l = df[(df["dataset_name"] == ds) & (df["model_name"] == model)]
        #     a_1_plot(df_l, f"{ds}/a1_a2_{model}")
        df_l = df[(df["dataset_name"] == ds)]
        for e in ["A", "B", "C", "D", "E"]:
            attack_plot(df_l, e, f"{ds}/attack_{e}")
        for contraints in df["constraints_access"].unique():
            df_l = df[
                (df["dataset_name"] == ds)
                & (df["constraints_access"] == contraints)
            ]
            threat_model_plot(df_l, f"{ds}/threat_model_{1 if contraints else 2}")


def new_run() -> None:

    df = get_all_data()
    df = preprocess(df)
    plot_all(df)
    # print(df)


if __name__ == "__main__":
    new_run()
