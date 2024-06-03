from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd

from constrained_attacks.graphics import barplot, lineplot
from experiments.fig.beautify_data import (
    auto_beautify_values,
    beautify_col_name,
)
from experiments.fig.beautify_latex import beautify_latex
from experiments.fig.format import (
    percent_format,
    percent_format_diff,
    time_format,
)
from experiments.fig.table_helper import save_table, to_latex, to_latex_min_max

DATA_PATH = "./data_xp.csv"
OUT_ROOT = "./data/fig/nips-20240518/"
N_DATASET = 4
N_MODEL_ARCH = 5
N_SEEDS = 5
N_EPS = 4
N_ITER_SEARCH = 4
N_ITER_GRADIENT = 4
N_SEEDS_BUDGET = 5

GROUP_BY = [
    "dataset",
    "attack",
    "target_model_arch",
    "source_model_arch",
    "target_model_training",
    "source_model_training",
    "scenario",
    "n_iter",
    "n_offsprings",
    "is_constrained",
    "n_gen",
    "steps",
    "eps",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def table_default(
    df: pd.DataFrame,
    metric: str,
    with_clean_attack: bool = True,
    auto_sem: bool = True,
    bold_min: bool = True,
    columns=None,
    index=None,
) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = ["attack"]
    if index is None:
        index = ["dataset", "target_model_arch"]

    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate

    df = (
        df.groupby(
            GROUP_BY,
            sort=False,
        )[metric]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )

    # Beautify
    df = auto_beautify_values(df)

    if metric == "robust_acc":
        to_latex_min_max(
            df,
            index,
            bold_min=bold_min,
            bold_max=False,
            float_round=3,
            auto_sem=auto_sem,
        )
    elif metric == "attack_duration":
        df["bold"] = False
        df.loc[df["attack"].isin(["CAA", "MOEVA"]), "bold"] = (
            (
                df[
                    df["attack"].isin(
                        [
                            "CAA",
                            "MOEVA",
                        ]
                    )
                ]
            )
            .groupby(index)["mean"]
            .transform(lambda x: x == x.min())
        )
        to_latex(
            df,
            bold=df["bold"],
            custom_format="{:.0f}",
            auto_sem=auto_sem,
        )

    # Pivot

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=["mean", "sem", "latex"],
        sort=False,
        aggfunc="first",
    )

    # Beautify

    pivot.columns.rename(
        [beautify_col_name(e) for e in pivot.columns.names], inplace=True
    )
    pivot.index.rename(
        [beautify_col_name(e) for e in pivot.index.names], inplace=True
    )

    return pivot


def minus_clean(x, ref):
    filter_values = ref["dataset"] == x["dataset"]
    filter_values &= ref["source_model_arch"] == x["target_model_arch"]
    filter_values &= ref["source_model_training"] == x["target_model_training"]

    value = ref[filter_values]["robust_acc"].values[0]
    return x["robust_acc"] - value


def table_transferability(df: pd.DataFrame) -> None:
    metric = "robust_acc"

    df_clean_perfs = (
        df[df["attack"] == "no_attack"]
        .groupby(["dataset", "source_model_training", "source_model_arch"])[
            "robust_acc"
        ]
        .first()
        .reset_index()
    )

    filter_in = df["attack"] == "caa5"
    filter_in &= df["scenario"] == "caa_transferability"
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    df = df[filter_in]

    expected = N_DATASET * N_MODEL_ARCH * N_MODEL_ARCH * N_SEEDS
    check_n_samples(df, expected)

    df["robust_acc"] = df.apply(
        lambda x: minus_clean(x, df_clean_perfs), axis=1
    )

    index = ["dataset", "target_model_arch"]
    columns = ["source_model_arch"]

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate
    group_by = index + columns + ["attack"]
    df = (
        df.groupby(
            group_by,
            sort=False,
        )[metric]
        .agg(["mean", "std", "min", "max", "sem"])
        .reset_index()
    )

    # Beautify
    df = auto_beautify_values(df)

    bold = df.groupby(index)["mean"].transform(lambda x: x == x.min())

    df["bold"] = bold
    to_latex(
        df,
        bold=df["bold"],
    )
    to_latex(df, auto_sem=False, bold=df["bold"], custom_format=percent_format)

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=["mean", "sem", "latex"],
        sort=False,
        aggfunc="first",
    )

    # Beautify

    pivot.columns.rename(
        [beautify_col_name(e) for e in pivot.columns.names], inplace=True
    )
    pivot.index.rename(
        [beautify_col_name(e) for e in pivot.index.names], inplace=True
    )

    save_table(
        pivot,
        f"./{OUT_ROOT}/table_transferability.csv",
    )

    return pivot


def table_transferability_small(df: pd.DataFrame) -> None:
    metric = "robust_acc"

    filter_in_1 = df["attack"] == "caa5"
    filter_in_1 &= df["scenario"] == "caa_transferability"
    filter_in_2 = df["attack"].isin(["no_attack"])
    filter_in_2 &= df["scenario"] == "default"
    filter_in = filter_in_1 | filter_in_2
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    df = df[filter_in]

    expected = N_DATASET * N_MODEL_ARCH * N_MODEL_ARCH * (N_SEEDS + 1)
    check_n_samples(df, expected)

    index = ["dataset", "source_model_arch"]
    columns = ["target_model_arch"]

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate
    group_by = index + columns + ["attack"]
    df = (
        df.groupby(
            group_by,
            sort=False,
        )[metric]
        .agg(["mean", "std", "min", "max", "sem"])
        .reset_index()
    )

    df["mean"] = df["min"].copy()

    clean_filter = df["attack"] == "no_attack"
    clean_df = df[clean_filter].copy()
    clean_df["source_model_arch"] = "Clean"

    no_transfer_filter = df["source_model_arch"] == df["target_model_arch"]
    no_transfer_filter &= df["attack"] == "caa5"
    no_transfer_df = df[no_transfer_filter].copy()
    no_transfer_df["source_model_arch"] = "Whitebox"

    transfer_filter = ~(df["source_model_arch"] == df["target_model_arch"])
    transfer_filter &= df["attack"] == "caa5"

    transfer_df = df[transfer_filter].copy().reset_index(drop=True)
    transfer_df["min"] = transfer_df.groupby(["dataset", "target_model_arch"])[
        "mean"
    ].transform(lambda x: x == x.min())
    transfer_df = transfer_df[transfer_df["min"]].copy().reset_index(drop=True)
    transfer_df = (
        transfer_df.groupby(["dataset", "target_model_arch"])["mean"]
        .first()
        .reset_index()
    )

    # is_min_df = transfer_df[df["i"]].copy()
    transfer_df["source_model_arch"] = "Transfer"

    df = pd.concat(
        [clean_df, no_transfer_df, transfer_df],
    )
    # Beautify
    df = auto_beautify_values(df)

    df["bold"] = False

    to_latex(df, auto_sem=False, bold=df["bold"], custom_format=percent_format)

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=["mean", "sem", "latex"],
        sort=False,
        aggfunc="first",
    )

    # Beautify

    pivot.columns.rename(
        [beautify_col_name(e) for e in pivot.columns.names], inplace=True
    )
    pivot.index.rename(
        [beautify_col_name(e) for e in pivot.index.names], inplace=True
    )

    save_table(
        pivot,
        f"./{OUT_ROOT}/table_transferability_small.csv",
    )

    return pivot


def table_transferability_absolute(df: pd.DataFrame) -> None:
    metric = "robust_acc"

    df_clean_perfs = (
        df[df["attack"] == "no_attack"]
        .groupby(["dataset", "source_model_training", "source_model_arch"])[
            "robust_acc"
        ]
        .first()
        .reset_index()
    )

    filter_in = df["attack"] == "caa5"
    filter_in &= df["scenario"] == "caa_transferability"
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    df = df[filter_in]

    expected = N_DATASET * N_MODEL_ARCH * N_MODEL_ARCH * N_SEEDS
    check_n_samples(df, expected)

    df["robust_acc"] = df.apply(
        lambda x: minus_clean(x, df_clean_perfs), axis=1
    )

    index = ["dataset", "target_model_arch"]
    columns = ["source_model_arch"]

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate
    group_by = index + columns + ["attack"]
    df = (
        df.groupby(
            group_by,
            sort=False,
        )[metric]
        .agg(["mean", "std", "min", "max", "sem"])
        .reset_index()
    )

    # Beautify
    df = auto_beautify_values(df)

    bold = df.groupby(index)["mean"].transform(lambda x: x == x.min())

    df["bold"] = bold
    to_latex(
        df,
        bold=df["bold"],
    )
    to_latex(df, auto_sem=True, bold=df["bold"], custom_format=percent_format)

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=["mean", "sem", "latex"],
        sort=False,
        aggfunc="first",
    )

    # Beautify

    pivot.columns.rename(
        [beautify_col_name(e) for e in pivot.columns.names], inplace=True
    )
    pivot.index.rename(
        [beautify_col_name(e) for e in pivot.index.names], inplace=True
    )

    save_table(
        pivot,
        f"./{OUT_ROOT}/table_transferability_absolute.csv",
    )

    return pivot


def check_n_samples(df, expected):
    if df.shape[0] != expected:
        raise ValueError(
            f"Wrong number of samples, expected {expected} got {df.shape[0]}."
        )


def table_default_gradient(df: pd.DataFrame) -> None:
    df = df.copy()

    attacks = ["pgdl2ijcai", "lowprofool", "apgd3"]

    filter_in = df["attack"].isin(["no_attack"] + attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"] == "default"

    df = df[filter_in].copy()

    # +1 for the clean attack, there is 5 seeds although equal
    expected = N_DATASET * N_MODEL_ARCH * N_SEEDS * (len(attacks) + 1)
    check_n_samples(df, expected)

    table_robust_acc = table_default(
        df, "robust_acc", with_clean_attack=True, auto_sem=False
    )
    save_table(
        table_robust_acc,
        f"./{OUT_ROOT}/table_gradient.csv",
    )


def table_search(df: pd.DataFrame) -> None:

    attacks = ["ucs", "moeva", "apgd3", "caa5"]

    filter_in = df["attack"].isin(["no_attack"] + attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"] == "default"
    df = df[filter_in].copy()

    # attacks - UCS + No attack = attacks + 1 Seed for UCS
    expected = N_DATASET * N_MODEL_ARCH * (N_SEEDS * len(attacks) + 1)
    # check_n_samples(df, expected)

    table_robust_acc = table_default(df, "robust_acc", with_clean_attack=True)
    duration_filter = df["attack"] != "no_attack"
    table_duration = table_default(
        df[duration_filter].copy(),
        "attack_duration",
        with_clean_attack=False,
    )

    rob_and_dur = pd.concat([table_robust_acc, table_duration], axis=1)

    rob_and_dur.columns = pd.MultiIndex.from_tuples(
        [
            (col[0], "Robust Accuracy", col[1])
            for col in table_robust_acc.columns
        ]
        + [(col[0], "Duration", col[1]) for col in table_duration.columns]
    )
    # rob_and_dur = rob_and_dur.reset_index(level=1, drop=True)

    save_table(
        rob_and_dur,
        f"./{OUT_ROOT}/table_search.csv",
    )


def table_ablation(df: pd.DataFrame) -> None:
    df = df.copy()
    attacks = [
        "apgd3",
        "apgd3-nrep",
        "apgd3-nini",
        "apgd3-nran",
        "apgd3-nada",
    ]

    filter_in = df["attack"].isin(["no_attack"] + attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"].isin(["default", "capgd_ablation"])
    df = df[filter_in].copy()

    expected = N_DATASET * N_MODEL_ARCH * N_SEEDS * (len(attacks) + 1)
    check_n_samples(df, expected)

    table_robust_acc = table_default(df, "robust_acc", with_clean_attack=True)

    save_table(
        table_robust_acc,
        f"./{OUT_ROOT}/table_capgd_ablation.csv",
    )


def table_defense(df: pd.DataFrame) -> None:

    df_in = df.copy()

    df = df_in.copy()
    attacks = ["caa5"]
    filter_in = df["attack"].isin(["no_attack"] + attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"] == "default"

    df = df[filter_in].copy()

    expected = N_DATASET * N_MODEL_ARCH * N_SEEDS * (len(attacks) + 1)
    check_n_samples(df, expected)

    table_clean = table_default(
        df,
        "robust_acc",
        with_clean_attack=True,
        columns=["source_model_arch"],
        index=["dataset", "attack"],
    )

    df = df_in.copy()

    df = df_in.copy()
    attacks = ["caa5"]
    filter_in = df["attack"].isin(["no_attack"] + attacks)
    filter_in &= df["source_model_training"] == "madry"
    filter_in &= df["target_model_training"] == "madry"
    filter_in &= df["scenario"] == "default"

    df = df[filter_in].copy()

    expected = N_DATASET * N_MODEL_ARCH * N_SEEDS * (len(attacks) + 1)
    check_n_samples(df, expected)

    tab_defense = table_default(
        df,
        "robust_acc",
        with_clean_attack=True,
        columns=["source_model_arch"],
        index=["dataset", "attack"],
        bold_min=False,
    )
    save_table(
        tab_defense,
        f"./{OUT_ROOT}/table_defense.csv",
    )

    tab_defense = table_default(
        df,
        "robust_acc",
        with_clean_attack=True,
        auto_sem=False,
        bold_min=False,
        columns=["source_model_arch"],
        index=["dataset", "attack"],
    )

    tab_defense["latex"] += (
        tab_defense["mean"] - table_clean["mean"]
    ).applymap(percent_format_diff)

    save_table(
        tab_defense,
        f"./{OUT_ROOT}/table_defense_diff.csv",
    )


def plot_eps(df: pd.DataFrame) -> None:
    df = df.copy()
    attacks = [
        "caa5",
    ]

    filter_in = df["attack"].isin(attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"].isin(["default", "caa_eps"])
    # Do not have all seeds
    # filter_in &= df["seed"] == 0

    df = df[filter_in].copy()
    df = df.sort_values(by=[f"{e}_order" for e in ["source_model_arch"]])

    expected = N_DATASET * N_MODEL_ARCH * N_SEEDS_BUDGET * len(attacks) * N_EPS
    check_n_samples(df, expected)

    df["eps"] = df["eps"].copy()
    table = table_default(
        df,
        "robust_acc",
        with_clean_attack=True,
        auto_sem=True,
        index=["dataset", "source_model_arch"],
        columns=["eps"],
    )

    save_table(
        table,
        f"./{OUT_ROOT}/table_eps.csv",
    )

    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset].copy()
        df_dataset = auto_beautify_values(df_dataset)

        lineplot(
            path=f"./{OUT_ROOT}/plot_eps/plot_eps_{dataset}.pdf",
            data=df_dataset,
            x="eps",
            y="robust_acc",
            hue="source_model_arch",
            x_label="Epsilon",
            y_label="Robust Accuracy",
        )
        lineplot(
            path=f"./{OUT_ROOT}/plot_eps/plot_eps_{dataset}_no_legend.pdf",
            data=df_dataset,
            x="eps",
            y="robust_acc",
            hue="source_model_arch",
            x_label="Epsilon",
            y_label="Robust Accuracy",
            legend_pos=None,
        )


def plot_iter_gradient(df: pd.DataFrame) -> None:
    df = df.copy()
    attacks = [
        "caa5",
    ]

    filter_in = df["attack"].isin(attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"].isin(["default", "caa_iter_gradient"])
    # Do not have all seeds
    # filter_in &= df["seed"] == 0
    df = df[filter_in].copy()

    expected = (
        N_DATASET
        * N_MODEL_ARCH
        * N_SEEDS_BUDGET
        * len(attacks)
        * N_ITER_GRADIENT
    )
    check_n_samples(df, expected)

    df = df.sort_values(by=[f"{e}_order" for e in ["source_model_arch"]])

    df["steps_order"] = df["steps"].copy()
    table = table_default(
        df,
        "robust_acc",
        with_clean_attack=True,
        auto_sem=True,
        index=["dataset", "source_model_arch"],
        columns=["steps"],
    )

    save_table(
        table,
        f"./{OUT_ROOT}/table_iter_gradient.csv",
    )

    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset].copy()
        df_dataset = auto_beautify_values(df_dataset)

        lineplot(
            path=f"./{OUT_ROOT}/plot_iter_gradient/plot_iter_gradient_{dataset}.pdf",
            data=df_dataset,
            x="steps",
            y="robust_acc",
            hue="source_model_arch",
            x_label="Number of CAPGD iterations",
            y_label="Robust Accuracy",
            dashes=False,
        )
        lineplot(
            path=f"./{OUT_ROOT}/plot_iter_gradient/plot_iter_gradient_{dataset}_no_legend.pdf",
            data=df_dataset,
            x="steps",
            y="robust_acc",
            hue="source_model_arch",
            x_label="Number of CAPGD iterations",
            y_label="Robust Accuracy",
            dashes=False,
            legend_pos=None,
        )


def plot_iter_search(df: pd.DataFrame) -> None:
    df = df.copy()
    attacks = [
        "caa5",
    ]

    filter_in = df["attack"].isin(attacks)
    filter_in &= df["source_model_training"] == "default"
    filter_in &= df["target_model_training"] == "default"
    filter_in &= df["scenario"].isin(["default", "caa_iter_search"])
    # Do not have all seeds
    # filter_in &= df["seed"] == 0
    df = df[filter_in].copy()

    expected = (
        N_DATASET
        * N_MODEL_ARCH
        * N_SEEDS_BUDGET
        * len(attacks)
        * N_ITER_SEARCH
    )
    check_n_samples(df, expected)

    df = df.sort_values(by=[f"{e}_order" for e in ["source_model_arch"]])

    df["n_gen_order"] = df["n_gen"].copy()

    table = table_default(
        df,
        "robust_acc",
        with_clean_attack=True,
        auto_sem=True,
        index=["dataset", "source_model_arch"],
        columns=["n_gen"],
    )
    save_table(
        table,
        f"./{OUT_ROOT}/table_iter_search.csv",
    )

    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset].copy()
        df_dataset = auto_beautify_values(df_dataset)

        lineplot(
            path=f"./{OUT_ROOT}/plot_iter_search/plot_iter_search_{dataset}.pdf",
            data=df_dataset,
            x="n_gen",
            y="robust_acc",
            hue="source_model_arch",
            x_label="Number of MOEVA iterations",
            y_label="Robust Accuracy",
        )
        lineplot(
            path=f"./{OUT_ROOT}/plot_iter_search/plot_iter_search_{dataset}_no_legend.pdf",
            data=df_dataset,
            x="n_gen",
            y="robust_acc",
            hue="source_model_arch",
            x_label="Number of MOEVA iterations",
            y_label="Robust Accuracy",
            legend_pos=None,
        )


def run() -> None:
    df = load_data(DATA_PATH)

    table_transferability_small(df)
    table_transferability_absolute(df)
    # table_default_gradient(df)
    # table_search(df)
    # table_defense(df)
    # table_transferability(df)
    # table_ablation(df)
    # plot_eps(df)
    # plot_iter_gradient(df)
    # plot_iter_search(df)


if __name__ == "__main__":
    run()
