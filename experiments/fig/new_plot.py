from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from constrained_attacks.graphics import barplot
from experiments.fig.beautify_latex import beautify_latex

from .beautify_data import beautify_col_name, data_order

DATA_PATH = "./data_xp.csv"
OUT_ROOT = "./data/fig/nips-20240513/"

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
    "eps",
]


def percent_format(x: float) -> str:
    x = x * 100
    return f"{x:.1f}"


def percent_format_diff(x: float) -> str:
    x = x * 100
    return "\\tiny{$" + f"{x:+.1f}" + "$}"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def auto_beautify_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in data_order:
            df[col] = df[col].map(data_order[col])
    return df


def to_latex_min_max(
    df: pd.DataFrame,
    index: List[str],
    bold_min: bool = False,
    bold_max: bool = False,
) -> None:
    df["is_max"] = df.groupby(index)["mean"].transform(lambda x: x == x.max())
    df["is_min"] = df.groupby(index)["mean"].transform(lambda x: x == x.min())

    df["bold"] = False
    bold = df["bold"].copy()
    if bold_min:
        bold = bold | df["is_min"]
    if bold_max:
        bold = bold | df["is_max"]

    to_latex(df, bold)


def to_latex(
    df: pd.DataFrame,
    bold: pd.Series,
    in_col: str = "mean",
    out_col: str = "latex",
    auto_sem: bool = True,
    sem: bool = True,
    custom_format: Union[str, Callable] = percent_format,
) -> None:

    if False:
        df["no_sem"] = True
        no_sem = df["no_sem"]
    else:
        if auto_sem:
            no_sem = df["attack"] == "no_attack"
        else:
            df["no_sem"] = True
            no_sem = df["no_sem"]

    if isinstance(custom_format, str):
        custom_format = custom_format.format

    # df["bold"] = False
    # bold = df["bold"].copy()
    # df.drop(columns="bold", inplace=True, errors="ignore")

    df.loc[bold, out_col] = (
        "$\\mathbf{"
        + df.loc[bold, in_col].map(custom_format)
        + "}$"
        + "\\tiny{$\\mathbf{\\pm "
        + (1.96 * df.loc[bold, "sem"]).map(custom_format)
        + "}$}"
    )
    df.loc[~bold, out_col] = (
        "$"
        + df.loc[~bold, in_col].map(custom_format)
        + "$"
        + "\\tiny{$\\pm "
        + (1.96 * df.loc[~bold, "sem"]).map(custom_format)
        + "$}"
    )
    df.loc[no_sem & bold, out_col] = (
        "$\\mathbf{" + df.loc[no_sem & bold, in_col].map(custom_format) + "}$"
    )

    df.loc[no_sem & ~bold, out_col] = (
        "$" + df.loc[no_sem & ~bold, in_col].map(custom_format) + "$"
    )


def table_AB(
    df: pd.DataFrame, metric: str, with_clean_attack: bool = True
) -> pd.DataFrame:
    columns = ["attack"]
    index = ["dataset", "target_model_training", "target_model_arch"]

    # Filter
    df = df[df["scenario"] == "AB"]
    df = df[df["is_constrained"]]
    if not with_clean_attack:
        df = df[df["attack"] != "no_attack"]
    df = df[df["source_model_training"].isin(["default", "madry"])]

    # Sort

    print(df.columns)

    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    print(df.columns)
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
            bold_min=True,
            bold_max=False,
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
            custom_format="{:.1f}",
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


def table_AB_ablation(
    df: pd.DataFrame, metric: str, with_clean_attack: bool = True
) -> pd.DataFrame:
    columns = ["attack"]
    index = ["dataset", "target_model_arch"]

    # Filter
    df = df[df["scenario"].isin(["AB", "ablation"])]
    df = df[df["is_constrained"]]
    if not with_clean_attack:
        df = df[df["attack"] != "no_attack"]
    df = df[df["source_model_training"].isin(["default", "madry"])]

    # Sort

    print(df.columns)

    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    print(df.columns)
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
            bold_min=True,
            bold_max=False,
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
            custom_format="{:.1f}",
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


def table_budget(
    df: pd.DataFrame, metric: str, with_clean_attack: bool = True
) -> pd.DataFrame:
    columns = ["attack", "budget"]
    index = ["dataset", "target_model_arch"]

    # Filter

    df = df[df["scenario"].isin(["AB", "budget"])]
    df = df[(df["source_model_training"] == "default")]
    df = df[df["is_constrained"]]
    if not with_clean_attack:
        df = df[df["attack"] != "no_attack"]
    df = df[df["source_model_training"].isin(["default"])]

    df["budget"] = 1
    budget_two = ((df["n_iter"] == 100) | (df["n_iter"] == 10)) & (
        df["scenario"] == "budget"
    )
    df.loc[budget_two, "budget"] = 2
    budget_three = ((df["n_iter"] == 200) | (df["n_iter"] == 20)) & (
        df["scenario"] == "budget"
    )
    df.loc[budget_three, "budget"] = 3
    df = df[df["attack"].isin(["caa4", "apgd2", "moeva", "no_attack"])]

    df["budget_order"] = df["budget"].copy()

    print(df[["budget", "attack"]].value_counts())

    # Sort

    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate

    df = (
        df.groupby(
            GROUP_BY + ["budget"],
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
            bold_min=True,
            bold_max=False,
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
            custom_format="{:.1f}",
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


def table_rank_source(
    df: pd.DataFrame, metric: str, with_clean_attack: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["scenario"]
    index = ["target_model_training", "source_model_arch"]

    # Filter
    df = df[df["scenario"].isin(["C", "D", "E"])]
    df = df[df["is_constrained"]]
    if not with_clean_attack:
        df = df[df["attack"] != "no_attack"]
    df = df[
        (
            df["source_model_training"].isin(["default"])
            & (df["scenario"] == "C")
        )
        | (
            df["source_model_training"].isin(["subset"])
            & (df["scenario"] == "D")
        )
        | (
            df["source_model_training"].isin(["dist"])
            & (df["scenario"] == "E")
        )
    ]
    df = df[df["target_model_training"].isin(["default", "madry"])]
    df = df[~(df["target_model_arch"] == df["source_model_arch"])]

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate

    df = df.groupby(GROUP_BY, sort=False)[metric].mean().reset_index()

    df["rank"] = df.groupby(
        ["scenario", "target_model_training", "target_model_arch"], sort=False
    )["robust_acc"].rank()

    df = (
        df.groupby(
            ["scenario", "target_model_training", "source_model_arch"],
            sort=False,
        )["rank"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    # Beautify
    df = auto_beautify_values(df)
    df["latex"] = "$" + df["mean"].map("{:.2f}".format) + "$"

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
    return pivot, df


def table_rank_target(
    df: pd.DataFrame, metric: str, with_clean_attack: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["scenario"]
    index = ["target_model_training", "target_model_arch"]

    # Filter
    df = df[df["scenario"].isin(["C", "D", "E"])]
    df = df[df["is_constrained"]]
    if not with_clean_attack:
        df = df[df["attack"] != "no_attack"]
    df = df[
        (
            df["source_model_training"].isin(["default"])
            & (df["scenario"] == "C")
        )
        | (
            df["source_model_training"].isin(["subset"])
            & (df["scenario"] == "D")
        )
        | (
            df["source_model_training"].isin(["dist"])
            & (df["scenario"] == "E")
        )
    ]
    df = df[~(df["target_model_arch"] == df["source_model_arch"])]
    df = df[df["target_model_training"].isin(["default", "madry"])]

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate

    df = df.groupby(GROUP_BY, sort=False)[metric].mean().reset_index()

    df["rank"] = df.groupby(
        ["scenario", "target_model_training", "source_model_arch"], sort=False
    )["robust_acc"].rank(method="min")

    df = (
        df.groupby(
            ["scenario", "target_model_training", "target_model_arch"],
            sort=False,
        )["rank"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    # Beautify
    df = auto_beautify_values(df)
    df["latex"] = "$" + df["mean"].map("{:.2f}".format) + "$"

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
    return pivot, df


def table_rank(df: pd.DataFrame) -> pd.DataFrame:
    df_source = table_rank_source(df.copy(), "robust_acc")[1]
    df_source["mode"] = "As a source"
    df_source["model_arch"] = df_source["source_model_arch"]

    df_target = table_rank_target(df.copy(), "robust_acc")[1]
    df_target["mode"] = "As a target"
    df_target["model_arch"] = df_target["target_model_arch"]

    columns = ["mode", "scenario"]
    index = ["target_model_training", "model_arch"]

    df = pd.concat([df_source, df_target], axis=0)

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


def save_table(
    df: pd.DataFrame,
    name: str,
    caption: str = "CAPTION",
    other_save: Optional[List[str]] = None,
) -> None:
    path = f"./{OUT_ROOT}/{name}_table.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    latex_col = "latex"
    latex_cols: Union[str, List[str]] = []
    if other_save is not None:
        latex_cols = [latex_col] + other_save
    else:
        latex_cols = latex_col

    df.drop(columns=latex_cols, inplace=False).to_csv(
        f"{path}_table.csv", index=True
    )
    df = df[latex_cols]

    if other_save is not None:
        for i in range(len(df.columns.names) - 1):
            df = df.swaplevel(i, i + 1, axis=1)

        def sort_custom(x: str) -> str:
            local_sort = {
                "latex": "0",
                "latex_min": "1",
                "latex_max": "2",
            }
            return local_sort.get(x, x)

        df = df.sort_index(
            axis=1, level=[0, 1], key=lambda x: x.map(sort_custom)
        )
        # df = df.sort_index(axis=1, level=0)

    style = df.style

    path = f"{path}_table.tex"

    style.to_latex(
        path,
        column_format="l" * len(df.index.names) + "|" + "l" * len(df.columns),
        multicol_align="c",
        caption=caption,
        clines="skip-last;data",
        hrules=True,
        position_float="centering",
        convert_css=True,
    )

    beautify_latex(path)


def map_scenario(scenario: str) -> str:
    if scenario == "AB":
        return "A"
    else:
        return scenario


def plot_acde_one(
    df: pd.DataFrame,
    is_constrained: bool,
    name: str,
    df_min: float,
    df_max: float,
) -> None:
    # df_delta = df_max - df_min

    # BEAUTIFY
    df = auto_beautify_values(df)
    df = df.copy()

    df["scenario"] = df["scenario"].map(map_scenario) + (
        "1" if is_constrained else "2"
    )
    df["source_model_arch_bk"] = df["source_model_arch"]
    df = df.drop(columns="source_model_arch")
    df = df.rename(columns=beautify_col_name)

    barplot(
        df,
        name,
        x="Scenario",
        y="robust_acc",
        hue="Model",
        x_label="Scenario",
        y_label="Accuracy",
        # fig_size=(24, 16),
        legend_pos="outside",
        # x_lim=None,
        # y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
        y_lim=(0, 1),
        rotate_ticks=0,
        error_min_max=True,
        fig_size=(6, 2),
    )
    barplot(
        df,
        name + "no_legend",
        x="Scenario",
        y="robust_acc",
        hue="Model",
        x_label="Scenario",
        y_label="Accuracy",
        # fig_size=(24, 16),
        legend_pos=False,
        # x_lim=None,
        # y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
        y_lim=(0, 1),
        rotate_ticks=0,
        error_min_max=True,
        fig_size=(6, 2),
    )


def plot_acde(df: pd.DataFrame, dataset: str) -> None:
    dimensions = ["scenario", "target_model_arch"]

    df_min = df["robust_acc"].min()
    df_max = df["robust_acc"].max()

    # FILTER
    df = df[df["scenario"].isin(["AB", "C", "D", "E"])]
    df = df[df["attack"] == "caa3"]
    df = df[
        ~(
            df["source_model_training"].isin(["madry"])
            & (df["scenario"] == "C")
        )
    ]

    # Aggregate
    # df = (
    #     df.groupby(
    #         GROUP_BY,
    #         sort=False,
    #     )
    #     .mean()
    #     .reset_index()
    # )
    # print(df.shape)

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in dimensions])

    for is_constrained in df["is_constrained"].unique():
        for target_training in ["default", "madry"]:
            df_l = df[
                (df["target_model_training"] == target_training)
                & (df["is_constrained"] == is_constrained)
            ]
            plot_acde_one(
                df_l.copy(),
                is_constrained,
                f"scenario_{dataset}_{target_training}_{is_constrained}",
                df_min,
                df_max,
            )


def table_moeva_budget(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    columns = ["source_model_arch"]
    index = ["dataset", "target_model_training", "budget"]

    # Augment

    df["budget"] = (
        df["n_iter"].astype(str) + "x" + df["n_offsprings"].astype(str)
    )
    df["budget_order"] = 1000 * df["n_iter"] + df["n_offsprings"]

    # Filter
    df = df[df["scenario"].isin(["AB", "B_STEPS"])]
    df = df[df["attack"] == "moeva"]
    df = df[df["is_constrained"]]

    # Sort
    df = df.sort_values(by=[f"{e}_order" for e in index + columns])

    # Aggregate

    df = (
        df.groupby(
            GROUP_BY + ["budget"],
            sort=False,
        )[metric]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )

    # Beautify
    df = auto_beautify_values(df)
    bold = df.groupby(
        ["dataset", "target_model_training", "source_model_arch"]
    )["mean"].transform(lambda x: x == x.min())
    to_latex(
        df,
        bold=bold,
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


def table_eps(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    columns = ["source_model_arch"]
    index = ["dataset", "target_model_training", "is_constrained", "eps"]

    # Filter
    df = df[df["scenario"].isin(["AB", "AB_EPS"])]
    # df = df[df["attack"] == "moeva"]
    # df = df[df["is_constrained"] == True]

    # Sort
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
    bold = df.groupby(
        [
            "dataset",
            "target_model_training",
            "is_constrained",
            "source_model_arch",
        ]
    )["mean"].transform(lambda x: x == x.min())
    to_latex(
        df,
        bold=bold,
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


def table_A_budget(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    columns = ["source_model_arch"]
    index = ["dataset", "target_model_training", "is_constrained", "n_iter"]

    # Filter
    df = df[df["scenario"].isin(["AB", "A_STEPS"])]
    # df = df[df["attack"] == "moeva"]
    # df = df[df["is_constrained"] == True]

    # Sort
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
    bold = df.groupby(
        [
            "dataset",
            "target_model_training",
            "is_constrained",
            "source_model_arch",
        ]
    )["mean"].transform(lambda x: x == x.min())
    to_latex(
        df,
        bold=bold,
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


def table_acde(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    index = ["dataset", "target_model_training", "scenario_constrained"]
    columns = ["target_model_arch"]

    # Add

    df["scenario_constrained"] = df["scenario"].map(map_scenario) + (
        df["is_constrained"].map({True: "1", False: "2"})
    )
    df["scenario_constrained_order"] = df["scenario_constrained"]

    # Filter

    df = df[df["scenario"].isin(["AB", "C", "D", "E"])]
    df = df[df["attack"] == "caa3"]
    df = df[
        ~(
            df["scenario"].isin(["C"])
            & (df["source_model_training"] == "madry")
        )
    ]

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
    df["bold"] = False
    to_latex(
        df,
        bold=df["bold"],
    )
    to_latex(
        df,
        bold=df["bold"],
        in_col="min",
        out_col="latex_min",
        auto_sem=False,
    )
    to_latex(
        df,
        bold=df["bold"],
        in_col="max",
        out_col="latex_max",
        auto_sem=False,
    )

    # Pivot

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=[
            "mean",
            "sem",
            "latex",
            "latex_min",
            "latex_max",
            # "min",
            # "max",
        ],
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


def filter_test(df: pd.DataFrame) -> pd.DataFrame:
    # df = df[
    #     df["source_model_arch"].isin(["tabtransformer", "torchrln", "vime"])
    # ]
    # df = df[
    #     df["target_model_arch"].isin(["tabtransformer", "torchrln", "vime"])
    # ]
    # df = df[df["dataset"]!= "ctu_13_neris"]

    return df


def transfer_bold(x):
    # print(x)
    # x["local_min"] = False
    # x.loc[x["target_model_arch"] != [], "local_min"] = x[
    # return x == x.min()
    pass


def minus_clean(x, ref):
    filter_values = ref["dataset"] == x["dataset"]
    filter_values &= ref["source_model_arch"] == x["target_model_arch"]
    filter_values &= ref["source_model_training"] == x["target_model_training"]

    value = ref[filter_values]["robust_acc"].values[0]

    # print(value)
    # print(x["robust_acc"])
    # print( x["robust_acc"] - value)
    return x["robust_acc"] - value


def transferability(df: pd.DataFrame) -> None:
    metric = "robust_acc"
    # print(df[df["attack"]=="no_attack"].shape)

    df_clean_perfs = (
        df[df["attack"] == "no_attack"]
        .groupby(["dataset", "source_model_training", "source_model_arch"])[
            "robust_acc"
        ]
        .first()
        .reset_index()
    )

    # print(df_clean.shape)

    custom_filter = df["source_model_training"] == "dist"
    custom_filter &= df["target_model_training"] == "default"
    custom_filter &= df["scenario"].isin(["E"])
    custom_filter &= df["is_constrained"]
    custom_filter &= df["attack"] == "caa4"
    print((df["source_model_training"] == "subset").sum())

    # exit(0)
    df = df[custom_filter]

    df["robust_acc"] = df.apply(
        lambda x: minus_clean(x, df_clean_perfs), axis=1
    )

    index = ["dataset", "target_model_arch"]
    columns = ["source_model_arch"]

    print(df[df["attack"] == "no_attack"])

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

    return pivot


def table_defense_do(
    df: pd.DataFrame, metric: str, with_clean_attack: bool = True
) -> pd.DataFrame:
    columns = ["attack"]
    index = ["dataset", "target_model_training", "target_model_arch"]

    # Filter
    df = df[df["scenario"] == "AB"]
    df = df[df["is_constrained"]]
    if not with_clean_attack:
        df = df[df["attack"] != "no_attack"]
    df = df[df["source_model_training"].isin(["default", "madry"])]

    # Sort

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
            bold_min=True,
            bold_max=False,
        )

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


def to_latex_min_max(
    df: pd.DataFrame,
    index: List[str],
    bold_min: bool = False,
    bold_max: bool = False,
) -> None:
    df["is_max"] = df.groupby(index)["mean"].transform(lambda x: x == x.max())
    df["is_min"] = df.groupby(index)["mean"].transform(lambda x: x == x.min())

    df["bold"] = False
    bold = df["bold"].copy()
    if bold_min:
        bold = bold | df["is_min"]
    if bold_max:
        bold = bold | df["is_max"]

    to_latex(df, bold)


def to_latex2(
    df: pd.DataFrame,
    bold: pd.Series,
    index: List[str],
    bold_min: bool = False,
    bold_max: bool = False,
    in_col: str = "mean",
    out_col: str = "latex",
    custom_format: Union[str, Callable] = percent_format,
) -> None:

    df["is_max"] = df.groupby(index)["mean"].transform(lambda x: x == x.max())
    df["is_min"] = df.groupby(index)["mean"].transform(lambda x: x == x.min())

    df["bold"] = False
    bold = df["bold"].copy()
    if bold_min:
        bold = bold | df["is_min"]
    if bold_max:
        bold = bold | df["is_max"]

    if isinstance(custom_format, str):
        custom_format = custom_format.format

    # df["bold"] = False
    # bold = df["bold"].copy()
    # df.drop(columns="bold", inplace=True, errors="ignore")

    df.loc[bold, out_col] = (
        "$\\mathbf{"
        + df.loc[bold, in_col].map(custom_format)
        + "}$"
        + "\\tiny{$\\mathbf{\\pm "
        + (1.96 * df.loc[bold, "diff"]).map(custom_format)
        + "}$}"
    )

    df.loc[~bold, out_col] = (
        "$"
        + df.loc[~bold, in_col].map(custom_format)
        + "$"
        + "\\tiny{$\\pm "
        + (1.96 * df.loc[~bold, "diff"]).map(custom_format)
        + "$}"
    )


def run() -> None:
    df = load_data(DATA_PATH)
    df = filter_test(df)

    table_effective_efficient = df[
        df["attack"].isin(["no_attack", "apgd2", "ucs", "moeva", "caa4"])
        & (df["source_model_training"] == "default")
    ].copy()
    robust_acc = table_AB(
        table_effective_efficient.copy(), "robust_acc", with_clean_attack=True
    )

    budget = table_budget(df.copy(), "robust_acc")
    save_table(
        budget.copy(),
        "main_table_ab_budget",
    )

    robust_acc = table_AB(
        table_effective_efficient.copy(), "robust_acc", with_clean_attack=True
    )

    table_ablation = df[
        df["attack"].isin(
            [
                "no_attack",
                "apgd2",
                "apgd2-nrep",
                "apgd2-nini",
                "apgd2-nran",
                "apgd2-nbes",
                "apgd2-nada",
            ]
        )
        & (df["source_model_training"] == "default")
    ].copy()
    print(table_ablation.shape)

    table_ablation = table_AB_ablation(
        table_ablation.copy(), "robust_acc", with_clean_attack=True
    )
    save_table(
        table_ablation.copy(),
        "capgd_ablation",
    )

    exit(0)

    robust_acc_copy = robust_acc.copy()

    duration = table_AB(
        table_effective_efficient.copy(),
        "attack_duration",
        with_clean_attack=False,
    )
    duration_copy = duration.copy()

    rob_and_dur = pd.concat([robust_acc, duration], axis=1)

    rob_and_dur.columns = pd.MultiIndex.from_tuples(
        [(col[0], "Robust Accuracy", col[1]) for col in robust_acc.columns]
        + [(col[0], "Duration", col[1]) for col in duration.columns]
    )
    rob_and_dur = rob_and_dur.reset_index(level=1, drop=True)
    save_table(
        rob_and_dur.copy(),
        "main_table_ab_robust_duration",
    )

    save_table(
        robust_acc.copy(),
        "main_table_ab_robust_accuracy",
    )
    save_table(
        duration.copy(),
        "main_table_ab_attack_duration",
    )

    table_defense = df[
        df["attack"].isin(["no_attack", "moeva", "apgd2", "caa4"])
        & (df["target_model_training"] == "madry")
    ].copy()

    table_defense = table_AB(
        table_defense.copy(), "robust_acc", with_clean_attack=True
    )
    print(table_defense.shape)
    print(table_defense)
    # table_defense["latex_"]

    # print(robust_acc[table_defense.columns].shape)
    table_defense = table_defense.reset_index(level=1, drop=True)
    # print(table_defense["mean"])
    robust_acc_copy = robust_acc_copy.reset_index(level=1, drop=True)
    print(robust_acc_copy.columns)
    robust_acc_copy = robust_acc_copy.drop(
        columns=[("mean", "BF"), ("latex", "BF")]
    )
    # print(robust_acc_copy["mean"])

    # print(robust_acc_copy["mean"] - table_defense["mean"])

    table_defense["latex"] += (
        table_defense["mean"] - robust_acc_copy["mean"]
    ).applymap(percent_format_diff)
    print(table_defense["latex"])

    # table_defense["latex"] = table_defense["latex"] + " (" + table_defense["diff"].map(percent_format) + ")"

    save_table(
        table_defense.copy(),
        "madry_training",
    )
    # exit(0)

    abblattion = df[
        df["attack"].isin(
            [
                "no_attack",
                "pgdl2org",
                "pgdl2rsae",
                "pgdl2nrsnae",
                "pgdl2",
                "apgd2",
            ]
        )
        & (df["source_model_training"] == "default")
    ].copy()

    abblattion = table_AB(
        abblattion.copy(), "robust_acc", with_clean_attack=True
    )

    save_table(
        abblattion.copy(),
        "abblattion",
    )

    table_gradient = df[
        df["attack"].isin(
            ["no_attack", "pgdl2ijcai", "lowprofool", "ucs", "apgd2"]
        )
        & (df["source_model_training"] == "default")
    ].copy()
    # print(df["attack"].unique())
    robust_acc = table_AB(
        table_gradient.copy(), "robust_acc", with_clean_attack=True
    )
    robust_acc = robust_acc.reset_index(level=1, drop=True)

    save_table(
        robust_acc.copy(),
        "sota_robustness",
    )

    table_gradient = df[
        df["attack"].isin(["no_attack", "lowprofool", "pgdl2ijcai", "apgd2"])
        & (df["source_model_training"] == "default")
    ].copy()
    # print(df["attack"].unique())
    robust_acc = table_AB(
        table_gradient.copy(), "robust_acc", with_clean_attack=True
    )
    robust_acc = robust_acc.reset_index(level=1, drop=True)

    save_table(
        robust_acc.copy(),
        "gradient_robustness",
    )

    table_search = df[
        df["attack"].isin(["no_attack", "ucs", "moeva", "apgd2"])
        & (df["source_model_training"] == "default")
    ].copy()
    # print(df["attack"].unique())
    robust_acc = table_AB(
        table_search.copy(), "robust_acc", with_clean_attack=True
    )
    robust_acc = robust_acc.reset_index(level=1, drop=True)

    save_table(
        robust_acc.copy(),
        "search_robustness",
    )

    transfer_table = transferability(df.copy())
    save_table(
        transfer_table,
        "transferability",
    )

    # moeva_table = table_moeva_budget(df.copy(), "robust_acc")
    # save_table(
    #     moeva_table.copy(),
    #     "moeva_table_robust_accuracy",
    # )
    # for ds in moeva_table.index.get_level_values(0).unique():
    #     moeva_table_ds = moeva_table.loc[ds]
    #     save_table(
    #         moeva_table_ds, f"{ds}/moeva_table_robust_accuracy", caption=ds
    #     )

    # acde_table = table_acde(df.copy(), "robust_acc")
    # save_table(
    #     acde_table,
    #     "acde_table_robust_accuracy",
    #     other_save=["latex_min", "latex_max"],
    # )

    # for ds in acde_table.index.get_level_values(0).unique():
    #     acde_table_ds = acde_table.loc[ds]
    #     save_table(
    #         acde_table_ds,
    #         f"acde_table_{ds}_robust_accuracy",
    #         other_save=["latex_min", "latex_max"],
    #         caption=ds,
    #     )

    # for attack in ["pgdl2", "apgd", "moeva", "caa3"]:
    #     df_l = df[df["attack"] == attack]
    #     save_table(
    #         table_eps(df_l.copy(), "robust_acc"),
    #         f"eps_{attack}_table_robust_accuracy",
    #     )

    # for attack in ["pgdl2", "apgd", "caa3"]:
    #     df_l = df[df["attack"] == attack]
    #     save_table(
    #         table_A_budget(df_l.copy(), "robust_acc"),
    #         f"A_budget_{attack}_table_robust_accuracy",
    #     )

    # for dataset in df["dataset"].unique():
    #     df_l = df[df["dataset"] == dataset]
    #     save_table(
    #         table_rank(df_l.copy()),
    #         f"{dataset}/table_rank_both",
    #     )
    #     plot_acde(df_l.copy(), dataset)


if __name__ == "__main__":
    run()
