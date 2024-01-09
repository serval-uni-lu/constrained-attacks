from pathlib import Path
import pandas as pd
from .beautify_data import data_order, column_names, beautify_col_name
from constrained_attacks.graphics import (
    DPI,
    FONT_SCALE,
    _color_palette,
    _get_filename,
    _setup_legend,
    barplot,
    lineplot,
)

DATA_PATH = "./data_tmp.csv"
OUT_ROOT = "./data/fig/new/"

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
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def auto_beautify_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in data_order:
            df[col] = df[col].map(data_order[col])
    return df


def table_AB(df: pd.DataFrame, metric, with_clean_attack=True) -> None:
    columns = ["attack"]
    index = ["dataset", "target_model_training", "target_model_arch"]

    # Filter
    df = df[df["scenario"] == "AB"]
    df = df[df["is_constrained"] == True]
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
    df["latex"] = (
        "$"
        + df["mean"].map("{:.3f}".format)
        + "$"
        + "\\tiny{$\\pm "
        + (1.96 * df["sem"]).map("{:.3f}".format)
        + "$}"
    )
    df.loc[df["attack"] == "Clean", "latex"] = (
        "$"
        + df.loc[df["attack"] == "Clean", "mean"].map("{:.3f}".format)
        + "$"
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
    df: pd.DataFrame, metric, with_clean_attack=True
) -> None:
    columns = ["scenario"]
    index = ["target_model_training", "source_model_arch"]

    # Filter
    df = df[df["scenario"].isin(["C", "D", "E"])]
    df = df[df["is_constrained"] == True]
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
    df: pd.DataFrame, metric, with_clean_attack=True
) -> None:
    columns = ["scenario"]
    index = ["target_model_training", "target_model_arch"]

    # Filter
    df = df[df["scenario"].isin(["C", "D", "E"])]
    df = df[df["is_constrained"] == True]
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
    )["robust_acc"].rank()

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


def save_table(df: pd.DataFrame, name: str, caption: str = "CAPTION") -> None:
    path = f"./{OUT_ROOT}/{name}_table.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df.drop(columns="latex", inplace=False).to_csv(
        f"{path}_table.csv", index=True
    )
    df = df["latex"]
    df.to_latex(
        f"{path}_table.tex",
        float_format="%.3f",
        column_format="l" * len(df.index) + "|" + "l" * len(df.columns),
        escape=False,
        multicolumn_format="c",
        multicolumn=True,
        multirow=True,
        caption=caption,
    )


# def plot_acde_one(df: pd.DataFrame) -> None:


#     lineplot(
#         df[df["scenario_name"].str.contains(c)],
#         name_l,
#         x="Scenario",
#         y="robust_acc",
#         hue="Model",
#         # style="Model",
#         x_label="Scenario",
#         y_label="Accuracy",
#         y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
#     )


#     df_all = df_all[df_all["attack_name"] == "CAA"]
#     df_all = df_all.copy()

#     df_new = []
#     for scenario_name in ["A", "C", "D", "E"]:
#         df = df_all[df_all["scenario_name"].str.contains(scenario_name)]
#         df = process_scenario2(df, scenario_name)
#         df_new.append(df)
#     df_new = pd.concat(df_new)

#     df_new = df_new[
#         (df_new["Model Source"] != "Robust")
#         | (~df_all["scenario_name"].str.contains("C"))
#     ]

#     for target in df_new["Model Target"].unique():
#         df = df_new[df_new["Model Target"] == target]

#         # df_augment = df[df["scenario_name"] == "A1"].copy()
#         # df_augment["Scenario"] = "Standard"
#         # df_augment["robust_acc"] = df_augment["clean_acc"]
#         # df = pd.concat([df_augment, df])
#         df_min = df["robust_acc"].min()
#         df_max = df["robust_acc"].max()
#         df_delta = df_max - df_min
#         print(f"-------------------- {df_min} {df_max}")
#         for c in ["1", "2"]:
#             name_l = f"{name}_{target}_{c}"
#             df["Model"] = df["model_name_target"].map(model_names)
#             lineplot(
#                 df[df["scenario_name"].str.contains(c)],
#                 name_l,
#                 x="Scenario",
#                 y="robust_acc",
#                 hue="Model",
#                 # style="Model",
#                 x_label="Scenario",
#                 y_label="Accuracy",
#                 y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
#             )


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
    # df_min = df["robust_acc"].min()
    # df_max = df["robust_acc"].max()
    df_delta = df_max - df_min

    # BEAUTIFY
    df = auto_beautify_values(df)
    df = df.copy()

    df["scenario"] = df["scenario"].map(map_scenario) + (
        "1" if is_constrained else "2"
    )
    df["source_model_arch_bk"] = df["source_model_arch"]
    df = df.drop(columns="source_model_arch")
    df = df.rename(columns=beautify_col_name)

    # lineplot(
    #     df,
    #     name,
    #     x="Scenario",
    #     y="robust_acc",
    #     hue="Model",
    #     # style="Model",
    #     x_label="Scenario",
    #     y_label="Accuracy",
    #     y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
    #     error_min_max=True,
    # )

    barplot(
        df,
        name,
        x="Scenario",
        y="robust_acc",
        hue="Model",
        x_label="Scenario",
        y_label="Accuracy",
        # fig_size=(24, 16),
        legend_pos="best",
        # x_lim=None,
        y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
        rotate_ticks=0,
        error_min_max=True,
    )
    barplot(
        df,
        name + "test",
        x="Scenario",
        y="robust_acc",
        hue="Model",
        x_label="Scenario",
        y_label="Accuracy",
        # fig_size=(24, 16),
        legend_pos="best",
        # x_lim=None,
        y_lim=(df_min - df_delta * 0.05, df_max + df_delta * 0.05),
        rotate_ticks=0,
        error_min_max=False,
    )


def plot_acde(df: pd.DataFrame, dataset: str) -> None:
    print(df.shape)
    dimensions = ["scenario", "target_model_arch"]

    df_min = df["robust_acc"].min()
    df_max = df["robust_acc"].max()

    # FILTER

    df = df[df["attack"] == "caa3"]
    # df = df[df["is_constrained"] == True]
    df = df[
        (~df["source_model_training"].isin(["madry"]))
        | (df["scenario"] == "AB")
    ]
    print(df.shape)

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
            print(df_l.shape)
            plot_acde_one(
                df_l.copy(),
                is_constrained,
                f"{dataset}/{target_training}_{is_constrained}",
                df_min,
                df_max,
            )


def filter_test(df: pd.DataFrame) -> pd.DataFrame:
    # df = df[
    #     df["source_model_arch"].isin(["tabtransformer", "torchrln", "vime"])
    # ]
    # df = df[
    #     df["target_model_arch"].isin(["tabtransformer", "torchrln", "vime"])
    # ]
    return df


def run():
    df = load_data(DATA_PATH)
    df = filter_test(df)
    save_table(
        table_AB(df.copy(), "robust_acc", with_clean_attack=True),
        "main_table_ab_robust_accuracy",
    )
    save_table(
        table_AB(df.copy(), "attack_duration", with_clean_attack=False),
        "main_table_ab_attack_duration",
    )

    for dataset in df["dataset"].unique():
        df_l = df[df["dataset"] == dataset]
        save_table(
            table_rank(df_l.copy()),
            f"{dataset}/table_rank_both",
        )
        plot_acde(df_l.copy(), dataset)


if __name__ == "__main__":
    run()
