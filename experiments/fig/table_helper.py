from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd

from experiments.fig.beautify_latex import beautify_latex
from experiments.fig.format import percent_format


def to_latex(
    df: pd.DataFrame,
    bold: pd.Series,
    in_col: str = "mean",
    out_col: str = "latex",
    auto_sem: bool = True,
    custom_format: Union[str, Callable] = percent_format,
) -> None:
    if auto_sem:
        no_sem = df["attack"].isin(["no_attack", "Clean"])
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


def to_latex_min_max(
    df: pd.DataFrame,
    index: List[str],
    bold_min: bool = False,
    bold_max: bool = False,
    float_round: int = 0,
    auto_sem: bool = True,
) -> None:

    df_in = df
    df = df.copy()
    df["mean_tmp"] = df["mean"].copy()
    df["ci"] = 1.96 * df["sem"]
    if float_round > 0:
        df["mean_tmp"] = df["mean_tmp"].round(float_round)
        df["ci"] = df["ci"].round(float_round)

    # replace nan ci with 0
    df["ci"] = df["ci"].fillna(0)

    df["ci_down"] = df["mean_tmp"] - df["ci"]
    df["ci_up"] = df["mean_tmp"] + df["ci"]

    df["ci_down_max"] = df.groupby(index)["ci_down"].transform(
        lambda x: x.max()
    )
    df["ci_up_min"] = df.groupby(index)["ci_up"].transform(lambda x: x.min())

    df["is_min"] = df["ci_down"] <= df["ci_up_min"]
    df["is_max"] = df["ci_up"] >= df["ci_down_max"]

    filter_test = df["dataset"] == "CTU"
    filter_test = filter_test & (df["source_model_arch"] == "VIME")

    df_test = df[filter_test]

    df["bold"] = False
    bold = df["bold"].copy()
    if bold_min:
        bold = bold | df["is_min"]
    if bold_max:
        bold = bold | df["is_max"]

    to_latex(df_in, bold, auto_sem=auto_sem)


def save_table(
    df: pd.DataFrame,
    path: str,
    caption: str = "CAPTION",
    other_save: Optional[List[str]] = None,
) -> None:
    df = df.copy()
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
        column_format="l" * len(df.index.names) + "r" * len(df.columns),
        # column_format="l" * len(df.index.names) + "|" + "l" * len(df.columns),
        multicol_align="c",
        caption=caption,
        clines="skip-last;data",
        hrules=True,
        position_float="centering",
        convert_css=True,
    )

    beautify_latex(path)
