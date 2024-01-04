import pandas as pd
from .beautify_data import data_order, column_names, beautify_col_name

DATA_PATH = "./data_tmp.csv"

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

    # Sort

    df = df.sort_values(by=[f"{e}_order" for e in index + columns])
    # return df

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


def save_table(df: pd.DataFrame, name: str, caption: str = "CAPTION") -> None:
    df.drop(columns="latex", inplace=False).to_csv(
        f"./table_{name}.csv", index=True
    )
    df = df["latex"]
    df.to_latex(
        f"./table_{name}.tex",
        float_format="%.3f",
        column_format="l" * len(df.index) + "|" + "l" * len(df.columns),
        escape=False,
        multicolumn_format="c",
        multicolumn=True,
        multirow=True,
        caption=caption,
    )


def run():
    df = load_data(DATA_PATH)
    save_table(
        table_AB(df.copy(), "robust_acc", with_clean_attack=True),
        "main_table_ab_robust_accuracy",
    )
    save_table(
        table_AB(df.copy(), "attack_duration", with_clean_attack=False),
        "main_table_ab_attack_duration",
    )


if __name__ == "__main__":
    run()
