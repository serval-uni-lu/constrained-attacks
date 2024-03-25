import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constrained_attacks.graphics import (
    DPI,
    FONT_SCALE,
    _color_palette,
    _get_filename,
    _setup_legend,
    barplot,
)

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


if __name__ == "__main__":
    run()
