from math import log2, log10
from pathlib import Path

import matplotlib
import pandas as pd
import pylab as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles, venn3

from .beautify_data import beautify_col_name, data_order

# IN_PATH = "intersection_search.csv"
# IN_PATH3 = "intersection3_search.csv"
# OUT_ROOT = "./data/fig/neurips-20240518_search/"

# IN_PATH = "intersection_gradient.csv"
# IN_PATH3 = "intersection3_gradient.csv"
# OUT_ROOT = "./data/fig/neurips-20240518_gradient/"

IN_PATH = "intersection_cagpd3_ablation.csv"
IN_PATH3 = "intersection3_cagpd3_ablation.csv"
OUT_ROOT = "./data/fig/neurips-20240518_capgd3_ablation/"

my_log = lambda x: log10(x) if x > 0 else 0


def plot_venn(df, path, title=None):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df = df.groupby(["attack_1_name", "attack_2_name"]).sum().reset_index()

    for index, row in df.iterrows():
        intersection = row["intersection"]
        attack_1_name = row["attack_1_name"]
        attack_2_name = row["attack_2_name"]
        data = (
            row["attack_1_count"] - intersection,
            row["attack_2_count"] - intersection,
            intersection,
        )
        v = venn2(
            subsets=data,
            set_labels=(row["attack_1_name"], row["attack_2_name"]),
        )
        plt.title(title)
        plt.savefig(f"{path}_{attack_1_name}_{attack_2_name}.pdf")
        plt.clf()


def three_way_wenn(df, attacks):

    df = df.copy()

    df = (
        df.groupby(["attack_1_name", "attack_2_name", "attack_3_name"])
        .sum()
        .reset_index()
    )
    df = df[
        df["attack_1_name"].isin(attacks)
        & df["attack_2_name"].isin(attacks)
        & df["attack_3_name"].isin(attacks)
    ]

    for index, row in df.iterrows():

        if row["attack_1_name"] == row["attack_2_name"]:
            continue
        if row["attack_1_name"] == row["attack_3_name"]:
            continue
        if row["attack_2_name"] == row["attack_3_name"]:
            continue

        a, b, c, ab, ac, bc, abc = (
            row["a"],
            row["b"],
            row["c"],
            row["ab"],
            row["ac"],
            row["bc"],
            row["abc"],
        )

        Abc = a - ab - ac + abc
        aBc = b - ab - bc + abc
        abC = c - ac - bc + abc
        ABc = ab - abc
        AbC = ac - abc
        aBC = bc - abc

        data_label = {
            "100": Abc,
            "010": aBc,
            "110": ABc,
            "001": abC,
            "101": AbC,
            "011": aBC,
            "111": abc,
        }

        # Abc = my_log(Abc) if Abc > 0 else 0
        # aBc = my_log(aBc) if aBc > 0 else 0
        # abC = my_log(abC) if abC > 0 else 0
        # ABc = my_log(ABc) if ABc > 0 else 0
        # AbC = my_log(AbC) if AbC > 0 else 0
        # aBC = my_log(aBC) if aBC > 0 else 0
        # abc = my_log(abc) if abc > 0 else 0

        data = {
            "100": Abc,
            "010": aBc,
            "110": ABc,
            "001": abC,
            "101": AbC,
            "011": aBC,
            "111": abc,
        }
        print(data)

        # data = tuple(data.values())
        # print(row["attack_1_name"], row["attack_2_name"], row["attack_3_name"])
        # print(data)

        for i in data.values():
            if i < 0:
                print("NEGATIVE")
                print(data)
        v = venn3(
            subsets=data,
            set_labels=(
                row["attack_1_name"]
                if row["attack_1_name"] != "BF"
                else "BF*",
                row["attack_2_name"]
                if row["attack_2_name"] != "BF"
                else "BF*",
                row["attack_3_name"]
                if row["attack_3_name"] != "BF"
                else "BF*",
            ),
            # normalize_to=5.0,
        )
        # v.get_patch_by_id('100').set_color('white')
        # v.get_label_by_id('101').set_text("red")

        # for key, value in data_label.items():
        #     if value > 0:
        #         v.get_label_by_id(key).set_text(f"{value}")
        #     else:
        #         v.get_label_by_id(key).set_text("")

        # plt.title(f"{row['dataset']} {row['model']}")
        plt.savefig(
            f"{OUT_ROOT}/venn3_{row['attack_1_name']}_{row['attack_2_name']}_{row['attack_3_name']}.pdf"
        )
        plt.clf()
    print(
        f"{OUT_ROOT}/venn3_{row['attack_1_name']}_{row['attack_2_name']}_{row['attack_3_name']}.pdf"
    )
    # print(df)

    # for index, row in df.iterrows():
    #     intersection = row["intersection"]
    #     data = (
    #         row[attacks[0]] - intersection,
    #         row[attacks[1]] - intersection,
    #         row[attacks[2]] - intersection,
    #         intersection,
    #     )
    #     v = venn3(
    #         subsets=data, set_labels=(attacks[0], attacks[1], attacks[2])
    #     )
    #     plt.title(f"{row['dataset']} {row['model']}")
    #     plt.savefig(f"{OUT_ROOT}_{row['dataset']}_{row['model']}_{attacks[0]}_{attacks[1]}_{attacks[2]}.pdf")
    #     plt.clf(


def add_inverse_data(df):
    df2 = df.copy()

    df2["attack_1_name"], df2["attack_2_name"] = (
        df["attack_2_name"].copy(),
        df["attack_1_name"].copy(),
    )
    df2["attack_1_count"], df2["attack_2_count"] = (
        df["attack_2_count"].copy(),
        df["attack_1_count"].copy(),
    )

    return pd.concat([df, df2])


def plot_heatmap(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df = df.groupby(["attack_1_name", "attack_2_name"]).sum().reset_index()
    df["one_coverage"] = df["intersection"] / df["attack_2_count"]

    pivot = df.pivot(
        index="attack_1_name", columns="attack_2_name", values="one_coverage"
    )

    sns.heatmap(
        pivot,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap=sns.cubehelix_palette(as_cmap=True),
    )
    plt.savefig(path)
    plt.clf()


mapper = {
    "CPGD": 0,
    "CPGD+R00": 1,
    "CPGD+R01": 2,
    "CPGD+R10": 3,
    "CPGD+R11": 4,
    "CAPGD": 5,
}


def plot_heatmap_addition(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update({"font.size": 14})
    df["a_sort"] = df["attack_1_name"].map(mapper)
    df["b_sort"] = df["attack_2_name"].map(mapper)
    df = df.sort_values(by=["b_sort", "a_sort"], ascending=[False, False])

    df = (
        df.groupby(["attack_1_name", "attack_2_name"], sort=False)
        .sum()
        .reset_index()
    )

    total = df["attack_1_count"] + df["attack_2_count"] - df["intersection"]
    df["left_out_percent"] = df["attack_1_count"] / total

    pivot = df.pivot_table(
        index="attack_1_name",
        columns="attack_2_name",
        values="left_out_percent",
        sort=False,
        # aggfunc="first",
    )

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        pivot,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap=sns.cubehelix_palette(as_cmap=True),
    )
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.85, top=0.85)

    plt.xlabel("Coverred Attack (A)")
    plt.ylabel("Coverring Attack (B)")

    ax.invert_yaxis()
    plt.savefig(path)
    plt.clf()


def attack_name(df: pd.DataFrame) -> pd.DataFrame:

    cols = ["attack_1_name", "attack_2_name", "attack_3_name"]

    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(data_order["attack"])

    return df


def run():
    df = pd.read_csv(IN_PATH)
    df = add_inverse_data(df)

    df = attack_name(df)
    # print(df.head())
    # exit(0)
    # suprise = df[(df["attack_1_name"] == "CAPGD")& (df["attack_2_name"] == "CAA") & ((df["attack_1_count"] - df["intersection"]) > 0)]
    # print(suprise)
    # exit(0)
    # for dataset in df["dataset"].unique():
    #     for model in df["model"].unique():
    #         filter_dataset = df["dataset"] == dataset
    #         filter_model = df["model"] == model
    #         df_local = df[filter_dataset & filter_model]
    #         path = f"{OUT_ROOT}/{dataset}_{model}.pdf"
    #         plot_heatmap(df_local, path)

    # plot_heatmap(df, f"{OUT_ROOT}/all.pdf")
    plot_heatmap_addition(df, f"{OUT_ROOT}/all_total.pdf")
    # plot_venn(df, f"{OUT_ROOT}/venn")

    df3 = pd.read_csv(IN_PATH3)
    df3 = attack_name(df3)

    print(df3.head())

    three_way_wenn(df3, attacks=df3["attack_1_name"].unique())


if __name__ == "__main__":
    run()
