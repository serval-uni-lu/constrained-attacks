from pathlib import Path

import pandas as pd
import pylab as plt
from matplotlib_venn import venn2, venn2_circles

OUT_ROOT = "./data/fig/ecai24-20240422_venn2/"


def plot_venn(data, attacks, path, title=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    print(data)
    v = venn2(subsets=data, set_labels=(attacks[0], attacks[1]))
    # v.get_label_by_id("10").set_text(attacks[0])
    # v.get_label_by_id("01").set_text(attacks[1])
    # v.get_label_by_id("11").set_text("Third")
    plt.title(title)
    plt.savefig(path)
    plt.clf()


def plot_each_file(df, attacks):
    for index, row in df.iterrows():
        intersection = row["intersection"]
        data = (
            row[attacks[0]] - intersection,
            row[attacks[1]] - intersection,
            intersection,
        )
        path = f"{OUT_ROOT}_{row['dataset']}_{row['model']}_{attacks[0]}_{attacks[1]}.pdf"
        plot_venn(
            data, attacks, path, title=f"{row['dataset']} {row['model']}"
        )


def subfig_venn(df, attacks, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(len(df))
    fig.suptitle(f"{df['dataset']} {df['model']}")

    for index, row in df.iterrows():
        intersection = row["intersection"]
        data = (
            row[attacks[0]] - intersection,
            row[attacks[1]] - intersection,
            intersection,
        )
        v = venn2(
            subsets=data, set_labels=(attacks[0], attacks[1]), ax=axs[index]
        )
    plt.savefig(path)
    plt.clf()


def run():
    path = "intersection_apgd_pgdl2_apgd.csv"
    attacks = ["pgdl2", "apgd"]

    df = pd.read_csv(path)
    plot_each_file(df, attacks)
    # subfig_venn(df, attacks, f"{OUT_ROOT}_all.pdf")


if __name__ == "__main__":
    run()
