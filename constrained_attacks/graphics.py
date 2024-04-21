from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from matplotlib import rc

# from pandas.plotting import register_matplotlib_converters

rc("text", usetex=True)
# font = {
#         'weight' : 'bold',
# }
# rc('font', **font)
# # matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
# plt.rc('text.latex', preamble=r'\boldmath')
# # register_matplotlib_converters()

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "text.latex.preamble": r"\boldmath",

#         # Enforce default LaTeX font.
#         "font.family": "serif",
#         "font.serif": ["Computer Modern"],
#     }
# )
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]# - global options
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
FIGURE_FOLDER = "data/fig/20240207_0/"
EXTENSION = ".pdf"
FONT_SCALE = 1.3
DPI = 300
PALETTE = "colorblind"
OUTSIDE_LEGEND = (1.05, 0.5)
ABOVE_LEGEND = (0, 1.02, 1, 0.2)
FONT_WEIGHT = "bold"
FULL_OUTSIDE_LEGEND = (1.2, 1)


def lineplot(
    data,
    name,
    x,
    y,
    y_label="",
    hue=None,
    x_label="",
    y_lim=None,
    fig_size=(6, 4),
    legend_pos="best",
    style=None,
    markers=None,
    dashes=True,
    v_lines=[],
    h_lines=[],
    error_min_max=False,
):
    plt.figure(figsize=fig_size)
    sns.set(style="darkgrid", color_codes=True, font_scale=FONT_SCALE)

    palette = _color_palette(data, hue)

    def error_f(x):
        return numpy.min(x), numpy.max(x)

    g = sns.lineplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        style=style,
        markers=markers,
        dashes=dashes,
        linestyle="dotted",
        marker="o",
        errorbar=error_f if error_min_max else ("ci", 95),
    )

    if hue and legend_pos:
        handles, labels = g.get_legend_handles_labels()
        if legend_pos == "outside":
            plt.legend(
                loc="center left",
                bbox_to_anchor=FULL_OUTSIDE_LEGEND,
                prop={"size": FONT_SCALE * 12},
                # handles=handles[1:],
                # labels=labels[1:],
            )
        else:
            plt.legend(
                loc=legend_pos,
                prop={"size": FONT_SCALE * 12},
                # handles=handles[1:],
                # labels=labels[1:],
            )

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    #
    # g.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
    # g.xaxis.set_major_locator(plt.MaxNLocator(5))

    if y_lim is not None and len(y_lim) == 2:
        plt.ylim(y_lim)

    for x_pos in v_lines:
        plt.axvline(x=x_pos, color="k", linestyle="--")

    for y_pos in h_lines:
        plt.axhline(y=y_pos, color="k", linestyle="--")

    plt.tight_layout()

    plt.savefig(_get_filename(name), dpi=DPI)
    plt.close("all")


def boxplot(
    data,
    name,
    x,
    y,
    y_label="",
    hue=None,
    x_label="",
    fig_size=(6, 4),
    legend_pos="best",
    x_lim=None,
    **kwargs,
):
    plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=FONT_SCALE)

    palette = _color_palette(data, hue)

    sns.boxplot(x=x, y=y, hue=hue, data=data, palette=palette, **kwargs)

    plt.ylabel(y_label)
    plt.xlabel(x_label)

    if x_lim is not None and len(x_lim) == 2:
        plt.xlim(x_lim)

    _setup_legend(data, legend_pos, hue)

    plt.savefig(_get_filename(name), dpi=DPI, bbox_inches="tight")
    plt.close("all")


def violinplot(
    data,
    name,
    x,
    y,
    y_label="",
    hue=None,
    x_label="",
    fig_size=(6, 4),
    legend_pos="best",
    x_lim=None,
    split=None,
    **kwargs,
):
    plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=FONT_SCALE)

    if split is None:
        split = hue and len(data[hue].unique()) == 2

    palette = _color_palette(data, hue)

    columns = [hue] if hue is not None else []
    columns += [x] if x is not None else []
    columns += [y] if y is not None else []

    clean = data[columns]
    clean = clean[~clean.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]

    sns.violinplot(
        x=x, y=y, hue=hue, data=clean, palette=palette, split=split, **kwargs
    )

    plt.ylabel(y_label)
    plt.xlabel(x_label)

    if x_lim is not None and len(x_lim) == 2:
        plt.xlim(x_lim)

    _setup_legend(data, legend_pos, hue)

    plt.savefig(_get_filename(name), dpi=DPI, bbox_inches="tight")
    plt.close("all")


def countplot(
    data,
    name,
    x=None,
    y=None,
    y_label="",
    hue=None,
    x_label="",
    fig_size=(6, 4),
    legend_pos="best",
):
    plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=FONT_SCALE)

    if hue is not None:
        palette = _color_palette(data, hue)
    elif x is not None:
        palette = _color_palette(data, x)
    else:
        palette = _color_palette(data, y)

    sns.countplot(x=x, y=y, hue=hue, data=data, palette=palette)

    plt.ylabel(y_label)
    plt.xlabel(x_label)

    _setup_legend(data, legend_pos, hue)

    plt.savefig(_get_filename(name), dpi=DPI, bbox_inches="tight")
    plt.close("all")


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
    error_min_max=False,
    **kwargs,
):
    plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=FONT_SCALE)

    def error_f(x):
        return numpy.min(x), numpy.max(x)

    palette = _color_palette(data, hue, overlay_x)

    if overlay_x and not hue:
        sns.barplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            color=palette[0],
            errorbar=error_f if error_min_max else ("ci", 95),
        )
        sns.barplot(x=overlay_x, y=y, hue=hue, data=data, color=palette[1])

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
        sns.barplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            palette=palette,
            errorbar=error_f if error_min_max else ("ci", 95),
            **kwargs,
        )
        if legend_pos:
            _setup_legend(data, legend_pos, hue)

    if not legend_pos:
        plt.legend().remove()
        
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    if x_lim is not None and len(x_lim) == 2:
        plt.xlim(x_lim)

    if y_lim is not None and len(y_lim) == 2:
        plt.ylim(y_lim)

    plt.xticks(rotation=rotate_ticks)

    plt.savefig(_get_filename(name), dpi=DPI, bbox_inches="tight")
    plt.close("all")


def scatterplot(
    data,
    name,
    x,
    y,
    y_label="",
    hue=None,
    style=None,
    x_label="",
    fig_size=(32, 4),
    legend_pos="best",
    markers=None,
    xlim=None,
    ylim=None,
    **kwargs,
):
    # - fig = plt.figure(figsize=fig_size)
    # sns.set_style("darkgrid")

    sns.set(style="darkgrid", color_codes=True, font_scale=FONT_SCALE)
    # sns.set_style()

    palette = _color_palette(data, hue)

    legend_out = True if legend_pos == "outside" else False
    sns.lmplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        legend_out=legend_out,
        fit_reg=False,
        markers=markers,
        height=fig_size[1],
        aspect=fig_size[0] / fig_size[1],
        # scatter_kws=dict(edgecolor="none"),
        **kwargs,
    )
    _setup_legend(data, legend_pos, hue)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()

    plt.savefig(_get_filename(name), dpi=DPI)
    plt.close("all")


def _setup_legend(data, legend_pos, hue):
    if hue and legend_pos:
        if legend_pos == "above":
            plt.legend(
                ncol=len(data[hue].unique()),
                loc="lower left",
                bbox_to_anchor=ABOVE_LEGEND,
                prop={"size": FONT_SCALE * 12},
            )
        elif legend_pos == "outside":
            plt.legend(
                loc="upper center",
                bbox_to_anchor=FULL_OUTSIDE_LEGEND,
                prop={"size": FONT_SCALE * 12},
            )
        else:
            plt.legend(loc=legend_pos, prop={"size": FONT_SCALE * 12})


def _get_filename(name):
    path = FIGURE_FOLDER + name + EXTENSION
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _color_palette(data, hue, is_overlay=None):
    n_colors = len(data[hue].unique()) if hue else 1
    n_colors = 2 if is_overlay else n_colors

    return sns.color_palette(PALETTE, n_colors=n_colors)
