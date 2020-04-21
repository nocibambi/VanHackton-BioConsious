from .config import config
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import generate_file_name


def _format_plot(ax, plot_name):

    ax.set_title(f"{plot_name}")
    ax.set_xlabel("window size")
    ax.set_ylabel("RMSE")
    ax.set_ylim((0, 300))
    ax.set_xlim((config.training_window_range[0], config.training_window_range[1]))

    return ax


def _save_plot(fig, plot_name):
    from .utils import prepare_directory
    import os

    path = prepare_directory("summary_plot")
    filepath = os.path.join(path, f"{plot_name}.png")

    fig.savefig(filepath)


def generate_plot(results, subject, predictors, show=False):

    df = results.loc[
        (results["subject"] == subject)
        & (results["predictors"].astype(str) == str(predictors)),
        ["window", "model", "error"],
    ].reset_index(drop=True)

    plt.clf()
    sns.set(font_scale=2)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    sns.lineplot(y=df["error"], x=df["window"], hue=df["model"], ax=ax)

    plot_name = generate_file_name(subject, predictors, "RMSE")
    ax = _format_plot(ax, plot_name)
    _save_plot(fig, plot_name)

    if show:
        plt.show()

    plt.close()
