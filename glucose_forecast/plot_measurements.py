import os
from .config import config
from .utils import prepare_directory

import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update(**config.rc)
sns.set_style(rc=config.rc)


def _style_plot(axes):
    axes.legend(loc=0)
    axes.grid("on")

    return axes


def _generate_plot(df, column, subject):
    _, axes = plt.subplots(1, 1, figsize=(20, 8))

    g = sns.lineplot(x=df["timestamp"], y=df[column], ax=axes)

    axes.set_xlabel("Timestamp")
    axes.set_ylabel(column.title())
    axes.set_title(subject + f": {column}")

    axes = _style_plot(axes)

    return g


def _generate_histogram(df, column, subject):
    
    _, axes = plt.subplots(1, 1, figsize=(20, 8))

    g = sns.distplot(df[column], ax=axes)

    axes.set_xlabel(column.title())
    axes.set_title(subject + f": {column} distribution.")

    axes = _style_plot(axes)

    return g


def _plot_file_path(subject, metric, path):

    filename = f"{subject}_{metric}.png"
    return os.path.join(path, filename)


def _plot_metrics(df, column, subject, show_plots):


    path = prepare_directory('plots')
    file_path = _plot_file_path(subject, column, path)

    if not os.path.exists(file_path):
        g = _generate_plot(df, column, subject)
        g.figure.savefig(file_path)
    else:
        print(f"{file_path} exists.")

    if show_plots:
        plt.show()
    plt.close()


def generate_plots(df, show_plots=False):
    for subject in df["subject"].unique():
        df_subject = df[df["subject"] == subject]

        for column in config.columns_to_plot:
            _plot_metrics(df_subject, column, subject, show_plots)
