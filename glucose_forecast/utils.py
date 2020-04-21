from .config import config


def add_suffix(name, suffix):
    return f"{name}_{config.column_suffixes[suffix]}"


def apply_suffixes(name, suffix_list):
    for suffix in suffix_list:
        if "glucose" not in name and suffix == "glucose_type":
            pass
        else:
            name = add_suffix(name, suffix)

    return name


def get_predictor_columns(suffixes):
    return [
        [apply_suffixes(column, suffixes) for column in combination]
        for combination in config.predictor_combinations
    ]


def prepare_directory(query):
    path = config.directories[query]

    import os

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def generate_file_name(
    subject, predictors, window=None, path=None, model=None, plot=None
):
    import os
    import re

    name = "_".join(
        [
            re.sub("(.{4}).*(.{2})$", "\\1-\\2", subject),
            "-".join([p[:3].title() for p in predictors]),
            str(window),
        ]
    )

    if model:
        name = "_".join([name, re.sub("([A-Z]..).*([A-Z]..).*", "\\1\\2", model)])

    return os.path.join(path, name) if path else name


def load_window_data(subject, predictors, window):
    windows_path = prepare_directory("window_data")
    win_filename = (
        f"{generate_file_name(subject, predictors, window, windows_path)}.npz"
    )

    from numpy import load

    window_data = load(win_filename)
    return window_data["X"], window_data["y"]


def load_predictions_data(subject, predictors, window, model):
    forecasts_path = prepare_directory("forecast")
    pred_filename = (
        f"{generate_file_name(subject, predictors, window, forecasts_path, model)}.npy"
    )

    from numpy import load

    return load(pred_filename)


def prepare_results():
    from pandas import datetime, read_csv, DataFrame
    import os

    results_path = prepare_directory("results")
    results_filename = f"results_{datetime.now().isoformat()[:10]}.csv"
    results_file_path = os.path.join(results_path, results_filename)

    if os.path.exists(results_file_path):
        results = read_csv(results_file_path)
    else:
        print("No results generated today.")
        results = results = DataFrame(columns=config.result_columns)

    return results




def save_results(results):
    from pandas import datetime
    import os

    results_path = prepare_directory("results")
    results_filename = f"results_{datetime.now().isoformat()[:10]}.csv"
    results_file_path = os.path.join(results_path, results_filename)

    results.to_csv(
        results_file_path,
        mode="a+",
        index=False,
        header=True if not os.path.exists(results_file_path) else False,
    )
