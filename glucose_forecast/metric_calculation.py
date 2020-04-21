from .utils import load_window_data
from .config import config


def get_y_test(subject, predictors, window):
    train_size, test_size = config.train_test_size

    _, y = load_window_data(subject, predictors, window)
    y_test = y[train_size : train_size + test_size]

    if y_test.shape[0] == 0:
        print(f"{subject, predictors, window}: Not enough data.")
        return None

    return y_test


def get_rmse(predictions, subject, predictors, window):

    y_test = get_y_test(subject, predictors, window)

    from numpy import ndarray, sqrt
    if isinstance(y_test, ndarray):
        from sklearn.metrics import mean_squared_error

        error = sqrt(mean_squared_error(predictions, y_test))

        return error
    else:
        return None
