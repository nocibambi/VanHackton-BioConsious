import pandas as pd
import numpy as np
from .config import config
import os

from .utils import (
    apply_suffixes,
    add_suffix,
    get_predictor_columns,
    generate_file_name,
    prepare_directory,
)

from tqdm import tqdm


def _generate_window(df, variable_to_predict, window, horizon):

    X = pd.DataFrame()
    y = pd.DataFrame()

    for i in tqdm(range(0, len(df) + 1, 1)):
        """Checks if
            - the series length minus the step number is greater than
            the sum of the window size and the horizon
            and
            - if there is no time gap in that particular window of y
            greater than 7 minutes

            E.g. 100 - 5 = 95 > 52 + 12 = 64
            E.g. 100 - 40 = 60 < 52 + 12 = 64"""

        if i + window + horizon < len(df):

            """The past glucose window
            1-52, 2-53, etc..."""

            """Appends feature values to the past glucose values in the same window
            The results is a single list of values with the prediction features
            stacked after each other.
            """
            # breakpoint()
            temp_X = np.array([])
            for col in df:
                temp_X = np.append(temp_X, df[i : i + window][col].values)

            """Defines the glucose values to predict.
            These are the values following the glucose values above with the size
            of the window defined in the horizon
            E.g. 53-65, 54-66 """
            temp_y = pd.Series(
                df.iloc[i + window : i + window + horizon][variable_to_predict].values
            )

            """Appends the X and y values to the pipe as separate rows
                
            E.g. After two iterations with three columns
            > pipeX.shape
            (2, 156)
            > pipeY.shape
            (2, 12)
            > 52 * 3
            156
            """
            X = X.append(pd.Series(temp_X), ignore_index=True)
            y = y.append(temp_y, ignore_index=True)

    return X, y


def create_window(df, subject, predictors, window, retfiles=False):
    """Separating instances of forecasting (discarting weeks with missing data)
    Iterates over window sizes between 48 and 150 with jumps of 4
    52, 56, 60, ..., 148"""
        
    variable_to_predict = add_suffix(config.variable_to_predict, "glucose_type")

    df_subject = df.loc[
        df["subject"] == subject, set(predictors + [variable_to_predict])
    ]

    horizon = config.forecasting_horizon
    X, y = _generate_window(df_subject, variable_to_predict, window, horizon,)

    assert df_subject.shape[0] == X.shape[0] + window + horizon, "Wrong final shape"
    assert y.shape[1] == horizon

    windows_path = prepare_directory("window_data")
    filename = generate_file_name(subject, predictors, window, windows_path) + ".npz"
    np.savez(filename, X=X, y=y)

    if retfiles:
        return X, y


