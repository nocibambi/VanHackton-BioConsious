import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import callbacks

import tensorflow as tf
from keras.backend import tensorflow_backend as K

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


from time import time
from math import sqrt

from .config import config


# Parameters

models = {
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Decision Tree": DecisionTreeRegressor(),
    "MLP Regressor": MLPRegressor(),
    "LSTM": Sequential(),
}

train_size, test_size = config.train_test_size
model_parameters = config.model_parameters
horizon = config.forecasting_horizon


def _train_test_split(X, y, train_size, test_size):

    X_train = X[0:train_size]
    y_train = y[0:train_size]

    X_test = X[train_size : train_size + test_size]
    y_test = y[train_size : train_size + test_size]

    return X_train, y_train, X_test, y_test


def _scikit_model(model, X_train, y_train, X_test, multiregressor=False):
    print("Fitting: " + model)

    clf = models[model]

    if multiregressor:
        regr = MultiOutputRegressor(clf)
    else:
        regr = clf

    regr.fit(X_train, y_train)
    predictions = regr.predict(X_test)

    return predictions


# def _scale_data(X_train, y_train, X_test, window_size, predictors):
#     """As now we use normalization, this might not be necssary any more."""
#     breakpoint()

#     shape = X_train.shape
#     scaler = StandardScaler()
#     scaled_X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(shape)
#     scaled_X_train = scaled_X_train.reshape((X_train.shape[0], -1, X_train.shape[1]))

#     shape = y_train.shape
#     scaled_y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(shape)

#     shape = X_test.shape
#     scaled_X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(shape)
#     scaled_X_test = scaled_X_test.reshape((X_test.shape[0], -1, X_test.shape[1]))

#     return scaled_X_train, scaled_y_train, scaled_X_test  # , scaled_y_test


def _keras_model(model, X_train, y_train, X_test, window_size, predictors):
    print("Fitting: LSTM")
    num_threads = config.number_of_threads
    # X_train, y_train, X_test = _scale_data(X_train, y_train, X_test, window_size, predictors)

    # X_train, X_test, scaler = _scale(X_train, X_test)

    X_train = X_train.reshape((-1, 1, window_size * len(predictors)))
    # X_train = X_train.reshape(
    #     X_train.shape[0], int(X_train.shape[1] / len(predictors)), len(predictors)
    # )
    # X_train = X_train.reshape((-1, 1, X_train.shape[1]))
    X_test = X_test.reshape((-1, 1, window_size * len(predictors)))
    # X_test = X_test.reshape(
    #     X_test.shape[0], int(X_test.shape[1] / len(predictors)), len(predictors)
    # )
    # X_test = X_test.reshape((-1, 1, X_train.shape[1]))
    with tf.Session(
        config=tf.ConfigProto(
            intra_op_parallelism_threads=num_threads,
            inter_op_parallelism_threads=num_threads,
            allow_soft_placement=True,
            device_count={"CPU": num_threads},
            log_device_placement=True,
        )
    ) as sess:

        import os

        os.environ["GOTO_NUM_THREADS"] = f"{num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        os.environ["openmp"] = "True"

        model = Sequential()
        model.add(LSTM(56, input_shape=(1, window_size * len(predictors))))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=horizon))
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])

        tensorboard = TensorBoard(log_dir="tensorboard/{}".format(time()))

        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=12)

        model.fit(
            X_train,
            y_train,
            batch_size=200,
            epochs=100,
            validation_split=0.1,
            verbose=0,
            callbacks=[tensorboard, early_stop],
        )
        # breakpoint()
        predictions = model.predict(X_test)

        # predictions = scaler.inverse_transform(predictions)

    return predictions


def _normalize(X_train, X_test):
    scaler = MinMaxScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def _scale(X_train, X_test):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler



def forecast(X, y, model, window, predictors):
    X_train, y_train, X_test, y_test = _train_test_split(X, y, train_size, test_size)

    if X_test.shape[0] == 0:
        return np.array([np.NaN]), np.NaN

    X_train, X_test = _normalize(X_train, X_test)

    if str(models[model].__class__).replace("<class '", "").startswith("sklearn"):
        predictions = _scikit_model(
            model,
            X_train,
            y_train,
            X_test,
            multiregressor=model_parameters[model].get("multiregressor", False),
        )

    elif str(models[model].__class__).replace("<class '", "").startswith("keras"):
        predictions = _keras_model(model, X_train, y_train, X_test, window, predictors)
    else:
        raise ValueError(f"Wrong model name: {model}\nPossible models: {models.keys()}")

    # predictions = _inverse_normalize(predictions, scaler)
    error = sqrt(mean_squared_error(predictions, y_test))

    return predictions, error
