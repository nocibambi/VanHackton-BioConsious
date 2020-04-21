import logging
import numpy as np

# Logging
log_filename = "log/glucose_forecast.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG)


# Paths to data folders
old_data_path = "data/old"
new_data_path = "data/new"
results_path = "results"


directories = {
    "results": f"{results_path}",
    "plots": f"{results_path}/figures",
    "window_data": f"{results_path}/windows",
    "forecast": f"{results_path}/forecast",
    "Clarke": f"{results_path}/clarke",
    "summary_plot": f"{results_path}/summary_plots",
}


maxgapsize = 7


# Columns for modeling
columns = {
    "subject": {"dtype": np.object},
    "timestamp": {"dtype": np.float64},
    "glucose_mg_dl": {"dtype": np.float64, "plot": True, "normalize": True},
    "glucose_mmol_l": {"dtype": np.float64, "plot": True, "normalize": True},
    "heart_rate": {"dtype": np.float64, "plot": True, "normalize": True},
}


# Plotting
# matplotlib style parameters
rc = {
    "font.size": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}

columns_to_plot = [
    "glucose_mg_dl",
    "glucose_mmol_l",
    "heart_rate",
]

# Forecasting
# Forecasting windows
training_window_range = (48, 152 + 1)  # 56 152
training_window_step = 4
forecasting_horizon = 4

variable_to_predict = "glucose"

column_suffixes = {
    "glucose_type": "mg_dl",  # 'mmol_l'
    "normalize": "norm",
}


predictor_combinations = [["glucose"], ["glucose", "heart_rate"]]

train_test_size = (int(36 * 60 / 15), int(12 * 60 / 15))


model_parameters = {
    "Linear Regression": {"multiregressor": True,},
    "Decision Tree": {"multiregressor": True,},
    "MLP Regressor": {"scale": True},
    "LSTM": {"scale": True},
}


errors = ["RMSE"]

result_columns = ["subject", "predictors", "window", "model", "error", "clarke", "zone"]

# 
number_of_threads = 3
