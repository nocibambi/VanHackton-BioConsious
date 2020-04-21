"""Loading and preparing the CRO2 dataset for forecasting
"""


import pandas as pd
from os import path
from .config import config

from IPython.display import display

cro2_columns = [
    "Time Stamp (min)",
    "Measured_mg_dl",
    "Transformed glucose value (mmol/l)",
    # "Normalised Transform",
    "Measured_HeartRate",
    # "NORMALISED HeartRate",
]


def _load_data():

    df = pd.read_excel(path.join(config.new_data_path, "CRO2 _test_data subset.xlsx"))

    df = df.iloc[1:, :].copy()

    df.loc[:, "subject"] = (
        df["subject code"]
        + " - "
        + df[["First_Name", "Last_Name"]].sum(axis=1).str.upper()
    ).fillna(method="ffill")

    df = df.loc[:, ["subject"] + cro2_columns]

    return df


def _rename_columns(df):

    df = df.rename(
        columns={
            "Time Stamp (min)": "timestamp",
            "Measured_mg_dl": "glucose_mg_dl",
            "Transformed glucose value (mmol/l)": "glucose_mmol_l",
            # "Normalised Transform": "glucose_norm",
            "Measured_HeartRate": "heart_rate",
            # "NORMALISED HeartRate": "heart_rate_norm",
        }
    )

    return df


def _sort_rows(df):
    return df.sort_values(["subject", "timestamp"])


def get_cro2():

    cro2 = _load_data()
    cro2 = _rename_columns(cro2)
    cro2 = _sort_rows(cro2)

    return cro2
