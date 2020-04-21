import pandas as pd
import numpy as np

from .data_validation import validate_sort
from .utils import add_suffix

from .config import config


cols_to_normalize = [k for k, v in config.columns.items() if v.get("normalize", False)]
renamed_cols = {col: add_suffix(col, "normalize") for col in cols_to_normalize}


def _normalize_array(array):
    return (array - array.min()) / (array.max() - array.min())


def _normalize_cols(df):
    norm_cols = (
        df[cols_to_normalize].apply(_normalize_array).rename(columns=renamed_cols)
    )

    df = pd.concat((df, norm_cols), axis=1)

    return df


def _validate_col_norm(df):
    for col in cols_to_normalize:
        norm_col = add_suffix(col, "normalize")
        
        inverse_norm = df[norm_col] * (df[col].max() - df[col].min()) + df[col].min()

        status = np.isclose(inverse_norm, df[col]).all()

        assert status, f"{col} inverse normalization does not align."


def _validate_normalization(df):
    df.groupby("subject").apply(_validate_col_norm)
    print("Inverse normalizations align.")


def _validate_col_equivalence(df1, df2):
    assert (df1[cols_to_normalize] == df2[cols_to_normalize]).all().all()


def normalize(df):
    print("Normalization...")
    norm_df = df.groupby("subject").apply(_normalize_cols)

    _validate_normalization(norm_df)
    _validate_col_equivalence(norm_df, df)
    validate_sort(norm_df)

    return norm_df
