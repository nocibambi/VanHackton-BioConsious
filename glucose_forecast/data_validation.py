import pandas as pd
from .config import config
from IPython.display import display


def gap_check(index, max_gap=config.maxgapsize):
    """TODO Currently does not work need review
    
    Checks if there is a gap between the pairs of timestamps 
    bigger than 'maxGap' in minutes
    """
    # for i in range(0, len(index) - 1):
    #     if (pd.Timedelta(index[i + 1] - index[i]).seconds) / 60 > maxGap:
    #         return True
    # return False


def get_intervals(df):
    return df.diff().iloc[1:].round(10).unique()


def validate_intervals(df):

    intervals = df.groupby("subject")["timestamp"].apply(get_intervals)

    # Checks if the timestamp intervals are consistent within and between
    # subject-level datsets
    assert (
        intervals.apply(len).max() == 1 and intervals.apply(lambda x: x[0]).nunique()
    ), f"""
    Intervals between timestamps are inconsistent:
    {intervals.apply(len).value_counts()}"""

    print(
        f"Intervals between timestamps are consistently \
            {intervals.apply(lambda x: x[0]).unique()[0]}."
    )


def validate_data_types(df):
    if all([config.columns[col]["dtype"] == df[col].dtype for col in config.columns]):
        print("Column data types are valid.")
    else:
        typediffs = pd.DataFrame(
            [
                [col, config.columns[col]["dtype"].__name__, df[col].dtype]
                for col in config.columns
            ],
            columns=["column", "expected", "actual"],
        ).set_index("column")
        raise TypeError(f"Invalid column data types:\n{typediffs}")


def validate_sort(df):
    assert (
        (
            df.reset_index(drop=True)
            == df.sort_values(["subject", "timestamp"]).reset_index(drop=True)
        )
        .all()
        .all()
    ), "Rows are not sorted by subject and timestamp!"

    print("Rows are sorted by subject and timestamp")


def validate_subject_names(df):
    assert (
        pd.Series(df["subject"].unique())
        .str.replace("(.{4}).*(.{2})$", r"\1-\2")
        .value_counts()
        == 1
    ).all(), "'subject': code-initial duplicates"


def validate_data(df):

    # gap_check(df)

    validate_intervals(df)
    validate_data_types(df)
    validate_sort(df)
    validate_subject_names(df)
