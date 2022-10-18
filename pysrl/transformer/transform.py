"""Transformer for raw_data after organization and prior to preparation"""
import os

from ..config.constants import INPUT_PATH
from ..config.helper import save_data
from .transform_fcts import *

# Suppress mistaken warning in df operations
pd.options.mode.chained_assignment = None


def do_trafo(formats=('csv', )):
    """Do trafo function to be called from main or directly

    Args:
        formats (tuple): The formats to save resulting data to (csv and/or xlsx)

    """
    print("Transform data_clean.csv to get proper numerical dataframe")

    # Load the cleaned up raw_data from data_clean.csv
    df = pd.read_csv(path.join(DATA_PATH, "data_clean.csv")).iloc[:, 1:]

    # Perform the transformations (it is possible to skip one by leaving
    # it out, but conflicts might result, requiring manual maintenance)

    df = df[["Date/Time", "User", "Title", "Topic",
             "Category", "Type", "Label"]]

    df = drop_incomplete(df)
    if os.path.isfile(os.path.join(INPUT_PATH, 'task_results.csv')):
        df = add_task_results(df)
    if os.path.isfile(os.path.join(INPUT_PATH, 'test_results.csv')):
        df = add_test_results(df)
    df = user_to_id(df, select="HE")

    df = time_to_seconds(df, correct_zero=60)

    df = split_type(df)
    df = map_columns(df)
    df = fill_na(df)
    df = drop_unwanted_columns(df, ["Title", "Topic"])

    df = df.dropna()
    df = add_user_cum_seconds(df)

    if os.path.isfile(os.path.join(INPUT_PATH, 'task_difficulties.csv')):
        df = add_task_difficulties(df)
    df = set_column_dtypes(df)

    # Save the resulting raw_data frame as csv:
    save_data(df, filename='data_trafo', formats=formats)

    print("Finished transformation, generated DATA_PATH/data_trafo.csv")
