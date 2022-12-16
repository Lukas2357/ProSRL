"""Helper functions for the pysrl package"""

import os
import pathlib
import shutil
import warnings
from pathlib import Path
import pandas as pd
from .constants import YEAR, DATA_PATH, ROOT

FILE_PATH = pathlib.Path(__file__).parent.resolve()


def clear_directory(path: Path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def load_data(filename: str, rm_first_col=True) -> pd.DataFrame:
    """Load any raw_data csv file from the raw_data folder

    Args:
        filename (str): Name of the csv file to load
        rm_first_col (bool): Whether to remove the first column (xlsx files)

    Returns:
        pd.DataFrame: Dataframe of all file raw_data

    """
    data_file = os.path.join(DATA_PATH, filename)
    data_df = pd.read_csv(data_file)

    if rm_first_col:
        return data_df.iloc[:, 1:]

    else:
        return data_df


def load_input(filename: str, columns=None, sep=',', year=None) -> pd.DataFrame:
    """Load any data csv file from the input folder

    Args:
        filename (str): Name of the csv file to load
        columns (optional): Columns to look for in the csv file
        sep (str): Separator for csv files
        year (int): The year from which to load data

    Returns:
        pd.DataFrame: Dataframe of all file data

    """

    if year is None:
        year = YEAR

    input_path = os.path.join(ROOT, "input", year)
    if year == '2022' and filename == 'data_complete.csv':
        sep = ';'

    warn_m = f"Could not find file {filename} in folder {input_path} \n"

    input_file = os.path.join(input_path, filename)
    if os.path.isfile(input_file):
        return pd.read_csv(input_file, sep=sep)
    if os.path.isfile(input_file + '.csv'):
        return pd.read_csv(input_file + '.csv', sep=sep)

    warnings.warn(warn_m + f'Look through other folders in {ROOT}. Highly '
                           f'recommend to place {filename} in {input_path}')

    for path, _, files in os.walk(ROOT):
        for name in files:
            if filename in name:
                warnings.warn(warn_m + f'But in {path}... load from here.')
                return pd.read_csv(os.path.join(path, name), sep=sep)

    warnings.warn(f'Could not find {filename} elsewhere... Look for '
                  f'differently named suitable files. Highly recommend to '
                  f'place {filename} in {input_path}')

    for path, _, files in os.walk(ROOT):
        if 'pysrl' not in path and 'raw_data' not in path:
            for name in files:
                warn_m = warn_m + 'But file {name} in {path} contains ' \
                                  'suitable columns. Try using this one...'
                if '.csv' in name:
                    df = pd.read_csv(os.path.join(path, name), sep=sep)
                    if all(cols in df.columns for cols in columns):
                        warnings.warn(warn_m)
                        return df

    warnings.warn(f'Could not find suitable file for {filename}. Return empty '
                  f'df. Program is expected to behave incorrect...')

    return pd.DataFrame()


def get_feature_types() -> list:
    """Load feature types from feature_types.txt

    Returns:
        list: List of existing feature types

    """

    with open(os.path.join(FILE_PATH, 'feature_types.txt'), 'r') as f:
        feature_types = f.read().splitlines()

    return feature_types


def set_root(root=""):
    """Set the root path in root.txt

    Args:
        root (str): The root path to write to root.txt

    """

    with open(os.path.join(FILE_PATH, 'root.txt'), 'w') as f:
        if root:
            f.write(root)
        else:
            f.write(os.getcwd())


def save_data(df: pd.DataFrame, filename: str, formats=('csv', 'xlsx')):
    """Save the raw_data in CSV format

    Args:
        df (pd.DataFrame): The raw_data as pandas dataframe
        filename (str): The name of the file to save
        formats (tuple): The formats to save the raw_data in

    """

    if 'csv' in formats:
        data_file = os.path.join(DATA_PATH, filename + '.csv')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        df.to_csv(data_file)

    if 'xlsx' in formats:
        data_file = os.path.join(DATA_PATH, filename + '.xlsx')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        df.to_excel(data_file)
