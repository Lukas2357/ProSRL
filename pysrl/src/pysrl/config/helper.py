"""Helper functions for the pysrl package"""

import os
import shutil
import warnings
from pathlib import Path
import pathlib
import pandas as pd

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


def load_data(filename: str) -> pd.DataFrame:
    """Load any data csv file from the data folder

    Args:
        filename (str): Name of the csv file to load

    Returns:
        pd.DataFrame: Dataframe of all file data

    """
    data_file = os.path.join(get_root(), "data", filename)
    data_df = pd.read_csv(data_file)

    return data_df


def load_input(filename: str, columns=None) -> pd.DataFrame:
    """Load any data csv file from the input folder

    Args:
        filename (str): Name of the csv file to load
        columns (optional): Columns to look for in the csv file

    Returns:
        pd.DataFrame: Dataframe of all file data

    """

    root = get_root()
    input_path = os.path.join(root, "input")

    warn_m = f"Could not find file {filename} in folder {input_path} \n"

    input_file = os.path.join(input_path, filename)
    if os.path.isfile(input_file):
        return pd.read_csv(input_file)
    if os.path.isfile(input_file + '.csv'):
        return pd.read_csv(input_file + '.csv')

    warnings.warn(warn_m + f'Look through other folders in {root}. Highly '
                           f'recommend to place {filename} in {input_path}')

    for path, _, files in os.walk(root):
        for name in files:
            if filename in name:
                warnings.warn(warn_m + f'But in {path}... load from here.')
                return pd.read_csv(os.path.join(path, name))

    warnings.warn(f'Could not find {filename} elsewhere... Look for '
                  f'differently named suitable files. Highly recommend to '
                  f'place {filename} in {input_path}')

    for path, _, files in os.walk(root):
        if 'pysrl' not in path and 'data' not in path:
            for name in files:
                warn_m = warn_m + 'But file {name} in {path} contains ' \
                                  'suitable columns. Try using this one...'
                if '.csv' in name:
                    df = pd.read_csv(os.path.join(path, name))
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


def get_root() -> str:
    """Get the root path from root.txt

    Return:
        str: The root path as string

    """

    m = "Could not specify ROOT. Go to src/pysrl/config, open root.txt, enter" \
        " the path of the desired root folder in the first line, save and " \
        "rerun. Alternatively run main.py to specify its location as ROOT."

    with open(os.path.join(FILE_PATH, 'root.txt'), 'r') as f:
        root = f.read().splitlines()[0]

    if not os.path.isdir(root):
        raise FileNotFoundError(m)

    return root


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
    """Save the data in CSV format

    Args:
        df (pd.DataFrame): The data as pandas dataframe
        filename (str): The name of the file to save
        formats (tuple): The formats to save the data in

    """

    data_path = os.path.join(get_root(), "data")

    if 'csv' in formats:
        data_file = os.path.join(data_path, filename + '.csv')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        df.to_csv(data_file)

    if 'xlsx' in formats:
        data_file = os.path.join(data_path, filename + '.xlsx')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        df.to_excel(data_file)
