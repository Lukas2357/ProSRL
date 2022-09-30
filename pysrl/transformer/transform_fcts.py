"""Functions for the transformer.py module"""

import pandas as pd
from ..config.helper import load_input

from .transform_help_fcts import *


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows having nan values in time, user or label column

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    # There is a user having no number, but it should be 920, correct that:
    df.loc[df.User == 'LBenutzer', "User"] = "LBenutzer HE1021-920"
    
    return df.dropna(subset=['Date/Time', 'User', 'Label'])


def user_to_id(df: pd.DataFrame, select="LBenutzer HE") -> pd.DataFrame:
    """Replace the username by a unique integer id

    Args:
        df (pd.DataFrame): Input dataframe
        select (str): A string to select users with it

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    df = df[df["User"].str.contains(select)]
    user_id = df["User"].str[-3:].astype(int)
    df["User"] = user_id - min(user_id)
    unique_ids = sorted(df['User'].unique())
    df["User"] = df['User'].map(dict(zip(unique_ids, range(len(unique_ids)))))

    return df


def time_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Replace timestamp with minutes since first activity

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    df = df.astype({'Date/Time': 'datetime64'})
    df = df.sort_values(by=['User', 'Date/Time'])
    
    # Add a column for the time spent on each page:
    df['SecSpent'] = (- df['Date/Time'].diff(periods=-1))
    df['SecSpent'] = df['SecSpent'].fillna(pd.Timedelta(seconds=10))
    df['SecSpent'] = df['SecSpent'].dt.total_seconds().astype(int)
    df.loc[df.SecSpent < 0, 'SecSpent'] = 0
    df['TotSec'] = df['Date/Time'] - min(df['Date/Time'])
    df['TotSec'] = df['TotSec'].dt.total_seconds().astype(int)

    return df.drop(['Date/Time'], axis=1)


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    """Fill na values in columns with -1

    Args:
        df (pd.DataFrame): The df corresponding to data_clean.csv

    Returns:
        pd.DataFrame: Corrected df
        
    """
    df['Title'].fillna(-1, inplace=True)
    df['Topic'].fillna(-1, inplace=True)
    df['Category'].fillna(-1, inplace=True)
    df['Level'].fillna(-1, inplace=True)
    df['LearnType'].fillna(6, inplace=True)

    return df


def split_type(df: pd.DataFrame) -> pd.DataFrame:
    """Split the type column into the learn type (Testen, Üben, Erarbeiten) and
        the level (Grundwissen, erweitertes Wissen)

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    df[["LearnType", "Level"]] = df["Type"].str.split(expand=True).iloc[:, 0:2]

    return df.drop(["Type"], axis=1)


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map categorical columns to discrete values

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    
    categories = list(get_categories().values())
    topics = list(get_topics().values())
    levels = list(get_levels().values())

    df = column_map(df, "Category", categories)
    df = column_map(df, "Topic", topics)
    df = column_map(df, "Level", levels)
    df = column_map(df, "Title", df["Title"].unique())

    # Present learntypes in the cleaned csv:
    learn_types = ['Testen', 'Üben', 'Erarbeiten']
    # Map those to 0, 1, 2:
    df = column_map(df, "LearnType", learn_types)
    # Now sort properly as: 0 -> Testen, 1 -> Ü_Info, 2 -> Üben,
    # 3 -> Kurzaufgaben, 4 -> BK_Info, 5 -> Beispiele, 6 -> Rest
    df['LearnType'] = df['LearnType'].map(dict(zip(range(3), [0, 2, 3])))
    # Add Rückmeldungen zu Testen since they belong here:
    df.loc[df["Label"] == "Rückmeldungen", "LearnType"] = 0
    # LearnType 5: Beispiele (Label contains 'b') ->
    df.loc[df["Label"].str[5:].str.contains("b"), "LearnType"] = 5
    # LearnType 4: KB_Info (Label 5 symbols long) ->
    df.loc[df['Label'].str.len() == 5, "LearnType"] = 4
    # LearnType 1: Ü_Info (Label 4 symbols long and in Üben) ->
    df.loc[(df['Label'].str.len() == 4) & (df["LearnType"] == 2),
           "LearnType"] = 1
    # LearnType 6: Übersicht (everything that does not fit elsewhere) ->
    df.loc[(df['Label'].str.len() == 4) & (df["LearnType"] != 1),
           'LearnType'] = 6
    df['LearnType'].fillna(6, inplace=True)
    
    # Rückmeldungen do not have Category or level, propagate them forward:
    df[['Category', 'Level']] = df[['Category', 'Level']].fillna(method='ffill')

    return df


def unmap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Undo mapping of categorical columns to discrete values

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Back-Transformed dataframe

    """
    
    maps = {'Category': get_categories(), 'Topic': get_topics(),
            'Level': get_levels(), 'LearnType': get_learn_types()}

    for key, value in maps.items():
        if key in df.columns:
            df[key] = df[key].map(value)
    
    return df


def drop_unwanted_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drops unwanted columns from dataframe (lossy)

    Args:
        df (pd.Dataframe): Input dataframe
        columns (list): A list of column names

    Returns:
        pd.Dataframe: Transformed dataframe

    """
    return df.drop(columns, axis=1)


def add_user_cum_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column showing the accumulated time in actual learn pages

    Args:
        df (pd.DataFrame): The input dataframe

    Returns:
        pd.DataFrame: The extended df
        
    """
    learn_time = df[df.LearnType < 4].groupby(['User'])['SecSpent']
    browse_time = df[df.LearnType >= 4].groupby(['User'])['SecSpent']

    df.loc[df.LearnType < 4, 'UserCumSec'] = learn_time.cumsum()
    df.loc[df.LearnType >= 4, 'UserCumSec'] = browse_time.cumsum()
    
    return df


def set_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set proper datatypes for dataframe columns

    Args:
        df (pd.Dataframe): Input dataframe

    Returns:
        pd.Dataframe: Transformed dataframe

    """
    data_type_map = dict(zip(df.columns, ['int64']*len(df.columns)))
    data_type_map["Label"] = 'str'
    return df.astype(data_type_map)


def add_task_difficulties(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with task difficulties to the transformed raw_data

    Args:
        df (pd.DataFrame): The transformed df

    Returns:
        pd.DataFrame: The extended df
    
    """
    diffs = load_input('task_difficulties.csv', columns=['Label', 'Niveau'])
    diffs.columns = ['Label', 'Niveau']
    mapper = dict(zip(['niedrig', 'mittel', 'hoch'], range(3)))
    diffs['difficulty'] = diffs['Niveau'].map(mapper).astype(int)
    diffs['Label'] = diffs['Label'].replace({'U': 'ue', 'KD': 'kd', 'KS': 'ks'}, 
                                            regex=True)
    
    df = pd.merge(df, diffs, how='left')
    df['difficulty'].fillna(-1, inplace=True)
    
    return df.drop('Niveau', axis=1)
    
    
def add_task_results(df: pd.DataFrame) -> pd.DataFrame:
    """Include the results of tasks from task_results.csv in user activity df

    Args:
        df (pd.DataFrame): The transformed df

    Returns:
        pd.DataFrame: The extended df
    
    """
    required_columns = ['User', 'Date/Time', 'ResponsTask', 'Task']
    results = load_input('task_results.csv', columns=required_columns)
    results.dropna(subset=['ResponsTask'], inplace=True)
    mapper = dict(zip(['f', 'twr', 'r'], range(3)))
    results['ResponsTask'] = results['ResponsTask'].map(mapper).astype(int)

    df, results = correct_columns_for_merge(df, results)
    
    results.sort_values(by='Time', inplace=True)
    df.sort_values(by='Time', inplace=True)

    results.drop(['Aufgabenniveau', 'Code', 'Task', 'User', 'Date/Time'],
                 axis=1, inplace=True)
    
    df = pd.merge_asof(df, results, on='Time', by=['User_id'], 
                       tolerance=pd.Timedelta(seconds=1), direction='nearest')
        
    df['ResponsTask'].fillna(-1, inplace=True)
    df.drop(['Time', 'User_id'], axis=1, inplace=True)
    
    return df


def add_test_results(df: pd.DataFrame) -> pd.DataFrame:
    """Include the results of test from test_results.csv in user activity df

    Args:
        df (pd.DataFrame): The transformed df

    Returns:
        pd.DataFrame: The extended df
    
    """
    required_columns = ['User', 'Date/Time', 'Testergebnis 1', 'Test',
                        'Testergebnis 2']
    results = load_input('test_results.csv', columns=required_columns)
    
    results.dropna(subset=['Testergebnis 1'], inplace=True)
    results.dropna(subset=['Testergebnis 2'], inplace=True)
    mapper = dict(zip(['nicht bestanden', 'bestanden', 'gut bestanden'], 
                      range(3)))
    results['TestResQual'] = results['Testergebnis 2'].map(mapper).astype(int)
    results['TestResQuant'] = results['Testergebnis 1'].astype(int)

    df, results = correct_columns_for_merge(df, results)
    
    results.drop(['Testergebnis 2', 'Testergebnis 1', 'User', 
                  'Date/Time'], axis=1, inplace=True)
    
    df = pd.merge_asof(df, results, on='Time', by=['User_id'], 
                       tolerance=pd.Timedelta(seconds=120), direction='nearest')
    
    df.loc[df.Label != 'Rückmeldungen', 'TestResQual'] = -1
    df.loc[df.Label != 'Rückmeldungen', 'TestResQuant'] = -1
    
    for user in df['User_id'].unique():
        user_res = df.loc[df.User_id == user, 'TestResQuant']
        for shift in range(1, 5):
            match_condition = user_res.eq(user_res.shift(periods=shift))
            user_res[match_condition] = -1
        df.loc[df.User_id == user, 'TestResQuant'] = user_res
        
    df.loc[df.TestResQuant == -1, 'TestResQual'] = -1
        
    df['TestResQual'].fillna(-1, inplace=True)
    df['TestResQuant'].fillna(-1, inplace=True)
    
    df.drop(['Test', 'Time', 'User_id', 'Code'], axis=1, inplace=True)
    
    return df
