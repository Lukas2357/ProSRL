"""Helper functions for the transformer_functions.py module"""

import io
from os import path
import pandas as pd
import yaml
from collections import OrderedDict

from numpy import datetime64

from ..config.constants import DATA_PATH


def get_categories() -> dict:
    """Get and save dictionary of category names mapped to their num. values

    Returns:
        dict: The category dictionary
        
    """
    categories = ['Grundgrößen zur Kraft', 'Mehrkraftsysteme',
                  'Besondere Kräfte', 'Kraft und Translationsbewegungen ',
                  'Kraft und Rotationsbewegungen ', 'Reibungskräfte']
    
    category_map = OrderedDict(zip(range(len(categories)), categories))
    save_map(category_map, 'category')
    
    return category_map


def get_topics() -> dict:
    """Get and save dictionary of topic names mapped to their num. values

    Returns:
        dict: The topic dictionary
        
    """
    topics = ['Kraft und Wirkung', 'Kraft als Vektor', 'Masse',
              'Kraft und Gegenkraft', 'Verschieben von Kräften',
              'Zeichnerische Addition von Kräften', 'Rechnen mit Kräften',
              'Kräftegleichgewicht', 'Zerlegen von Kräften',
              'Vernetzende Übungen', 'Kräfte im dreidimensionalen Raum',
              'Kraft dreidimensional', 'Komponentengleichungen',
              'Gewichtskraft', 'Hangabtriebskraft',
              'Hangabtriebskraft und Auflagekraft',
              'Auflagekraft und Hangabtriebskraft', 'Normalkraft', 'Federkraft',
              'Gravitationskraft', 'Kraftfelder', 'Grundgesetz der Translation',
              'Zentripetalkraft', 'Drehmoment', 'Trägheitsmoment',
              'Grundgesetz der Rotation', 'Hebel', 'Trägheitsmoment allgemein',
              'Haftreibung', 'Gleitreibung', 'Rollreibung',
              'Strömungswiderstand', 'Vernetzendes Wissen']
    
    topic_map = OrderedDict(zip(range(len(topics)), topics))
    save_map(topic_map, 'topic')
    
    return topic_map
    
    
def get_levels() -> dict:
    """Get and save dictionary of level names mapped to their num. values

    Returns:
        dict: The level dictionary
        
    """    
    levels = ['Grundwissen', 'Erweitertes']
    
    level_map = OrderedDict(zip(range(len(levels)), levels))
    save_map(level_map, 'level')
    
    return level_map


def get_learn_types() -> dict:
    """Get and save dictionary of learntype names mapped to their num. values

    Returns:
        dict: The learntype dictionary
        
    """
    learn_types = ['Tests', 'Ü_Info', 'Übungen', 'Kurzaufgaben',
                   'BK_Info', 'Beispiele', 'Übersicht']
    
    learntype_map = OrderedDict(zip(range(len(learn_types)), learn_types))
    save_map(learntype_map, 'learntype')
    
    return learntype_map


def column_map(df: pd.DataFrame, column: str,
               unique_values: list) -> pd.DataFrame:
    """Map a column of categorical values to discrete values

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): The column to map
        unique_values (list): The unique values in the column

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    mapper = dict(zip(unique_values, list(range(len(unique_values)))))
    df[column] = df[column].map(mapper)
    return df


def save_map(mapping: dict, filename=""):
    """Save mapping of df entries to a yaml file

    Args:
        mapping (dict): The mapping to save
        filename (str, optional): The yaml filename. Defaults to "".
        
    """
    filename = path.join(DATA_PATH, filename + '_map.yaml')
    with io.open(filename, 'w', encoding='utf8') as file:
        yaml.dump(mapping, file, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def correct_columns_for_merge(df: pd.DataFrame, results: pd.DataFrame) -> tuple:
    """Prepare columns for merge of activity df with task/test result dfs

    Args:
        df (pd.DataFrame): The activity df
        results (pd.DataFrame): The task/test results df

    Returns:
        tuple: Both dfs corrected properly
    
    """
    df = df[df["User"].str.contains('HE')]
    results = results[results["User"].str.contains('HE')]
    results['User_id'] = results["User"].str[-3:].astype(int)
    df['User_id'] = df["User"].str[-3:].astype(int)

    results['Time'] = results['Date/Time'].astype(datetime64)
    results['Time'] = results['Time'] - pd.Timedelta(seconds=3)
    df['Time'] = df['Date/Time'].astype(datetime64)

    results['Time'] = pd.to_datetime(results["Time"].dt.strftime('%H:%M:%S'))
    df['Time'] = pd.to_datetime(df["Time"].dt.strftime('%H:%M:%S'))

    results.sort_values(by='Time', inplace=True)
    df.sort_values(by='Time', inplace=True)

    return df, results
