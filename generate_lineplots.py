import io
import os
import warnings
from collections import Counter, OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt


ROOT = "/home/lukas/Projects/ProSRL"  # Ordner in dem diese Datei liegt
YEAR = 2021
RESULTS_PATH = os.path.join(ROOT, 'results', str(YEAR))
DATA_PATH = os.path.join(ROOT, 'data', str(YEAR))
RECENT_RESULTS_PATH = os.path.join(ROOT, 'results', 'recent')


def learn_types_lineplots(cluster=pd.Series(dtype=str), save=True, dpi=120,
                          legend=False, titles=tuple(), path='',
                          file_extension='', learntypes=None, max_user=100,
                          selected_user=None, max_time_shown=None,
                          user_ids=None, silent=False, use_codes=True):
    """Subsequent lineplots of learn type runs for each user

    Args:
        cluster (pd.Series): Cluster for each user to annotate in the plot
        save (bool): Whether to save the resulting plot
        dpi (int): Dots per inch for saving the figure
        legend (bool): Whether to show the legend
        titles (Tuple): The titles of the plots'
        path (str): The file path to save the figure
        file_extension (str): Extension to add to the file name
        learntypes (list(list(str)): The learntypes to aggregate (inner list
            elements) and to plot separately (outer list elements)
        max_user (int): The maximum user to show per plot
        selected_user (list): List of users to plot (other params ignored)
        max_time_shown (float): The maximum number of minutes shown in the plot
        user_ids (list): List of user ids to annotate in the plots
        silent (bool): Whether to suppress the plt.show() call
        use_codes (bool): Whether to use user codes in plot labeling (else id)

    """
    pd.options.mode.chained_assignment = None  # default='warn'
    plt.rcParams.update({'font.size': 14})

    if not path:
        path = os.path.join(RESULTS_PATH, 'user_lineplots')

    df = pd.read_csv(os.path.join(DATA_PATH, "data_trafo.csv")).iloc[:, 1:]
    n_user = len(user_ids) if user_ids else len(df.User.unique())
    user_ids = user_ids if user_ids else sorted(df.User.unique())

    learntypes = [[0], [1], [2], [3], [4], [5], [6]] if learntypes is None \
        else learntypes

    df, labels = map_learntype_labels(df, learntypes)

    learntypes = [learntype[0] for learntype in learntypes]

    cluster = cluster if len(cluster.index) != 0 else list(range(n_user))
    c_counts = dict(Counter(cluster))
    c_labels, c_sizes = list(c_counts.keys()), list(c_counts.values())

    categories = list(get_categories().values())

    relations = load_input('relation_user_code.csv')
    relations = relations.set_index('user_id')

    indices = [0] if selected_user else range(min(len(c_sizes), max_user))

    figures, user_id = [], 0

    for c_idx in indices:

        if c_idx not in relations.index:
            continue

        print(c_idx)

        rows = len(selected_user) if selected_user else c_sizes[c_idx]
        fig, ax = plt.subplots(rows, 1, figsize=[20, 2.9 * rows])
        fig.patch.set_facecolor('white')

        c_label = c_labels[c_idx]

        if selected_user:
            c_user_ids = selected_user
        else:
            c_user_idxs = [i for i in range(n_user) if cluster[i] == c_label]
            c_user_ids = [user_ids[i] for i in c_user_idxs]

        for idx, user_id in enumerate(c_user_ids):

            user_df = df[df.User == user_id].reset_index()

            user_df['TotMin'] = (user_df.TotSec - min(user_df.TotSec)) / 60

            user_df, day_breaks, hour_breaks = introduce_breaks(user_df)

            user_df['CumMin'] = user_df['SecSpent'].cumsum() / 60
            user_df['CumMin'] = user_df['CumMin'].shift(1, fill_value=0)

            axvline_params = get_axvline_params(user_df, day_breaks,
                                                hour_breaks)

            user_df = user_df[user_df.LearnType.isin(learntypes)].reset_index()

            c_ax = ax[idx] if isinstance(ax, Iterable) else ax

            legend_entries = [categories[done]
                              for done in user_df["Category"].unique()]

            diffs = sorted(user_df['difficulty'].unique())
            marker_styles = [['v', '>', '^', 'o'][m] for m in diffs]
            sizes = sorted([10 * np.sqrt(x)
                            for x in user_df['SecSpent'].unique()])

            sns.lineplot(ax=c_ax, data=user_df, x='CumMin', y='LearnType',
                         linestyle='-', color='gray', zorder=-1)

            sns.scatterplot(ax=c_ax, data=user_df, x='CumMin', y='LearnType',
                            markers=marker_styles, sizes=sizes, hue='Category',
                            palette='tab10', linewidth=0.7, edgecolor='k',
                            style='difficulty', size='SecSpent')

            if 'TestResQuant' in user_df.columns:
                response = ['-', 'o', '+']
                for i in range(len(user_df.index) - 1):
                    test_res_quant = user_df['TestResQuant'].iloc[i]
                    test_res_qual = user_df['TestResQual'].iloc[i]
                    task_res = user_df['ResponsTask'].iloc[i]
                    if test_res_qual >= 0:
                        text = response[test_res_qual] + f' ({test_res_quant})'
                    elif task_res >= 0:
                        text = response[task_res]
                    else:
                        text = ''

                    lt = user_df['LearnType']
                    if i == 0 or i == len(user_df.index):
                        off = 1
                    elif (lt[i - 1] == lt[i] + 1
                          or lt[i + 1] == lt[i] + 1) and \
                            (lt[i - 1] != lt[i] - 1 and lt[i + 1] != lt[i] - 1):
                        off = - 1
                    else:
                        off = 1
                    pos = [user_df['CumMin'][i], lt[i] + off]
                    c_ax.annotate(text, pos, horizontalalignment='center',
                                  verticalalignment='center', size=10)

            level_diff = user_df['CumMin'][user_df['Level'].diff() != 0]
            level_val = user_df['Level'][user_df['Level'].diff() != 0]
            levels = tuple(zip(level_diff, level_val))
            last = levels[-1]

            cols = ['green', 'darkred']
            c_max_t = user_df['CumMin'].iloc[-1]
            for k, (start, level) in enumerate(levels[:-1]):
                c_ax.axvspan(start, levels[k + 1][0], alpha=0.2,
                             color=cols[level])
            c_ax.axvspan(last[0], c_max_t, alpha=0.2, color=cols[last[1]])
            max_time = 180 if c_max_t < 180 else c_max_t
            max_time = max_time if not legend else max_time + 50
            c_ax.axvspan(c_max_t, max_time + 1, alpha=0.1, color='gray')

            # c_ax.annotate('Tag 1', [day_break_time - 5, 5], size=10)
            # c_ax.annotate('Tag 2', [day_break_time + 1, 5], size=10)
            for axvline_param in axvline_params:
                c_ax.axvline(*axvline_param[:-1], color=axvline_param[-1])

            if max_time_shown is not None:
                max_time = max_time_shown
            plt.setp(c_ax, xlim=[0, max_time + 1], ylim=[-1, 7],
                     xlabel='Zeit / min', yticks=list(range(7)),
                     yticklabels=labels)

            code = relations.loc[user_id, 'Person'] if use_codes else user_id

            y_label = f'{code} - ' + str(c_label) if len(indices) > 1 \
                else f'{code}'
            c_ax.set_ylabel(y_label, size=16, weight='bold')
            if legend:
                c_ax.legend(legend_entries, loc='right')
            else:
                c_ax.legend([], [], frameon=False)
            if len(indices) > 1 and len(titles) > c_idx:
                title = str(c_label) + ' - ' + titles[c_idx] + '\n' * (rows//5)
            elif len(indices) > 1:
                title = str(c_label)
            else:
                title = str('User Lineplots') + '\n' * (rows // 5)
            fig.suptitle(title, size=24, weight='bold')
            plt.tight_layout()

        file = f'{len(c_sizes)}C_' + str(c_label) + file_extension
        if file_extension == 'simple':
            save_path = os.path.join(path, str(user_id))
        else:
            save_path = os.path.join(path, file)
        save_figure(save=save, path=save_path, dpi=dpi, fig=fig)

        figures.append((fig, ax))

        if not silent:
            plt.show()

    return figures


def introduce_breaks(user_df: pd.DataFrame) -> tuple:
    """Get day and hour breaks for the user and correct them to 240sec

    Args:
        user_df (pd.DataFrame): The df of user activity

    Returns:
        pd.DataFrame: Corrected df, day break list and hour break list

    """
    day_breaks = user_df[user_df.SecSpent > 60 * 60 * 6]
    hour_breaks = user_df[user_df.SecSpent.between(60 * 60, 60 * 60 * 6)]

    for _, learn_breaks in enumerate([day_breaks, hour_breaks]):
        if len(learn_breaks.index) > 0:
            for _, day_break in learn_breaks.iterrows():
                learn_break_time = day_break.TotMin
                user_df.loc[user_df.TotMin == learn_break_time,
                            'SecSpent'] = 240

    return user_df, day_breaks, hour_breaks


def get_axvline_params(
        user_df: pd.DataFrame, day_breaks: pd.DataFrame,
        hour_breaks: pd.DataFrame
        ) -> list:
    """Get parameters for vertical lines of day and hour breaks

    Args:
        user_df (pd.DataFrame): The user activity df
        day_breaks (pd.DataFrame): df of activities corresponding to day breaks
        hour_breaks (pd.DataFrame): df of activities corresp. to hour breaks

    Returns:
        list: Parameter list to be put in axvline function

    """
    vcs = ['black', 'red']
    axvline_params = []

    for i, break_type in enumerate((day_breaks, hour_breaks)):
        for _, br in break_type.iterrows():
            for d in [1.85, 2.15]:
                pos = user_df[user_df['index'] == br['index']]
                pos = pos['CumMin'].iloc[0]
                axvline_params.append([pos + d, -0.5, 6.5, vcs[i]])

    return axvline_params


def map_learntype_labels(df: pd.DataFrame, learntypes: list) -> tuple:
    """Map learntype labels in case of aggregated learntypes

    Args:
        df (pd.DataFrame): The df to show in lineplots
        learntypes (list): Nested list to aggregate learntypes

    Returns:
        tuple: The df mapped to aggregated learntypes and corresponding labels

    """
    learntype_tags = list(get_learn_types().values())
    learntype_tags[3] = 'Kurzaufg.'
    short_tags = []
    for tag in learntype_tags:
        short_tag = tag[:-3].replace('_', '') if '_' in tag else tag[0]
        short_tags.append(short_tag)

    labels = ['' for _ in range(7)]
    for agg_learntype in learntypes:
        df.loc[df.LearnType.isin(agg_learntype), 'LearnType'] = agg_learntype[0]
        if len(agg_learntype) == 1:
            labels[agg_learntype[0]] = learntype_tags[agg_learntype[0]]
        else:
            labels[agg_learntype[0]] = '+'.join(
                [short_tags[i]
                 for i in agg_learntype]
                )

    return df, labels


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


def save_map(mapping: dict, filename=""):
    """Save mapping of df entries to a yaml file

    Args:
        mapping (dict): The mapping to save
        filename (str, optional): The yaml filename. Defaults to "".

    """
    filename = os.path.join(DATA_PATH, filename + '_map.yaml')
    with io.open(filename, 'w', encoding='utf8') as file:
        yaml.dump(
            mapping, file, default_flow_style=False,
            allow_unicode=True, sort_keys=False
            )


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

    input_path = os.path.join(ROOT, "input", str(year))
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


def save_figure(
        save: bool, dpi: int, path: str, fig=None, tight=True,
        fig_format='png', recent=True
):
    """Save figures to proper location

    Args:
        save (bool): Whether to do the saving
        dpi (int): The dpi used in the figure
        path (str): The path where to save it
        fig (plt.Figure): The figure handle
        tight (bool): Whether to save in tight layout
        fig_format (str): The file format to use
        recent (bool): Whether to save the figure in recent folder

    """
    if save:

        path = path + '.' + fig_format

        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(RECENT_RESULTS_PATH, exist_ok=True)

        if tight:
            plt.tight_layout()
        if fig is None:
            plt.savefig(path, dpi=dpi, format=fig_format)
            if recent:
                path = os.path.join(
                    RECENT_RESULTS_PATH,
                    os.path.split(path)[-1]
                )
                plt.savefig(path, dpi=dpi, format=fig_format)
        else:
            fig.patch.set_facecolor('white')
            plt.savefig(
                path, dpi=dpi, facecolor=fig.get_facecolor(),
                format=fig_format
            )
            if recent:
                path = os.path.join(
                    RECENT_RESULTS_PATH,
                    os.path.split(path)[-1]
                )
                plt.savefig(
                    path, dpi=dpi, facecolor=fig.get_facecolor(),
                    format=fig_format
                )


learn_types_lineplots()