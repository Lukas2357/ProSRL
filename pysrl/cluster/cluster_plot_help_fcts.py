"""Helper functions for the cluster_plot_fcts.py module"""
import difflib
import os
from matplotlib.axes import Axes
import pandas as pd
from matplotlib import pyplot as plt

from ..config.constants import RECENT_RESULTS_PATH
from ..preparer.prep_fcts import load_transformed_data


def init_subplots_plot(n_plots: int, scales=(3.5, 3.2)) -> list:
    """Initialize subplots one for each entry in data_list

    Args:
        n_plots (int): The number of subplots to generate
        scales (tuple): sizes for each row and column in the plot

    Returns:
        list[list[Axis]]: The axis objects for the subplots

    """

    rows = (n_plots - 1) // 3 + 1
    columns = (n_plots - 1) % 3 + 1 if n_plots < 4 or n_plots % 3 == 0 \
        else n_plots % 3 + 1
    while rows * columns < n_plots:
        if columns < 4:
            columns += 1
        else:
            rows += 1
    figsize = (scales[0] * columns, scales[1] * rows)

    _, ax = plt.subplots(rows, columns, figsize=figsize)

    return ax


def get_current_axis(data: list, ax: any, idx: int) -> any:
    """Get the current axis in a potentially multidimensional subplot

    Args:
        data (list): The raw_data list to get the length from
        ax (any): A list or nested list of/or axis object
        idx (int): The index of the current axis

    Returns:
        plt.Axis: The current axis based on idx

    """
    if len(data) == 1:
        c_ax = ax
    elif len(data) == 4:
        c_ax = ax[idx // 2][idx % 2]
    elif len(data) > 4:
        c_ax = ax[idx // 3][idx % 3]
    else:
        c_ax = ax[idx]

    return c_ax


def set_labels(columns: list, x_tag: str, c_ax: Axes, dim=1, 
               y_tag=None) -> Axes:
    """Set the labels of 1D and 2D feature plots with correct units

    Args:
        columns (list): The columns used in the plot
        x_tag (str): The x label without unit
        c_ax (Axes): The axis handle
        dim (int, optional): The dimension of the plot. Defaults to 1.
        y_tag (str, optional): The y tag in case of dim=2. Defaults to None.

    Returns:
        Axes: The axes with proper label
    
    """
    cols = [data.columns[0] for data in columns]
    perc_cols = ['Tests', 'Übungen', 'Kurzaufgaben', 'Beispiele', 'BK_Info',
                 'Ü_Info', 'Übersicht', 'Grund', 'Erweitert', 
                 'PredChance1', 'PredChance2', 'PredChance3', 
                 'PredChance4', 'LearnTypeEntropy', 'CategoryEntropy'] + \
                [f'Cat_{i}' for i in range(6)] + \
                [c for c in cols if 'Amp' in c or 'Perc' in c]

    c_ax.set_xlabel(f"{x_tag}")
    if dim == 2:
        c_ax.set_ylabel(f"{y_tag}")
    if 'Time' in x_tag or x_tag == 'fMean':
        c_ax.set_xlabel(f"{x_tag} / min")
    if x_tag in perc_cols:
        c_ax.set_xlabel(f"{x_tag} / %")
    if dim == 2 and ('Time' in y_tag or y_tag == 'fMean'):
        c_ax.set_ylabel(f"{y_tag} / min")
    if dim == 2 and y_tag in perc_cols:
        c_ax.set_ylabel(f"{y_tag} / %")
    return c_ax


def save_figure(save: bool, dpi: int, path: str, fig=None, tight=True,
                fig_format='png', recent=True):
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
                path = os.path.join(RECENT_RESULTS_PATH,
                                    os.path.split(path)[-1])
                plt.savefig(path, dpi=dpi, format=fig_format)
        else:
            fig.patch.set_facecolor('white')
            plt.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(),
                        format=fig_format)
            if recent:
                path = os.path.join(RECENT_RESULTS_PATH,
                                    os.path.split(path)[-1])
                plt.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(),
                            format=fig_format)


def get_centers_list(centers: pd.DataFrame):
    """Get a list of df of cluster centers for each combination of columns

    Args:
        centers (pd.DataFrame): The df of cluster centers with multiindex

    Returns:
        list: A list of dfs for each combi, effectively resolving the multiindex
    
    """
    centers_dfs = [pd.concat([centers[combi, feature]
                              for feature in combi], axis=1)
                   for combi in set([combi[0]
                                     for combi in centers.columns])]
    return centers_dfs


def get_user_learntype_strings(exclude="G", replace=None) -> dict:
    """Transform the learntype column for each user in a string

    Args:
        exclude (string): LearnTypes to exclude (by letter, see below)
        replace (dict): Letters to replace (possible letters see below)

    Returns:
        dict: key=User, value=String of LearnTypes

    """
    if replace is None:
        replace = {'I': 'U', 'J': 'B'}

    df = load_transformed_data()[["User", "LearnType"]]

    learn_type_labels = ["T", "I", "U", "K", "J", "B", "G"]
    df["LearnType"] = df["LearnType"].map(
        dict(zip(range(7), learn_type_labels))
    )

    user_strings = {}

    for user in df.User.unique():

        temp = "".join(df.loc[df.User == user, "LearnType"])
        temp = "".join([x for x in temp if x not in exclude])

        for key, value in replace.items():
            temp = temp.replace(key, value)

        user_strings[user] = temp

    return user_strings


def get_learntype_string_ratios(user_strings=None, pad='max', exclude="G",
                                replace=None) -> pd.DataFrame:
    """Get similarity ratios for user LearnType strings

    Args:
        user_strings (dict): Strings as returned by get_user_learntype_strings
        pad (string): Min for cutting, max for broadcasting, None for keeping
        exclude (string): exclude parameter passed to get_user_learntype_strings
        replace (dict): replace parameter passed to get_user_learntype_strings

    Returns:
        pd.DataFrame: columns -> user1, user2, string1, string2, sim_ratio

    """

    if user_strings is None:
        user_strings = get_user_learntype_strings(exclude, replace)

    df = pd.DataFrame(
        columns=("user1", "user2", "string1", "string2", "sim_ratio")
    )

    for user1, string1 in user_strings.items():

        for user2, string2 in user_strings.items():

            min_length = min(len(string1), len(string2))
            max_length = max(len(string1), len(string2))

            if min_length == 0:
                ratio = 0

            elif user1 == user2:
                ratio = 0.5

            else:
                if pad == "min":
                    s1, s2 = string1[:min_length], string2[:min_length]
                elif pad == "max":
                    s1 = (string1 * (int(max_length / min_length) + 1))[
                         :max_length
                         ]
                    s2 = (string2 * (int(max_length / min_length) + 1))[
                         :max_length
                         ]
                else:
                    s1, s2 = string1, string2

                ratio = difflib.SequenceMatcher(None, s1, s2).ratio()

            idx = 0 if len(df.index) == 0 else max(df.index) + 1

            df.loc[idx] = user1, user2, string1, string2, ratio

        df = df.sort_values(by="sim_ratio", ascending=False)

    return df
