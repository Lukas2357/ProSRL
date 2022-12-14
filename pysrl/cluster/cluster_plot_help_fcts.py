"""Helper functions for the cluster_plot_fcts.py module"""
import difflib
import os
from typing import Callable

from matplotlib.axes import Axes
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

from ..config.constants import RECENT_RESULTS_PATH
from ..config.helper import load_data


def init_subplots_plot(n_plots: int, scales=(3.5, 3.2), n_col=3) -> list:
    """Initialize subplots one for each entry in data_list

    Args:
        n_plots (int): The number of subplots to generate
        scales (tuple): sizes for each row and column in the plot
        n_col (int): The number of columns

    Returns:
        list[list[Axis]]: The axis objects for the subplots

    """

    rows = (n_plots - 1) // n_col + 1
    columns = (n_plots - 1) % n_col + 1 if n_plots < n_col + 1 or n_plots % n_col == 0 \
        else n_plots % n_col + 1
    while rows * columns < n_plots:
        if columns < n_col + 1:
            columns += 1
        else:
            rows += 1
    figsize = (scales[0] * columns, scales[1] * rows)

    _, ax = plt.subplots(rows, columns, figsize=figsize)

    return ax


def get_current_axis(data: list, ax: any, idx: int, n_col=3) -> any:
    """Get the current axis in a potentially multidimensional subplot

    Args:
        data (list): The raw_data list to get the length from
        ax (any): A list or nested list of/or axis object
        idx (int): The index of the current axis
        n_col (int): The number of columns

    Returns:
        plt.Axis: The current axis based on idx

    """
    if len(data) == 1:
        c_ax = ax
    elif len(data) == n_col + 1:
        c_ax = ax[idx // (n_col - 1)][idx % n_col - 1]
    elif len(data) > n_col + 1:
        c_ax = ax[idx // n_col][idx % n_col]
    else:
        c_ax = ax[idx]

    return c_ax


def set_labels(
        columns: list, x_tag: str, c_ax: Axes, dim=1,
        y_tag=None
        ) -> Axes:
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
    perc_cols = ['Tests', '??bungen', 'Kurzaufgaben', 'Beispiele', 'BK_Info',
                 '??_Info', '??bersicht', 'Grund', 'Erweitert',
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


def get_centers_list(centers: pd.DataFrame):
    """Get a list of df of cluster centers for each combination of columns

    Args:
        centers (pd.DataFrame): The df of cluster centers with multiindex

    Returns:
        list: A list of dfs for each combi, effectively resolving the multiindex
    
    """
    centers_dfs = [pd.concat(
        [centers[combi, feature]
         for feature in combi], axis=1
        )
                   for combi in set(
            [combi[0]
             for combi in centers.columns]
            )]
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

    df = load_data('data_trafo.csv')[["User", "LearnType"]]

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


def get_learntype_string_ratios(
        user_strings=None, pad='max', exclude="G",
        replace=None
        ) -> pd.DataFrame:
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


def cor_test(method='pearson') -> Callable:
    """Get function for p-value of correlation test with method specified

    Args:
        method (str): Corr method to test

    Returns:
        Callable: Function to get two-sided p-value for correlation != 0

    """
    if method == 'pearsonr':
        def pvalue(x, y):
            return pearsonr(x, y)[1]
    else:
        def pvalue(x, y):
            return spearmanr(x, y)[1]

    return pvalue


def get_significant_correlations(
        method, pearson, p_pearson, spearman,
        p_spearman, threshold, min_corr,
        max_corr=1
        ):
    """Get a df with significant correlations and corresponding p-values

    Args:
        method (str): Corr method to test
        p_pearson (pd.DataFrame): pearson p-values to check for significance
        pearson (pd.DataFrame): corresponding correlation values
        p_spearman (pd.DataFrame): spearman p-values to check for significance
        spearman (pd.DataFrame): corresponding correlation values
        threshold (float): The threshold p-value to consider significant
        min_corr (float): The minimum correlation to check
        max_corr (float): The maximum correlation to check


    Returns:
        Tuple: Df of significant correlations with columns of
                - first feature
                - second feature
                - correlation coefficient
                - two-sided p-value against r=0
                - priority category of feature 1
                - priority category of feature 2
               and copy of that df with the first two columns set to the last
               two to get a numeric df that can be plotted in a heatmap

    """
    prio = load_data(
        'features_categories.csv',
        rm_first_col=False
        ).set_index('Feature')
    label = 'p-value / %'
    cols = ['Feature 1', 'Feature 2', 'pearson', label, 'spearman', label,
            'Prio 1', 'Prio 2']
    significant = set()
    corrs = spearman if method == 'spearman' else pearson
    pvalues = p_spearman if method == 'spearman' else p_pearson

    for col in pvalues.columns:
        for idx, p in enumerate(pvalues[col]):
            row = pvalues.index[idx]
            corr = corrs.loc[pvalues.index[idx], col]
            prio1 = prio.loc[col, 'Prio'] if col in prio.index else 3
            prio2 = prio.loc[row, 'Prio'] if row in prio.index else 3
            if p < threshold and min_corr <= abs(corr) <= max_corr:
                c_pearson = pearson.loc[pvalues.index[idx], col]
                c_spearman = spearman.loc[pvalues.index[idx], col]
                c_pearson_p = p_pearson.loc[pvalues.index[idx], col]
                c_spearman_p = p_spearman.loc[pvalues.index[idx], col]

                entry = c_pearson, c_pearson_p, c_spearman, c_spearman_p
                if (row, col) + entry + (prio2, prio1) not in significant:
                    significant.add((col, row) + entry + (prio1, prio2))

    significant = pd.DataFrame(significant, columns=cols, dtype=float)
    significant = significant.sort_values(method, ascending=False)
    significant = significant.reset_index().drop('index', axis=1)
    numeric = significant.copy()
    numeric['Feature 1'] = numeric['Prio 1']
    numeric['Feature 2'] = numeric['Prio 2']

    return significant, numeric