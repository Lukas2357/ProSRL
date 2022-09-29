"""Helper functions for the preparer_fcts.py module"""
import os
from collections import Counter
from itertools import product, repeat
from matplotlib import pyplot as plt, colors
import numpy as np
import pandas as pd
import regex as re
from numpy.typing import ArrayLike

from ..cluster.cluster_plot_help_fcts import save_figure
from ..organizer.orga_fcts import load_learn_pages


def get_duplicates(df: pd.DataFrame) -> pd.Series:
    """Get number of duplicated labels for each user

    Args:
        df (pd.DataFrame): The transformed dataframe

    Returns:
        pd.Series: The number of duplicated labels for each user

    """
    duplicates = df.groupby(['User', 'Label']).size().reset_index()

    duplicates = duplicates[duplicates[0] > 1].drop('Label', axis=1)
    duplicates[0] = duplicates[0] - 1

    duplicates = duplicates.groupby('User').sum()
    duplicates = duplicates.reindex(list(range(df['User'].nunique())),
                                    fill_value=0)

    return duplicates


def get_user_dfs(df: pd.DataFrame) -> list:
    """Get a list of dataframes for each user

    Args:
        df (pd.Dataframe): The df corresponding to data_trafo

    Returns:
        pd.DataFrame: List of single df for each user

    """
    user_dfs = []
    for user in df["User"].unique():
        user_dfs.append(df[df["User"] == user].reset_index())

    return user_dfs


def get_learn_runs(learn_types: pd.Series, times: pd.Series) -> pd.DataFrame:
    """Get a list with length of all runs for the given learn types of a user

    Args:
        learn_types (pd.Series): Series of learn activities of a user
        times (pd.Series): Series of activity times of a user

    Returns:
        pd.DataFrame: Learn runs for the user

    """
    learn_types, times = list(learn_types), list(times)

    runs = []
    current_learn_type = learn_types[0]
    run_length = 1
    run_time = times[0]

    for idx, i in enumerate(learn_types[1:]):
        if i == current_learn_type:
            run_length += 1
            run_time += times[idx + 1]
        else:
            runs.append((run_length, run_time))
            current_learn_type = i
            run_length = 1
            run_time = times[idx + 1]
    runs.append((run_length, run_time))

    return pd.DataFrame(runs, columns=['acts', 'time'])


def predict_chance(arr: any, known=1) -> float:
    """Calculate the chance to predict the next value in an array

    Args:
        arr (any): Any iterable object
        known (int): The number of values known to predict the next

    Returns:
        float: The chance that we predict the right next value if we are given 
                a random value in the array
    
    """
    count = {}
    for idx, i in enumerate(arr[:-known]):
        given = tuple(arr[k] for k in range(idx, idx+known))
        if given in count.keys():
            count[given].append(arr[idx+known])
        else:
            count[given] = [arr[idx+known]]
    result = 0
    for _, value in count.items():
        result += max(dict(Counter(value)).values())
        
    return result/(len(arr)-known)


def _entropy(p_i: list) -> float:
    """Get the entropy for probabilities p_i
    
    Args:
        p_i (list): List of probabilities for outcomes of a random variable

    Returns:
        float: The entropy of the distribution
    
    """
    if len(p_i) == 0:
        return 0
    if isinstance(p_i[0], float):
        n_variables = len(p_i)
    else:
        n_variables = len(p_i[0])
        
    p_i = np.array(p_i)
    p_i = p_i[p_i != 0]
    p_i = p_i[p_i != 1]
    
    if len(p_i) < 2:
        return 0
    
    return np.sum(-p_i*np.log(p_i))/np.log(n_variables)


def cond_entropy(arr: np.array, distance=0) -> float:
    """Get the conditional entropy for an array and a given distance
    
    Args:
        arr (np.array): An array to get conditional entropies for
        distance (int): The number of points known 

    Returns:
        float: The conditional entropy of the array
    
    """
    unique, counts = np.unique(arr, return_counts=True)
    
    if distance == 0:
        return _entropy(counts / len(arr))
    
    cond_entr = 0
    
    for y in product(*repeat(unique, distance)):
        
        arr = list(arr)
        
        cond_ps = []
        y_str = ', '.join([str(i) for i in y])
        y_occ = len(re.findall(fr'{y_str}', repr(arr), overlapped=True))
        
        for x in unique:
            yx_occ = len(re.findall(fr'{y_str}, {x}', repr(arr), 
                                    overlapped=True))
            if y_occ:
                cond_ps.append(yx_occ/y_occ)
                
        tot_prob = sum(cond_ps)
        
        if tot_prob:
            cond_ps = [c/tot_prob for c in cond_ps]
            
        cond_entr += y_occ * _entropy(cond_ps) / len(arr)
        
    return cond_entr


def get_type_entropy(sum_df: pd.DataFrame, f_type='learntype',
                     learntypes=None) -> list:
    """Get the entropy associated with specific feature types for each user

    Args:
        sum_df (pd.DataFrame): The summarized df
        f_type (str): The feature type to use
        learntypes (tuple, optional): The learntypes to consider for entropy.

    Returns:
        list: The list of entropies for each user
        
    """
    if f_type == 'learntype':
        if learntypes is None:
            columns = ('Tests', 'Ü+ÜI', 'Beispiele', 'Kurzaufgaben', 'BK_Info')
        else:
            columns = learntypes
    elif f_type == 'cat':
        columns = [f'Cat_{i}' for i in range(6)]
    else:
        raise AttributeError('Not implemented for this feature type')
        
    type_entropy = []
    for user in range(len(sum_df.index)):
        arr = np.array(sum_df[sum_df['User'] == user][list(columns)])
        c_entropy = _entropy(arr / sum(sum(arr)))
        type_entropy.append(c_entropy*100)
        
    return type_entropy
    
    
def get_before_after_test_times(user_dfs: list) -> pd.DataFrame:
    """Get the times before and after each test averaged over categories

    Args:
        user_dfs (list): The list of user dfs

    Returns:
        dict: Dictionary of results of new features
        
    """
    new_features = {'BeforeTestTime': [], 'AfterTestTime': [], 'TimePerCat': []}

    for user_df in user_dfs:
        
        result = pd.DataFrame()
        cat_min = pd.DataFrame()
        user_df = user_df[['LearnType', 'Category', 'SecSpent']]
        
        for cat in user_df['Category'].unique():
            
            cat_df = user_df[user_df['Category'] == cat].reset_index()
            
            tests_idx = cat_df[cat_df['LearnType'] == 0].index
            cat_minutes = cat_df['SecSpent'].sum()
            cat_min[f'cat_{cat}'] = [cat_minutes]
            
            if len(tests_idx) > 0:
                mi = min(tests_idx) if len(tests_idx) > 0 else len(cat_df.index) 
                
                parts = cat_df.iloc[:mi], cat_df.iloc[mi:]
                part_minutes = [part[part['LearnType'] > 0]['SecSpent'].sum()
                                for part in parts]
                result[f'cat_{cat}'] = part_minutes
            
        result = result.mean(axis=1)
        cat_min = cat_min.mean(axis=1)
        for idx, key in enumerate(new_features.keys()):
            if idx == 2:
                new_features[key].append(cat_min.iloc[0])
            else:
                if len(result.index) > 0:
                    new_features[key].append(result.iloc[idx])
                else:
                    new_features[key].append(0)
        
    return pd.DataFrame(new_features)


def freq_features(user_dfs: list, use_time=False, used_learntypes=None,
                  merge=False) -> pd.DataFrame:
    """Get the features about the frequency of learn task changes

    Args:
        user_dfs (list): List of df for each user
        use_time (bool): Whether to use the time spent as weights
        used_learntypes (any): Learntypes included in the analysis
        merge (bool) Whether to merge K+B+BK_Info and Ü+Ü_Info

    Returns:
        pd.DataFrame: The df of frequency features for each user

    """
    pd.options.mode.chained_assignment = None
    largest_runs, mean_run_length = [], []
    used = [0, 2, 3, 5] if used_learntypes is None else used_learntypes

    for user_df in user_dfs:
        user_df = user_df[user_df.LearnType.isin(used)]
        if merge:
            mapping = dict(zip(range(7), [0, 1, 1, 2, 2, 2, 2]))
            user_df['LearnType'] = user_df['LearnType'].map(mapping)

        runs = get_learn_runs(user_df["LearnType"], user_df['SecSpent'])
        runs['count'] = runs['acts']

        grouped_runs = runs[['count', 'acts', 'time']].groupby('acts').sum()
        col = 'time' if use_time else 'count'

        largest_two = grouped_runs[col].sort_values(ascending=False).iloc[:2]
        length_norm = [item / sum(grouped_runs[col]) for item in largest_two]
        largest_runs.append(tuple(zip(largest_two.index, length_norm)))
        mean_run_length.append(runs[col].mean())

    run_data = zip(mean_run_length, largest_runs)
    features = []
    for mean, f in run_data:
        if len(f) == 1:
            f = f + f
        features.append((mean, f[0][0], f[1][0],
                         (f[0][0] + f[1][0]) / 2,
                         f[0][1] * 100, f[1][1] * 100,
                         f[0][1] * 100 + f[1][1] * 100))

    labels = ["fMean", "f1", "f2", "f12", "f1Amp", "f2Amp", "f12Amp"]
    return pd.DataFrame(features, columns=labels)


def time_dependent_features(user_dfs: list, labels: list, f_typ='LearnType',
                            mapping=None, split_cats=False, plot=False,
                            n_min=3, dpi=200, save=False, n_ranges=2,
                            path='', threshold=2) -> pd.DataFrame | list:
    """Get time dependent features for each user by temporal regression

    Args:
        user_dfs (list): The dfs with activities of each user
        labels (list): The labels for the new features generated
        f_typ (str, optional): The feature type used. Defaults to 'learntype'.
        mapping (list, optional): The learntype mapping used. Defaults to None.
        split_cats (bool, optional): Whether to split categories and get the 
                                     mean. Defaults to False.
        plot (bool, optional): Whether to generate plots. Defaults to False.
        n_min (int, optional): Minimum points used in regression. Defaults to 3.
        dpi (int): Dots per inch for saving the figure
        save (bool): Whether to save the figure
        n_ranges (int): The number of ranges to split time axis in
        path (string): Path to save the figure
        threshold (int): The threshold for the number of points needed for fit

    Returns:
        pd.DataFrame | list: The df with time dependent features of each user
                             if plot is false, else the list of generated plots
        
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    n_cats = 6 if split_cats else 1
    fit_param = {f'{label} Fit': [] for label in labels}
    if len(labels) > 1 and mapping is None:
        mapping = list(range(len(labels)))

    figures, n_artists = [], 1
    fig, ax = None, None

    for user_df in user_dfs:
        
        user_df = user_df[user_df.SecSpent < 1800]
        user_df['UserCumSec'] = user_df['SecSpent'].cumsum()
        user_df = user_df[user_df[f_typ] >= 0]

        if mapping is not None and len(labels) > 1:
            user_df = user_df[user_df[f_typ] < len(mapping)]
            mapper = dict(zip(range(len(mapping)), mapping))
            user_df[f_typ] = user_df[f_typ].map(mapper)
        else:
            mapping = range(len(labels))

        if plot and n_artists > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3.5), dpi=dpi)

        y_max, n_artists = 0, 0
        if len(labels) == 1:
            x = np.array(user_df.UserCumSec) / 60
            y = np.array(user_df[f_typ])
            if len(y) > 0:
                y_max = max(y_max, max(y))
            x_ranges, y_ranges = get_ranges(x, n_ranges, y)
            if len(x) > 0:
                a, w = add_regression(x, x, y, y, ax, [], [], plot, n_min,
                                      x_ranges, y_ranges,
                                      np.array([0]*(n_ranges+1)), 'k',
                                      labels[0])
                n_artists += 1
            else:
                a = [0]
            fit_param[f'{labels[0]} Fit'].append(a[0]*100)

        else:
            values = dict(Counter(mapping)).keys()
            prev_range_data = np.zeros(n_ranges+1)

            for idx, (label, value) in enumerate(zip(labels, values)):

                a, w = [], []
                color = list(colors.TABLEAU_COLORS.values())[idx]

                for cat in range(n_cats):

                    data = user_df[user_df.Category == cat] if split_cats \
                        else user_df
                    data['UserCumFeatureSec'] = data['SecSpent'].cumsum()

                    if len(data.index) > threshold*len(labels):

                        data[label] = user_df['SecSpent']
                        data.loc[user_df[f_typ] != value, label] = 0
                        label_actions = data.index[data[label] > 0]
                        data[label] = data[label].cumsum()

                        x = np.array(data.UserCumSec) / 60
                        y = np.array(data[label])

                        norm = np.array(data['UserCumFeatureSec'])
                        x_ranges, y_ranges = get_ranges(x, n_ranges, y, norm)

                        y_line, y_marker = y / norm, data[label] / norm
                        x_marker = data.UserCumSec.loc[label_actions] / 60
                        y_marker = np.array(y_marker.loc[label_actions])

                        if len(y_line) > 0:
                            y_max = max(y_max, max(y_line))

                        if len(x) > 0:
                            a, w = add_regression(x, x_marker, y_line,
                                                  y_marker, ax, a, w,
                                                  plot, n_min, x_ranges,
                                                  y_ranges, prev_range_data,
                                                  color, labels[idx])
                            n_artists += 1
                        else:
                            a.append(0)
                            w.append(0)
                        prev_range_data += y_ranges

                a_mean = 100*np.average(a, weights=w) if a else 0

                fit_param[f'{label} Fit'].append(a_mean)
            
        if plot:
            plt.xlabel('Zeit des Users auf betrachteten Seiten / min')
            if len(labels) > 1:
                plt.ylabel('Anteil an allen bisherigen Aktionen')
            if len(user_df['UserCumSec']) > 0:
                if max(user_df['UserCumSec']) > 0:
                    plt.xlim([0, max(user_df['UserCumSec'])/60])
                if y_max > 0:
                    plt.ylim([-0.02, y_max*1.02])
            if n_artists > 0:
                plt.legend(ncol=len(labels))
                user_id = str(min(user_df.User))
                save_figure(save=save, path=os.path.join(path, user_id),
                            dpi=dpi, fig=fig, recent=False)
                figures.append((fig, ax))

    if plot:
        return figures
            
    return pd.DataFrame(fit_param)


def get_ranges(x: ArrayLike, n_ranges: int, y: ArrayLike,
               norm: ArrayLike = None) -> tuple[np.array, np.array]:
    """Split x and y points into n ranges of same x size

    Args:
        x (ArrayLike): The x values to use
        n_ranges (int): The number of ranges to use
        y (ArrayLike): The y values to use
        norm (ArrayLike): The normalization values to use

    """
    x_ranges, y_ranges, norm_ranges, k, prev_i = [], [], [], 0, 0
    for i, val in enumerate(x):
        if val >= k * max(x) / n_ranges:
            x_ranges.append(k * max(x) / n_ranges)
            y_correction = 0 if i == 0 else y_ranges[-1]
            if norm is not None:
                norm_correction = 0 if i == 0 else norm_ranges[-1]
                norm_ranges.append(norm[i] - norm_correction)
                y_ranges.append(y[i] - y_correction)
            else:
                if i == 0:
                    y_ranges.append(0)
                else:
                    y_ranges.append(np.mean(y[prev_i:i+1]))
            prev_i = i
            k += 1
        if k > n_ranges:
            break
    l_diff = n_ranges + 1 - len(x_ranges)
    if l_diff > 0:
        if len(x_ranges) > 0:
            x_ranges.extend([x[-1]]*l_diff)
            y_ranges.extend([y[-1]]*l_diff)
            if norm is not None:
                norm_ranges.extend([norm[-1]]*l_diff)

    if norm is None:
        return np.array(x_ranges), np.array(y_ranges)
    else:
        return np.array(x_ranges), np.array(y_ranges)/np.array(norm_ranges)


def add_regression(x_line: np.array, x_marker: np.array, y_line: np.array,
                   y_marker: np.array, axes: plt.Axes, a: list, weights: list,
                   plot: bool, n_min: int, x_ranges: np.array,
                   y_ranges: np.array, prev_range_data: np.array, color: any,
                   label: str) -> tuple:
    """Add a linear regression result and plot to a given list of results

    Args:
        x_line (np.array): x raw_data to plot as line
        x_marker (np.array): x raw_data to plot with markers
        y_line (np.array): y raw_data to plot as line
        y_marker (np.array): y raw_data to plot with markers
        axes (plt.Axes): The axis handle
        a (list): List of slope parameters from linear regressions
        weights (list): List of weights for each linear regression result
        plot (bool): Whether to show the regression plot
        n_min (int): Minimum number of raw_data points required for regression
        prev_range_data (np.array): Previous mean raw_data of different ranges
        x_ranges (np.array): Borders of different ranges
        y_ranges (np.array): Mean raw_data of different ranges
        color (any): The color to use for plotting
        label (str): Label for the legend

    Returns:
        tuple: The new axis handle and the extended result lists
        
    """

    x_line = np.concatenate([[0], x_line])
    y_line = np.concatenate([[y_line[0]], y_line])

    reg = np.polyfit(x_line, y_line, 1) if len(x_line) >= n_min else [0, 0]

    if plot:
        axes.plot(x_line, y_line, ':', color=color, ms=8,
                  label=label + ' Anteil')
        axes.plot(x_marker, y_marker, '.', color=color, ms=8,
                  label=label + ' Actions')
        # axes.plot(x, np.polyval(reg, x), '--', color=color, label='_')
        for i in range(1, len(y_ranges)):
            y_min = prev_range_data[i]
            if color == 'k':
                y_min = y_min / max(y_line) if max(y_line) > 0 else y_min
            y_max = (y_ranges[i]+prev_range_data[i])
            if color == 'k':
                y_max = y_max / max(y_line) if max(y_line) > 0 else y_max
            axes.axvspan(
                x_ranges[i-1], x_ranges[i], y_min, y_max,
                alpha=0.35, color=color, label='_'
            )

    a.append(reg[-2])
    weights.append(len(x_line))
        
    return a, weights


def add_divided_columns(df, cols, div_col, perc=False, postfix='PerX'):
    """Add columns to df generated by division by another column

    Args:
        df (pd.DataFrame): The df to use
        cols (Iterable[str]): The columns to divide
        div_col (Iterable(str) | str): The column(s) to divide by.
            If str, all divided by that, else expect len(div_col) == len(cols)
        perc (bool): Whether to make percentage (* by 100), defaults to False
        postfix (str): The postfix to add to new column names, defaults to PerX

    Returns:
        pd.DataFrame: The concatenated df with new columns

    """
    if isinstance(div_col, str):
        df_div_col = df[cols].div(df[div_col], axis=0)
    else:
        df_cols = df[cols]
        df_cols.columns = div_col
        df_div_col = df_cols.div(df[div_col])
    df_div_col = df_div_col * 100 if perc else df_div_col
    div_col = div_col if isinstance(div_col, str) else div_col[0]

    if postfix == 'PerX':
        if perc:
            postfix = 'Perc'
        elif 'Acts' in div_col:
            postfix = '_TimePerAct'
    elif postfix == '':
        df = df.drop(cols, axis=1)

    df_div_col.columns = [c + postfix for c in cols]
    df = pd.concat([df, df_div_col], axis=1)

    return df


def get_fool_features(user_dfs: list, plot=False) -> pd.DataFrame:
    """Get features indicating if user stupidly follows learning environment

    Args:
        user_dfs (list): The dfs with activities of each user
        plot (bool): Whether to plot the features

    """
    pages = load_learn_pages("learn_pages_sorted")
    features = pd.DataFrame()

    for user_df in user_dfs:

        line = []
        for label in user_df.Label:
            if label in pages.keys() and pages[label]["Title"]:
                line.append(list(pages.keys()).index(label))

        if plot:
            plot_lines(user_df, pages, line)
            plt.show()
            plt.close()

        n = len(line)
        back_frac = sum(line[i] > line[i + 1] for i in range(n - 1)) / n
        fool_frac_1 = sum(line[i] + 1 == line[i + 1] for i in range(n - 1)) / n
        fool_frac_2 = sum(line[i] + 2 == line[i + 1] for i in range(n - 1)) / n

        c_features = pd.DataFrame([back_frac, fool_frac_1, fool_frac_2,
                                   fool_frac_1 + fool_frac_2]).T
        features = pd.concat([features, c_features], axis=0)

    features.columns = ['backfrac', 'foolfrac1', 'foolfrac2', 'foolfrac12']
    features.index = range(len(features))

    return features


def get_type_idx(pages: dict) -> list:
    """Get indices of learn pages where the Type of pages changes

    Args:
        pages (dict): The dictionary of all sorted learn pages

    Returns:
        list: List of indices where the Type changes (for coloring in plot)

    """
    ltype, type_idx = '', []

    for idx, page in enumerate(pages.values()):
        if page['Type'] != ltype:
            ltype = page['Type']
            type_idx.append(idx)

    type_idx = type_idx + [len(pages)]

    return type_idx


def plot_lines(user_df, pages, line=None, save=False, path='', dpi=100):
    """Plot lines comparing user actions with intuitive order of learnpages

        Args:
            user_df (pd.DataFrame): The user activities df
            pages (dict): A dictionary of all sorted learnpages
            line (list): The line to plot. Defaults to None.
            save (bool): Whether to save the plot. Defaults to False.
            path (str): The path to save the plot. Defaults to ''.
            dpi (int): The dpi of the plot. Defaults to 100.

        Returns:
            list: List of indices where the Type changes (for coloring in plot)

        """
    if line is None:
        line = []
        for label in user_df.Label:
            if label in pages.keys() and pages[label]["Title"]:
                line.append(list(pages.keys()).index(label))

    type_idx = get_type_idx(pages)

    fig, ax = plt.subplots(1, 1, figsize=(12, 2), dpi=dpi)
    ax.plot(line, range(len(line)), "k-")
    for i in range(len(type_idx) - 1):
        ax.axvspan(
            type_idx[i] + 1,
            type_idx[i + 1],
            alpha=i % 3 / 4 + 0.1,
            color=list(colors.TABLEAU_COLORS.values())[i // 3],
        )

    ax.set_xlim([0, len(pages)])
    plt.tick_params(bottom=False, top=False, labelbottom=False, labelleft=True,
                    right=False, left=True)
    ax.set_xlabel('Intuitive Anordnung der Seiten auf der Lernplattform -> '
                  'Farben = Kategorien')
    ax.set_ylabel('User Aktionen')
    plt.gca().invert_yaxis()

    save_figure(save=save, path=path, dpi=dpi, fig=fig, recent=False)

    return fig, ax
