"""Plot functions prior to analysis (Lineplots, Heatmaps, etc.)"""

import os
from collections import Counter
from typing import Iterable
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px
from plotly.graph_objs.scatter import Line, Marker
from sklearn.decomposition import PCA

from config.helper import load_input
from ..cluster.cluster_plot_help_fcts import save_figure
from ..config.constants import RESULTS_PATH
from .prep_fcts import load_transformed_data
from ..transformer.transform_help_fcts import get_categories, \
    get_learn_types


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

    df = load_transformed_data()
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


def spent_times_histogram(df: pd.DataFrame, x_max=(30, 6, 6, 6)):
    """Histogram of the times spent on a page for each learntype

    Args:
        df (pd.DataFrame): The transformed dataframe
        x_max (list, optional): Maximum time values. Defaults to [30, 6, 6, 6].

    """
    labels = ["Tests", "Übungen", "Kurzaufgaben", "Beispiele"]

    df = df[df.Title > 0]
    df['LearnType'] = df['LearnType'].map(dict(zip(range(4), labels)))

    _, ax = plt.subplots(4, 1, figsize=(16, 12))
    plt.setp(ax, xlabel='Zeit / min')

    for idx, label in enumerate(labels):
        data = df[df.LearnType == label]
        data['MinSpent'] = data['SecSpent'] / 60
        c_x_max = max(data['MinSpent']) if x_max[idx] < 0 else x_max[idx]
        data = data[data['MinSpent'] < c_x_max]
        sns.histplot(ax=ax[idx], data=data, x='MinSpent', bins=40)
        ax[idx].set_xlim([0, c_x_max])
        ax[idx].legend([labels[idx]])

    plt.tight_layout()


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
            labels[agg_learntype[0]] = '+'.join([short_tags[i]
                                                 for i in agg_learntype])

    return df, labels


def plot_pca(pca: PCA, components: pd.DataFrame, clusters: pd.DataFrame,
             center: pd.DataFrame, columns: list, ca=0, cb=1,
             user_ids=None) -> px.scatter:
    """Plot the result of a PCA analysis

    Args:
        pca (PCA): The PCA object from sklearn decomposition
        components (pd.DataFrame): List of PCA components
        clusters (pd.DataFrame): The clusters of the raw_data points
        center (pd.DataFrame): The center of the cluster (only for kmeans)
        columns (list): The columns used as features
        ca (int, optional): The first PC to use. Defaults to 0.
        cb (int, optional): The second PC to use. Defaults to 1.
        user_ids (any, optional): The user ids to annotate in the plot

    Returns:
        px.scatter: The plotly scatter plot handle

    """
    pca_variance = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca_variance)
    loadings = loadings * np.sqrt(len(loadings))
    center = pd.DataFrame(center)

    labels = {str(i + ca): f"PC {i + 1 + cb} ({var:.1f}%)"
              for i, var in enumerate(pca_variance[ca:cb + 1] * 100)}

    pcs = components.columns
    components['size'] = 1
    clusters = clusters[tuple(columns)]

    text = clusters.index if user_ids is None else user_ids
    fig = px.scatter(components, x=pcs[ca], y=pcs[cb], labels=labels,
                     color=[str(c) for c in clusters], text=text,
                     size='size', size_max=15)

    if center is not None and len(center.index) != 0:
        fig.add_scatter(x=center[0], y=center[1], line=Line(width=0),
                        marker=Marker(size=15, color='rgb(0,0,0)', symbol=17,
                                      opacity=0.75))

        fig['raw_data'][-1]['name'] = 'Cluster Zentren'

    for i, feature in enumerate(columns):
        fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i, ca],
                      y1=loadings[i, cb])

        fig.add_annotation(x=loadings[i, ca], y=loadings[i, cb], ax=0, ay=0,
                           yanchor="top" if loadings[i, cb] < 0 else 'bottom',
                           xanchor="center", text=feature)

    fig.update_traces(textfont_size=12)
    fig.update_layout(legend=dict(x=0.7, y=1, bgcolor='rgba(0,0,0,0)',
                                  title=''))

    return fig


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


def get_axvline_params(user_df: pd.DataFrame, day_breaks: pd.DataFrame,
                       hour_breaks: pd.DataFrame) -> list:
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
