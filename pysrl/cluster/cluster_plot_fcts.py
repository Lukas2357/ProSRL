"""Functions for the cluster_plot.py module"""

import warnings
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from .cluster_plot_help_fcts import *
from .cluster_fcts import load_prep_data
from ..config.constants import RESULTS_PATH

pd.options.mode.chained_assignment = None  # default='warn'


def correlation_heatmap(features=None, df=None, save=False, dpi=300, 
                        filename='cor_heatmap', cbar=False,
                        all_with_these_features=False):
    """Heatmap of all correlations of the prepared dataframe

    Args:
        features (list): The features to which all correlations are shown
        df (any): The raw_data to plot, if None it will be loaded from raw_data prep
        save (bool): Whether to save the plot
        dpi (int): Dots per inch to save the plot
        filename (str): The filename to save the plot
        cbar (bool): Whether to show the colorbar
        all_with_these_features (bool): Show all cors of given features

    """    
    if df is None:
        df = load_prep_data()
        
    df = df.drop(['User'], axis=1)
               
    if features is None:
        features = df.columns
    
    if isinstance(features, str):
        features = [features]

    if not all_with_these_features:
        df = df[features]
        
    fig = plt.figure(figsize=(len(features), len(df.columns)/4+2))
    sns.heatmap(df.corr()[features], annot=True, fmt='.3f', vmin=-1, vmax=1,
                cbar=cbar)
    plt.suptitle('Feature correlation heatmap', fontweight='bold')

    path = os.path.join(RESULTS_PATH, 'heatmaps', filename)
    save_figure(save, dpi, path, fig)
    
    plt.show()


def user_heatmap(user=None, features=None, df=None, save=False, dpi=300, 
                 filename='user_heatmap', cbar=False):
    """Heatmap of all features for given set of users

    Args:
        user (list): The users to show
        features (list): The features to show
        df (any): The raw_data to plot, if None it will be loaded from raw_data prep
        save (bool): Whether to save the plot
        dpi (int): Dots per inch to save the plot
        filename (str): The filename to save the plot
        cbar (bool): Whether to show the colorbar

    """
    if isinstance(user, int):
        user = [user]
        
    if isinstance(features, int):
        features = [features]
        
    if df is None:
        df = load_prep_data()
        
    scaled_df = df.drop(['User'], axis=1)
    scaled_df = StandardScaler().fit_transform(scaled_df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns[1:])
    
    if user is not None:
        df = df[df['User'].isin(user)]
        scaled_df = scaled_df[df['User'].isin(user)]
        
    if features is not None:
        df = df[features]
        scaled_df = scaled_df[features]
    else:
        df = df.drop(['User'], axis=1)
    
    if len(df.columns) > len(df.index):
        df = df.transpose()
        scaled_df = scaled_df.transpose()
    
    rows = max(len(df.columns), len(df.index))
    cols = min(len(df.columns), len(df.index))
    fig = plt.figure(figsize=(cols, rows/4+2))
    sns.heatmap(scaled_df, annot=df, fmt='.3f', vmin=-3, vmax=3, cbar=cbar)
    plt.suptitle('User feature heatmap', fontweight='bold')
    
    path = os.path.join(RESULTS_PATH, 'heatmaps', filename)
    save_figure(save, dpi, path, fig)
    
    plt.show()


def cluster_pairs_plot(pairs: list[pd.DataFrame], labels: pd.DataFrame,
                       abline=False, save=True, path="", dpi=300,
                       user_ids=None, show_cluster_of=None) -> list:
    """Plot the kmeans clustering result for a pair of features

    Args:
        pairs (list[pd.DataFrame]): The input dataframes to be plotted
        labels (pd.DataFrame): Df with labels for each kmeans result
        abline (bool): Whether to show bisector line in the plot
        save (bool): Whether to save the figure
        path (string): Path to save the figure
        dpi (int): Dots per inch for saving the figure
        user_ids (any): User ids to annotate in the plot, uses index if None
        show_cluster_of (list): Feature combi whose clusters shown in all plots

    """
    ax = init_subplots_plot(len(pairs))
    if show_cluster_of is not None:
        label = get_color_label(show_cluster_of, labels)

    for idx, data in enumerate(pairs):

        c_ax = get_current_axis(pairs, ax, idx)

        x_data, y_data = data.iloc[:, 0], data.iloc[:, 1]
        x_tag, y_tag = data.columns[0], data.columns[1]
        if show_cluster_of is None or len(label) == 0:
            if (x_tag, y_tag) in labels.columns:
                c_label = [i + 1 for i in labels[(x_tag, y_tag)]]
            else:
                c_label = [i + 1 for i in labels.iloc[:, 0]]
        else:
            c_label = label
        legend = True if idx == 0 else False
        sns.scatterplot(x=x_data, y=y_data, ax=c_ax, hue=c_label,
                        palette="tab10", legend=legend, s=60)

        user_ids = data.index if user_ids is None else user_ids
        for i, j in enumerate(user_ids):
            c_ax.annotate(str(j), [list(x_data)[i], list(y_data)[i]],
                          horizontalalignment='center',
                          verticalalignment='center', size=5)

        c_ax.set_title(f"{x_tag} vs {y_tag}", fontweight='semibold')
        set_labels(pairs, x_tag, c_ax, 2, y_tag)
        if abline:
            end = min(max(y_data), max(x_data))
            c_ax.plot([0, end], [0, end], 'r-')

    plt.tight_layout()
    save_figure(save, dpi, os.path.join(path, "scatterplot"))
    plt.show()

    return ax


def cluster_single_plot(columns: list[pd.DataFrame], labels: pd.DataFrame,
                        save=True, path="", dpi=300, n_bins=20,
                        show_cluster_of=None) -> list:
    """Plot the kmeans clustering result for single features

    Args:
        columns (list[pd.DataFrame]): The input dataframes to be plotted
        labels (list): List of pandas series with labels for each kmeans result
        save (bool): Whether to save the figure
        path (string): Path to save the figure
        dpi (int): Dots per inch for saving the figure
        n_bins (int): Number of bins to show in histogram
        show_cluster_of (list): Feature combi whose clusters shown in all plots
        
    Returns:
        list: List of axes of the plots

    """
    ax = init_subplots_plot(len(columns))

    for idx, data in enumerate(columns):
        c_ax = get_current_axis(list(labels.columns), ax, idx)
        data = data.reset_index().drop('index', axis=1)
        tag = data.columns[0]
        label = [i + 1 for i in labels[(tag,)]]
        data = pd.concat([data.iloc[:, 0], pd.Series(label)], axis=1)
        data.columns = [tag, 'label']
        legend = True if idx == 0 else False
        if len(data[tag].unique()) < 10:
            data['count'] = 1
            data = data.groupby([tag, 'label']).count().reset_index()
            sns.barplot(ax=c_ax, data=data, x=tag, y='count', hue='label', 
                        palette="tab10")
        else:
            sns.histplot(ax=c_ax, data=data, x=tag, hue='label', 
                         multiple='stack', bins=n_bins, palette="tab10", 
                         legend=legend)
        c_ax.set_title(f"{tag}", fontweight='semibold')
        c_ax.set_ylabel("occurrences")
        set_labels(columns, tag, c_ax)

    plt.tight_layout()
    save_figure(save, dpi, os.path.join(path, "histogram"))
    plt.show()

    return ax


def plot_kmeans_centers(centers: pd.DataFrame, centers_inv: pd.DataFrame,
                        save=True, path="", dpi=300):
    """Plot the centers of the kmeans clustering in terms of a heatmap

    Args:
        centers (pd.DataFrame): The centers determined
        centers_inv (pd.DataFrame): The centers inverse transformed
        save (bool, optional): Whether to save the plot. Defaults to True.
        path (str, optional): The path where to save. Defaults to "".
        dpi (int, optional): The dpi to use for the figure. Defaults to 300.
    
    """
    centers_dfs = get_centers_list(centers)
    centers_inv_dfs = get_centers_list(centers_inv)

    first_df = centers_dfs[0]
    scales = len(first_df.columns) * 1.5, len(first_df.index) * 0.5 + 1

    ax = init_subplots_plot(len(centers_dfs), scales=scales)

    for idx, df in enumerate(centers_dfs):
        c_ax_0 = get_current_axis(centers_dfs, ax, idx)
        df.columns = [c[1] for c in df.columns]
        sns.heatmap(df, ax=c_ax_0, annot=centers_inv_dfs[idx], fmt='.2f',
                    cbar=False)
        
    plt.suptitle('Cluster centers of kMeans', fontweight='bold')
    plt.tight_layout()
    save_figure(save, dpi, os.path.join(path, "center"))
    plt.show()


def get_color_label(show_cluster_of: list, labels: pd.DataFrame) -> list:
    """Get the proper color labels for the clusters using show_cluster_of

        Args:
            show_cluster_of (list): List of features to show the clusters of
            labels (pd.DataFrame): The user labels of the corresponding clusters

        """
    label = []

    all_cols = ' or '.join([str(list(lab)) for lab in labels.columns])
    warn_m = 'The chosen feature combi in show_cluster_of does not exist, ' \
             'showing individual clusters instead. ' \
             f'\nSelect {all_cols} to show this cluster in each plot.'
    if tuple(show_cluster_of) in list(labels.columns):
        label = list(labels[tuple(show_cluster_of)])
    else:
        warnings.warn(warn_m)

    return label
