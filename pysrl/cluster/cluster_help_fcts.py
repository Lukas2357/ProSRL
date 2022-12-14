"""Helper functions for the cluster_fct.py module"""

import os
from itertools import combinations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import clustermap

from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, \
    SpectralClustering, estimate_bandwidth

from .cluster_plot_help_fcts import save_figure
from ..config.constants import RESULTS_PATH
from ..config.helper import save_data, load_input


def get_folder_path(alg: str, features: list) -> str:
    """Get folder path to store results of raw_data and plots

    Args:
        alg (str): The algorithm used for clustering
        features (list): The features used for clustering

    Returns:
        str: The folder path to store results
        
    """
    folder_path = os.path.join(RESULTS_PATH, 'cluster', alg)
    os.makedirs(folder_path, exist_ok=True)

    tag = '-'.join(features) if len(features) < 7 \
        else '-'.join(features[:7]) + f'-and_{len(features)-7}_more'

    folder_path = os.path.join(folder_path, tag)

    return folder_path


def save_cluster_result(combis: list, result: dict, path: str, user: pd.Series):
    """Save the result raw_data of a clustering process

    Args:
        combis (list): The feature combis used
        result (dict): The result dict with labels, centers, ...
        path (str): Path to where results should be saved
        user (pd.Series): The user ids to concatenate with the results
        
    """
    relation = load_input("relation_user_code")
    user = pd.merge(user, relation, left_on="User", right_on="user_id")
    user = user.drop("User", axis=1)

    for idx, df in enumerate(combis):
        df = pd.concat([df, user], axis=1)
        save_data(df, os.path.join(path, f'combi_{idx}'))
    for param, df in result.items():
        if param != "model":
            if param == "labels":
                df = pd.concat([df, user], axis=1)
            save_data(df, os.path.join(path, param))


def get_feature_combis(df: pd.DataFrame, features: list, dim=2) -> list:
    """Get combinations of columns for the given feature type or selection

    Args:
        df (pd.DataFrame): The input dataframe
        features (str): The list of features to use
        dim (int): The dimension to use, i.e. number of columns to gather

    Returns:
        list: A list of combinations of features

    """
    selected_df = df[features]

    combis = [pd.concat([selected_df[combi[i]] for i in range(dim)], axis=1)
              for combi in combinations(selected_df.columns, dim)]

    return combis


def cluster_map(df: pd.DataFrame, save=True, path='', dpi=300) -> clustermap:
    """Generate seaborns cluster map for specific features and return it

    Args:
        df (pd:DataFrame): The df including features to show in cluster map
        save (bool, optional): Whether to save th plot. Defaults to True.
        path (str): The path where to save the figure
        dpi (int): Dots per inch for saving the figure

    Returns:
        sns.clustermap: The generated seaborn cluster map object

    """
    df = df.set_index('User')

    c_map = sns.clustermap(df.transpose(), xticklabels=True, z_score=0,
                           annot=df.transpose(), fmt='.0f', vmin=-2, vmax=2)

    c_map.ax_heatmap.figure.set_size_inches(len(df.index)/2,
                                            len(df.columns)/2+1)

    plt.setp(c_map.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(c_map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    c_map.cax.set_visible(False)
    plt.tight_layout()
    save_figure(save=save, path=path, dpi=dpi, fig_format='jpg')
    plt.show()

    return c_map


def generate_model(alg: str, n_cluster: int, data: pd.DataFrame) -> any:
    """Generate a cluster model of the selected algorithm

    Args:
        alg (str): The algorithm used
        n_cluster (int): The number of clusters to generate
        data (pd.DataFrame): The raw_data to use for fitting afterwards

    Raises:
        ValueError: If wrong algorithm is selected

    Returns:
        any: The cluster model constructed for the given algorithm
        
    """
    if alg == 'k_means':
        model = KMeans(n_clusters=n_cluster)
        
    elif alg == 'meanshift':
        bandwidth = estimate_bandwidth(data)
        bandwidth = None if bandwidth < 10 ** -6 else bandwidth
        model = MeanShift(bandwidth=bandwidth)
        
    elif alg == 'spectral':
        model = SpectralClustering(n_clusters=n_cluster)
        
    elif alg == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=None, linkage='ward',
                                        compute_full_tree=True,
                                        distance_threshold=1)
    else:
        raise ValueError('The chosen algorithm does not exist. Use '
                         'k_means, meanshift, spectral or agglomerative')

    return model


def initialize_result_dict(combis: list) -> dict:
    """Initialize an empty dict to use to store cluster modelling results
    
    Args:
        combis (list): The feature combis to use

    Returns:
        dict: The empty dict with the proper structure
        
    """
    combi_tags = [(tuple(combi.columns), feature)
                  for combi in combis for feature in combi.columns]

    multi_idx = pd.MultiIndex.from_tuples(combi_tags)

    result = {'labels': pd.DataFrame(),
              'model': {},
              'center': pd.DataFrame(columns=multi_idx),
              'center_inv': pd.DataFrame(columns=multi_idx)}

    return result


def log_result(result: dict):
    """Log the result of clustering to the console

    Args:
        result (dict): Dict of labels, center etc. from clustering

    """
    print('-> Finished clustering, print results for clusters ...')
    
    for col in result['labels'].columns:
        lab = result['labels'][col]
        
        print(f'User cluster indices found for {col}:')
        
        ids = [' ' + str(i) for i in range(10)] + \
              [str(i) for i in range(10, len(lab))]
        
        print('User_ID:', *ids)
        print('Cluster:', *[' ' + str(i) for i in lab])
