"""Functions for the clustering.py module"""

from ..cluster.cluster_fcts import load_prep_data
from .cluster_eval_help_fcts import *

df = load_prep_data()
methods = ['elbow_plot', 'cross_valid', 'cross_average', 'sil_analysis']


def get_file_path(feature, dim, alg, idx):
    tag = feature + '-single' if dim == 1 else feature + '-pairs'
    folder_path = os.path.join(os.getcwd(), "../cluster/result_plots", alg,
                               methods[idx])
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, tag)
    return file_path


def do_elbow(feature, dim, alg):
    file_path = get_file_path(feature, dim, alg, 0)
    elbow_plot(df, feature, save=True, path=file_path, dim=dim, alg=alg)
    
    
def do_cross_validation(left_outs, clusters, feature, dim, alg):
    file_path = get_file_path(feature, dim, alg, 1)
    cross_validation(df, feature, left_outs=left_outs, dim=dim, plot=True, 
                     save=True, path=file_path, alg=alg, clusters=clusters)    
    

def do_average_cross_validation(n_runs, left_outs, clusters, feature, dim, alg):
    runs = []
    for run in range(n_runs):
        runs.append(cross_validation(df, feature, left_outs=left_outs, dim=dim, 
                                     clusters=clusters, alg=alg))
        print(f"run {run+1}/{n_runs} finished")
    file_path = get_file_path(feature, dim, alg, 2)
    average_cross_validation(runs, save=True, path=file_path)


def do_silhouette_analysis(n_clusters, feature, dim, alg):
    file_path = get_file_path(feature, dim, alg, 3)
    silhouette_analysis(df, features=feature, dim=dim, save=True,
                        path=file_path, alg=alg, n_clusters=n_clusters)
