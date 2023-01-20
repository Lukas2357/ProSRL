"""Functions for the classify.py module"""

import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree

from ..cluster.cluster_fcts import load_prep_data
from ..cluster.cluster_plot_help_fcts import save_figure
from ..config.constants import RESULTS_PATH
from ..config.helper import load_input, save_data


def merge_pers_feature(rm_incomplete_user=True, fill_na=True) -> pd.DataFrame:
    """Merge activity features with personal features

    Args:
        rm_incomplete_user (bool, optional): Remove user that did not do a test 
                                             afterwards. Defaults to True.
        fill_na (bool, optional): Whether to fill NA with 0 afterwards.
                                  Defaults to True.

    Returns:
        pd.DataFrame: The merged df or data_prep.csv and Personenmerkmale.csv
    
    """
    required_columns = ['Person', 'Rasch_Stacking_Niveaustufen_Differenz']
    data = load_input('person_features.csv', columns=required_columns)
    required_columns = ['Person', 'User', 'user_id']
    relation = load_input('relation_user_code.csv', columns=required_columns)

    df = pd.merge(relation, data).sort_values('user_id')
    df = df.drop([x for x in ['user', 'Person', 'Unnamed: 0'] if x in df.columns],
                 axis=1)

    try:
        columns_nivst = ['NSt_' + x[28:] for x in df.columns if 'Niveaustufe' in x]
        columns_nivst[-1] = 'NSt_Diff'
        columns_rf = ['RF_' + x[18:] for x in df.columns if 'RF' in x]
        columns_rf[-1] = 'RF_Diff'
        df.columns = ['User'] + columns_nivst + list(df.columns)[4:-3] + columns_rf
    except IndexError:
        pass

    tot = pd.merge(df, load_prep_data(), left_on='User', right_index=True)
    tot["User"] = tot["User"]

    if rm_incomplete_user:
        tot = tot.dropna(subset=['NSt_Post'])
    if fill_na:
        tot = tot.fillna(0)

    save_data(tot, 'data_prep_with_personal')
    print('Finished merging data_prep and person_features, created '
          'ROOT/raw_data/data_prep_with_personal.csv')

    return tot


def decision_tree(df: pd.DataFrame, features: list, predict: str, max_depth=3, 
                  min_samples_leaf=6, save=True, dpi=100):
    """Implementation of scikit-learn decision tree algorithm

    Args:
        df (pd.DataFrame): The df to use for the decision tree
        features (list): The features to consider
        predict (str): The feature to predict with the tree
        max_depth (int, optional): Max depth of the tree. Defaults to 3.
        min_samples_leaf (int, optional): Min leaves of the tree. Defaults to 6.
        save (bool, optional): Whether to save the plot. Defaults to True.
        dpi (int, optional): Dots per inch of the plot. Defaults to 100.
    
    """
    x = df[features].drop(predict, axis=1) if predict in features \
        else df[features]
    y = df[predict]

    clf = tree.DecisionTreeRegressor(max_depth=max_depth, 
                                     min_samples_leaf=min_samples_leaf)
    clf = clf.fit(x, y)

    fig, _ = plt.subplots(1, 1, figsize=(2*clf.get_depth(), 2*clf.get_depth()))
    
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    tree.plot_tree(clf, feature_names=x.columns, filled=True)
    plt.tight_layout()
    plt.title(f'Classification of {predict} \n \n', fontweight='bold')
    
    if save:
        path = os.path.join(RESULTS_PATH, 'classify', f'dec_tree_{predict}')
        save_figure(save=save, dpi=dpi, path=path, tight=False, fig=fig)
    
    plt.show()
    