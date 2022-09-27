"""Helper functions for the preparer_fcts.py module"""

from collections import Counter
from itertools import product, repeat
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import statsmodels.api as sm


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


def time_dependent_features(user_dfs: list, labels: list, f_typ='learntype', 
                            mapping=None, split_cats=False, plot=False,
                            n_min=3) -> pd.DataFrame:
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

    Returns:
        pd.DataFrame: The df with time dependent features of each user
        
    """
    n_cats = 6 if split_cats else 1
    fit_param = {f'{label}Fit': [] for label in labels}
    fit_param_lb = {f'{label}FitLb': [] for label in labels}
    fit_param_ub = {f'{label}FitUb': [] for label in labels}

    for user_df in user_dfs:
        
        user_df = user_df[user_df.SecSpent < 1800]
        user_df['UserCumSec'] = user_df['SecSpent'].cumsum()
        user_df = user_df[user_df[f_typ] >= 0]
        
        if mapping is not None:
            user_df = user_df[user_df[f_typ] < len(mapping)]
            mapper = dict(zip(range(len(mapping)), mapping))
            user_df[f_typ] = user_df[f_typ].map(mapper)
        
        if plot:
            plt.figure(figsize=(16, 3))
        axes = []
        
        values = [None] if len(labels) == 1 else dict(Counter(mapping)).keys()

        for label, value in zip(labels, values):
            
            a, a_lb, a_ub, weights = [], [], [], []
            
            for cat in range(n_cats):
                
                data = user_df[user_df.Category == cat] if split_cats \
                    else user_df
                data['UserCumFeatureSec'] = data['SecSpent'].cumsum()
                    
                if value is None:
                    x, y = np.array(data.UserCumSec)/60, np.array(data[f_typ])
                    reg = add_regression(x, y, axes, a, a_lb, a_ub, weights, 
                                         plot, n_min)
                    axes, a, a_lb, a_ub, weights = reg
                    
                elif len(data.index) > 3*len(labels):
                    data[label] = user_df['SecSpent']
                    data.loc[user_df[f_typ] != value, label] = 0
                    data[label] = data[label].cumsum()
                    data[label] = data[label].div(data['UserCumFeatureSec'])
                    x, y = np.array(data.UserCumSec)/60, np.array(data[label])
                    reg = add_regression(x, y, axes, a, a_lb, a_ub, weights, 
                                         plot, n_min)
                    axes, a, a_lb, a_ub, weights = reg

            a_mean = 100*np.average(a, weights=weights) if a else 0
            a_lb_mean = 100*np.average(a_lb, weights=weights) if a_lb else 0
            a_ub_mean = 100*np.average(a_ub, weights=weights) if a_ub else 0
            
            fit_param[f'{label}Fit'].append(a_mean)
            fit_param_lb[f'{label}FitLb'].append(a_lb_mean)
            fit_param_ub[f'{label}FitUb'].append(a_ub_mean)
            
        if plot:
            legend_entries = [x for item in zip(labels, fit_param.keys()) 
                              for x in item]
            plt.legend(dict(zip(legend_entries, axes)))
            
    return pd.DataFrame(fit_param)


def add_regression(x: np.array, y: np.array, axes: plt.Axes, a: list, 
                   a_lb: list, a_ub: list, weights: list, plot: bool, 
                   n_min: int) -> tuple:
    """Add a linear regression result and plot to a given list of results

    Args:
        x (np.array): x data
        y (np.array): y data
        axes (plt.Axes): The axis handle
        a (list): List of slope parameters from linear regressions
        a_lb (list): List of lower bounds for slope parameters
        a_ub (list): List of upper bounds for slope parameters
        weights (list): List of weights for each linear regression result
        plot (bool): Whether to show the regression plot
        n_min (int): Minimum number of data points required for regression

    Returns:
        tuple: The new axis handle and the extended result lists
        
    """
    if len(x) >= n_min:
        x_fit = sm.add_constant(x)
        reg = sm.GLS(y, x_fit).fit()
        if plot:
            axes.append(plt.plot(x, y, 'x'))
            axes.append(plt.plot(x, x*reg.params[1] + reg.params[0]))
        a.append(reg.params[1])
        a_lb.append(reg.conf_int(0.01)[1][0])
        a_ub.append(reg.conf_int(0.01)[1][1])
        weights.append(len(x))
        
    return axes, a, a_lb, a_ub, weights
