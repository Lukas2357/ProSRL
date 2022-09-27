"""Functions for the preparer.py module"""

from os import path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .prep_help_fcts import *
from ..config.constants import DATA_PATH
from ..transformer.transform_help_fcts import get_learn_types


def load_transformed_data():
    """Loads the transformed data from csv file

    Returns:
        pd.DataFrame: The loaded data as pandas dataframe

    """
    return pd.read_csv(path.join(DATA_PATH, "data_trafo.csv")).iloc[:, 1:]


def make_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Use one-hot encoding to get 0/1 values from categorical features

    Args:
        df (pd.DataFrame): The df corresponding to data_trafo

    Returns:
        df (pd.DataFrame): The df with dummies instead of categorical columns

    """
    d1 = pd.get_dummies(df["LearnType"], prefix='lType')
    d1.columns = list(get_learn_types().values())
    d2 = pd.get_dummies(df["Level"], prefix='Level')
    d2.columns = ["Grund", "Erweitert"]
    d3 = pd.get_dummies(df["Category"], prefix='Cat')
    df_dummy = pd.concat([df, d1, d2, d3], axis=1)

    return df_dummy.drop(["Level"], axis=1)


def get_column_types() -> dict:
    """Get a dictionary to indicate for each column to which kind from learn,
        info, type, category, level or normalized it belongs.

    Returns:
         dict: The dictionary of column types

    """

    c = {'learn': ['Tests', 'Übungen', 'Kurzaufgaben', 'Beispiele'],
         'info': ['BK_Info', 'Ü_Info'],
         'learn2': ['Tests', 'Übungen', 'Kurzaufgaben'],
         'info2': ['BK_Info', 'Ü_Info', 'Beispiele'],
         'cat': [f'Cat_{i}' for i in range(6)],
         'level': ['Grund', 'Erweitert'],
         'freq': ['fMean', 'f1', 'f12', 'f1Amp', 'f12Amp'],
         'per_act': ['Learn_TimePerAct', 'Info_TimePerAct', 'Tot_TimePerAct'],
         'per_act2': ['Learn2_TimePerAct', 'Info2_TimePerAct'],
         'time': ['Learn', 'Info', 'Tot',
                  'LearnPerc', 'InfoPerc'],
         'time2': ['Learn2', 'Info2', 'Learn2Perc',
                   'Info2Perc'],
         'acts': ['LearnActs', 'InfoActs', 'TotActs', 'LearnActsPerc',
                  'InfoActsPerc'],
         'acts2': ['Learn2Acts', 'Info2Acts', 'Learn2ActsPerc',
                   'Info2ActsPerc'],
         'long_short': ['ShortLearnActsPerc', 'LongLearnActsPerc',
                        'ShortInfoActsPerc', 'LongInfoActsPerc',
                        'LongActsPerc', 'ShortActsPerc'],
         'other': ['nCat', 'duplicatesPerc', 'DidTest'],
         'agg': ['K+B', 'K+B+BKI', 'Ü+ÜI'],
         'cat_test': ['BeforeTestTime', 'AfterTestTime', 'TimePerCat'],
         'entropy': ['LearnTypeEntropy', 'CategoryEntropy'],
         'mean': ['meanResponsTask', 'meanTestResQuant', 'meandifficulty'],
         'regression': ['TRegrFit', 'ÜRegrFit', 'BKRegrFit', 'CatTRegrFit',
                        'CatÜRegrFit', 'CatBKRegrFit', 'DiffHighRegrFit',
                        'KÜrRegrFit', 'TResQuantRegrFit', 'TSecSpentRegrFit',
                        'ÜSecSpentRegrFit', 'BKSecSpentRegrFit',
                        'SecSpentRegrFit'],
         'count': ['nResponsTask', 'nTestResQuant', 'ndifficulty'],
         'rasch_nst': ['NSt_Pre', 'NSt_Post', 'NSt_Diff'],
         'rasch_rf': ['RF_Pre', 'RF_Post', 'RF_Diff'],
         'person': ['SWJK', 'Note', 'SWK', 'FSK', 'AV', 'ALZO', 'VLZO', 'LZO',
                    'Inte', 'KLO', 'KLW', 'MKL', 'RZM', 'RSA', 'RAK'],
         'fool': ['backfrac', 'foolfrac1', 'foolfrac2', 'foolfrac12']}

    agg_norm = add_agg_columns(get_labels_only=True)
    c = c | {'type': c['learn'] + c['info']}
    c = c | {'norm': c['type'] + c['cat'] + c['level'] + agg_norm,
             'all': [feature for typ in c.values() for feature in typ]}

    return c


def add_secondary_columns(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Add columns to activity df derived from the existing columns

    Args:
        df (pd.DataFrame): The transformed df
        column_types (dict): The dictionary for column types

    Returns:
        pd.DataFrame: The extended df

    """
    df = add_act_time_columns(df, column_types)
    df = add_long_short_columns(df, column_types)

    return df


def add_act_time_columns(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Add columns related to number of acts and time spent

    Args: 
        df (pd.DataFrame): The transformed df
        column_types (dict): The dictionary for column types

    Returns:
        pd.DataFrame: The extended df

    """
    df = df[df.SecSpent > 0]
    drops = ['User', 'LearnType', 'SecSpent', 'Category', 'ResponsTask',
             'TestResQual', 'TestResQuant', 'difficulty', 'Label']
    times = df.mul(df['SecSpent'], axis=0)

    learn_cols = df[column_types['learn']]
    learn_cols_no_b = learn_cols.drop(["Beispiele"], axis=1)
    learnings = learn_cols.sum(axis=1)
    learnings_no_b = learn_cols_no_b.sum(axis=1)

    info_cols = df[column_types['info']]
    info_cols_with_b = pd.concat([learn_cols, df["Beispiele"]], axis=1)
    infos = info_cols.sum(axis=1)
    infos_with_b = info_cols_with_b.sum(axis=1)

    overviews = df[['Übersicht']].sum(axis=1)

    times = times.drop(drops, axis=1)
    cols = list(df.columns)
    for idx, c in enumerate(cols):
        if c not in drops:
            cols[idx] = c + "Acts"
    df.columns = cols
    type_time_cols = [c + "Acts" for c in get_column_types()['type']]
    df = pd.concat([df[drops], df[type_time_cols], times], axis=1)

    df["LearnActs"] = learnings
    df["InfoActs"] = infos
    df["Learn2Acts"] = learnings_no_b
    df["Info2Acts"] = infos_with_b

    df["TotActs"] = df["LearnActs"] + df["InfoActs"] + overviews
    df["Learn"] = times[column_types['learn']].sum(axis=1)
    df["Info"] = times[column_types['info']].sum(axis=1)
    df["Learn2"] = times[column_types['learn2']].sum(axis=1)
    df["Info2"] = times[column_types['info2']].sum(axis=1)
    overview_time = times[['Übersicht']].sum(axis=1)
    df["Tot"] = overview_time + df["Learn"] + df["Info"]

    return df[df.Tot < 1800]


def add_long_short_columns(df, column_types):
    """Add columns related to short and long acts

    Args: 
        df (pd.DataFrame): The transformed df
        column_types (dict): The dictionary for column types

    Returns:
        pd.DataFrame: The extended df

    """

    short_condition_learn = df['Learn'].between(0, 15, inclusive='neither')
    short_condition_info = df['Info'].between(0, 15, inclusive='neither')
    df["ShortLearnActs"] = np.where(short_condition_learn, 1, 0)
    df["ShortInfoActs"] = np.where(short_condition_info, 1, 0)

    long_conditions = {idx: df[df[c] > 0][c].mean() + 2 * df[df[c] > 0][c].std()
                       for idx, c in enumerate(column_types['type'])}
    df["LongCondition"] = df["LearnType"].map(long_conditions)
    df["LongLearnActs"] = np.where(df['Learn'] > df['LongCondition'], 1, 0)
    df["LongInfoActs"] = np.where(df['Info'] > df['LongCondition'], 1, 0)

    return df


def scale_features(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Scale the features to percent of total learn time/activities etc.

    Args:
        df (pd.DataFrame): The input dataframe
        column_types (dict): The column types to get norm columns from

    Returns:
        pd.Dataframe: The scaled dataframe

    """

    norm_cols = column_types['norm']
    df = add_divided_columns(df, norm_cols, 'Tot', perc=True, postfix='')
    time_cols = [c for c in df.columns if 'Time' in c] + ['fMean']
    df[time_cols] = df[time_cols] / 60

    perc_time_cols = ['Learn', 'Info', 'Learn2', 'Info2']
    df = add_divided_columns(df, perc_time_cols, 'Tot', perc=True)

    perc_act_cols = ['LearnActs', 'InfoActs', 'Learn2Acts', 'Info2Acts',
                     'duplicates', 'ShortLearnActs', 'ShortInfoActs',
                     'LongLearnActs', 'LongInfoActs', 'ShortActs', 'LongActs']

    df = add_divided_columns(df, perc_act_cols, 'TotActs', perc=True)

    return df


def get_summed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get the df grouped by user and summed over all columns

    Args:
        df (pd.Dataframe): The df corresponding to data_trafo

    Returns:
        pd.DataFrame: The summed dataframe

    """
    df = df.drop(["TotSec", "SecSpent", "UserCumSec",
                  "LongCondition", "Label", "LearnType"], axis=1)

    grouped_df = df.groupby("User")
    sum_df = grouped_df.sum().reset_index()

    for col in ['ResponsTask', 'TestResQual', 'TestResQuant', 'difficulty']:
        col_mean = df[df[col] >= 0].groupby('User').mean()[col]
        sum_df[col] = col_mean
        sum_df[col].fillna(-1, inplace=True)

    return sum_df


def add_agg_columns(df=pd.DataFrame(), get_labels_only=False) -> any:
    """Add aggregated columns from various percentage columns

    Args:
        df (pd.DataFrame): The input df
        get_labels_only (bool): Get only feature labels and do not perform

    Returns:
        any: The df extended by the aggregated columns or just the labels list

    """

    labels = ['K+B', 'K+Ü', 'K+Ü+T', 'K+B+Ü', 'K+BKI', 'B+BKI', 'Ü+ÜI',
              'BKI+ÜI', 'K+B+BKI', 'K+B+BKI+ÜI', 'K+B+BKI+Ü+ÜI', 'ShortActs',
              'LongActs', 'read']

    if get_labels_only:
        return labels

    combis = [('Kurzaufgaben', 'Beispiele'),
              ('Kurzaufgaben', 'Übungen'),
              ('K+Ü', 'Tests'),
              ('K+B', 'Übungen'),
              ('Kurzaufgaben', 'BK_Info'),
              ('Beispiele', 'BK_Info'),
              ('Übungen', 'Ü_Info'),
              ('BK_Info', 'Ü_Info'),
              ('K+B', 'BK_Info'),
              ('K+B+BKI', 'Ü_Info'),
              ('K+B+BKI+ÜI', 'Übungen'),
              ('ShortInfoActs', 'ShortLearnActs'),
              ('LongInfoActs', 'LongLearnActs'),
              ('Beispiele', 'BK_Info')]

    for label, combi in zip(labels, combis):
        df[label] = df[[c for c in combi]].sum(axis=1)
        if label not in ['ShortActs', 'LongActs']:
            df[label+"Acts"] = df[[c+"Acts" for c in combi]].sum(axis=1)

    print()
    return df


def add_additional_features(df: pd.DataFrame, sum_df: pd.DataFrame,
                            feature='LearnType', max_n_cond_entropy=1,
                            mapper=(0, 1, 1, 2, 2, 2, None)) -> pd.DataFrame:
    """Add additional features after summing the dfs

    Args:
        df (pd.DataFrame): The transformed df
        sum_df (pd.DataFrame): The summed df
        feature (str): The feature to use for predict chance and cond entropy
        mapper (tuple): The mapping to use for predict chance and cond entropy
        max_n_cond_entropy (int): Max n for conditional entropy calculation

    Returns:
        pd.DataFrame: The summed df extended by additional features

    """
    user_dfs = get_user_dfs(df)

    div_cols = [c for c in sum_df.columns
                if 'Acts' in c and 'Long' not in c and 'Short' not in c]
    cols = [c[:-4] for c in div_cols]
    sum_df = add_divided_columns(sum_df, cols, div_cols)

    sum_df['duplicates'] = get_duplicates(df)
    sum_df['nCat'] = sum_df[get_column_types()['cat']].astype(bool).sum(axis=1)
    sum_df['LearnTypeEntropy'] = get_type_entropy(sum_df, f_type='learntype')
    sum_df['CategoryEntropy'] = get_type_entropy(sum_df, f_type='cat')
    sum_df['DidTest'] = sum_df['Tests'].astype(bool).astype(int)

    add_feature = [get_freq_features(user_dfs),
                   get_predict_features(user_dfs, feature=feature,
                                        mapper=mapper),
                   get_entropy_features(user_dfs, feature=feature,
                                        mapper=mapper,
                                        max_n_cond_entropy=max_n_cond_entropy),
                   get_before_after_test_times(user_dfs),
                   get_time_dependent_features(user_dfs),
                   get_mean_features(user_dfs),
                   get_fool_features(user_dfs)]

    all_f = [sum_df.drop(['Category', 'ResponsTask', 'TestResQual',
                          'TestResQuant', 'difficulty'], axis=1)] + add_feature

    return pd.concat(all_f, axis=1)


def get_predict_features(user_dfs: list, feature='LearnType',
                         mapper=(0, 1, 1, 2, 2, 2, None)) -> pd.DataFrame:
    """Get features describing how good user activities can be predicted

    Args:
        user_dfs (list): The user dfs to analyse
        feature (str): The feature to use
        mapper (tuple): Feature integer values will be mapped on this

    Returns:
        pd.DataFrame: Df for the predict features

    """
    mapper = dict(zip(range(len(mapper)), mapper))
    result = pd.DataFrame()
    for known in range(1, 5):
        chances = []
        for user_df in user_dfs:
            user_arr = user_df[feature].map(mapper)
            user_arr = np.array([i for i in user_arr if i is not None])
            chances.append(predict_chance(user_arr, known=known) * 100)
        result[f'PredChance{known}'] = chances

    return result


def get_entropy_features(user_dfs: list, feature='LearnType',
                         mapper=(0, 1, 1, 2, 2, 2, None),
                         max_n_cond_entropy=1) -> pd.DataFrame:
    """Get features of the conditional entropies for any user

    Args:
        user_dfs (list): The dfs of users to analyse
        feature (str): The feature to use
        mapper (tuple): Feature integer values will be mapped on this
        max_n_cond_entropy (int): Max n for conditional entropy calculation


    Returns:
        pd.DataFrame: The dfs of entropy features of each user

    """
    mapper = dict(zip(range(len(mapper)), mapper))
    result = pd.DataFrame()
    for distance in range(1, max_n_cond_entropy+1):
        entropy = []
        for user_df in user_dfs:
            user_arr = user_df[feature].map(mapper)
            user_arr = np.array([i for i in user_arr if i is not None])
            entropy.append(cond_entropy(user_arr, distance=distance) * 100)
        result[f'CondEntropy{distance}'] = entropy

    return result


def get_freq_features(user_dfs: list) -> pd.DataFrame:
    """Call freq_features function to get the frequency features for each user

    Args:
        user_dfs (list): The dfs of users to analyse

    Returns:
        pd.DataFrame: The df of frequency features for each user

    """
    freq_domi = freq_features(user_dfs)
    freq_mean = freq_features(user_dfs, True, list(range(6)), True)

    return pd.concat([freq_domi.drop(['fMean'], axis=1),
                      freq_mean['fMean']], axis=1)


def get_time_dependent_features(user_dfs: list) -> pd.DataFrame:
    """Call time_dependent_features function to get the regression features

    Args:
        user_dfs (list): The dfs of users to analyse

    Returns:
        pd.DataFrame: The df of regression features for each user

    """
    mapping = (0, 1, 1, 2, 2, 2)
    labels = ['TRegr', 'ÜRegr', 'BKRegr']
    f1 = time_dependent_features(user_dfs, labels, f_typ='LearnType',
                                 mapping=mapping)
    labels = ['CatTRegr', 'CatÜRegr', 'CatBKRegr']
    f2 = time_dependent_features(user_dfs, labels, f_typ='LearnType',
                                 mapping=mapping, split_cats=True)
    labels = ['DiffLowRegr', 'DiffMedRegr', 'DiffHighRegr']
    f3 = time_dependent_features(user_dfs, labels, f_typ='difficulty')
    labels = ['KÜfRegr', 'KÜtwrRegr', 'KÜrRegr']
    f4 = time_dependent_features(user_dfs, labels, f_typ='ResponsTask')
    labels = ['TResQuantRegr']
    f5 = time_dependent_features(user_dfs, labels, f_typ='TestResQuant')

    f6 = time_dependent_sec_spent(user_dfs)

    return pd.concat([f1, f2, f3, f4, f5] + f6, axis=1)


def time_dependent_sec_spent(user_dfs, clabel='SecSpentRegr', cpath='',
                             **args) -> list:
    """Call time_dependent_features function to get time dependent second spent

        Args:
            user_dfs (list): The dfs of users to analyse
            clabel (str): The label of the plot legend entry
            cpath (str): The path to save the figure in
            args (any): Optional arguments for time_dependent_features function

        Returns:
            list: The list of second spent feature value for each user

        """

    temp_user_dfs = []
    for user_df in user_dfs:
        temp_user_df = user_df[user_df['LearnType'] < 6]
        mapping = dict(zip(range(6), [0, 1, 1, 2, 2, 2]))
        temp_user_df['LearnType'] = temp_user_df.LearnType.map(mapping)
        temp_user_dfs.append(temp_user_df)

    f6 = []
    for label, learntype in zip(('T', 'Ü', 'BK'), range(3)):
        labels = [f'{label}{clabel}']
        data = [df[df.LearnType == learntype] for df in temp_user_dfs]
        spath = os.path.join(cpath, labels[0])
        f = time_dependent_features(data, labels, f_typ='SecSpent',
                                    path=spath, **args)
        f6.append(f)
    f6.append(time_dependent_features(temp_user_dfs, [clabel], f_typ='SecSpent',
                                      path=os.path.join(cpath, clabel), **args))

    return f6


def get_mean_features(user_dfs: list) -> pd.DataFrame:
    """Get features associated with means of activities or responses

    Args:
        user_dfs (list): The dfs of users to analyse

    Returns:
        pd.DataFrame: The df of mean features for each user

    """
    features = ['ResponsTask', 'TestResQual', 'TestResQuant', 'difficulty']
    means = {f'mean{feature}': [] for feature in features}
    counts = {f'n{feature}': [] for feature in features}
    for user_df in user_dfs:
        for feature in features:
            c_mean = user_df.loc[user_df[feature] >= 0, feature].mean()
            c_count = user_df.loc[user_df[feature] >= 0, feature].count()
            c_mean = -1 if np.isnan(c_mean) else c_mean
            means[f'mean{feature}'].append(c_mean)
            counts[f'n{feature}'].append(c_count)
    return pd.DataFrame(means | counts)


def scaled_pca(df: pd.DataFrame, center=None, scaler="MinMax") -> tuple:
    """Apply scaling and PCA to dataframe

    Args:
        df (pd.DataFrame): The input dataframe
        center (optional): Kmeans cluster center to transform
        scaler (string): The type of scaler to use

    Returns:
        tuple: The PCs and the PCA handle as well as headers of used columns

    """
    n_components = len(df.columns)

    if scaler is not None:
        scaler = MinMaxScaler() if scaler == 'MinMax' else StandardScaler()
        data = scaler.fit_transform(df)
    else:
        data = df

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    if center is not None:
        center.columns = center.columns.droplevel(0)
        center = np.array(center)
        center = pca.transform(center)

    var_percent = pca.explained_variance_ratio_

    pcs = pd.DataFrame({f"PC{i + 1}_{var:.1f}%": components[:, i]
                        for i, var in enumerate(var_percent * 100)})

    return pcs, pca, df.columns, center


def drop_super_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows corresponding to super pages of the actual learn pages (lossy)

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Transformed dataframe

    """
    super_pages = ['Startseite', 'Tour', 'Impressum', 'Notizen',
                   'Logout', 'Übersicht']

    df = df[~df['Label'].isin(super_pages)]

    return df
