"""Preparer for raw_data after transformation (clustering, classification, ...)"""

from .prep_fcts import *
from ..config.helper import save_data


def do_prep(use_time=True, formats=('csv', ), feature='LearnType',
            mapper=(0, 1, 1, 2, 2, 2, None), max_n_cond_entropy=1):
    """Do preparation function to be called from main or directly

    Args:
        use_time (bool, optional): Whether to use times instead of activity
                                   count for features. Defaults to True.
        formats (tuple): The formats to save resulting raw_data to (csv and/or xlsx)
        feature (str): The feature to use for predict chance and cond entropy
        mapper (tuple): The mapping to use for predict chance and cond entropy
        max_n_cond_entropy (int): Max n for conditional entropy calculation
    
    """
    print("Prepare for clustering by getting new features and sum df by user")

    df = load_transformed_data()
    column_types = get_column_types()

    df = drop_super_pages(df)
    df.loc[df.SecSpent > 1800, 'SecSpent'] = 1800

    df = make_dummies(df)
    df = add_secondary_columns(df, column_types)
    df = add_agg_columns(df)

    sum_df = get_summed_df(df)
    sum_df = add_additional_features(df, sum_df, feature=feature,
                                     max_n_cond_entropy=max_n_cond_entropy,
                                     mapper=mapper)

    sum_df = scale_features(sum_df, column_types)

    save_data(sum_df, 'data_prep', formats=formats)

    print("Finished preparation, created ROOT/raw_data/data_prep.csv")
