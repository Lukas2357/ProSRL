"""Classify features using the scikit learn decision tree model"""

from classify_fcts import decision_tree, merge_pers_feature


def do_classify(df, features, predict, max_depth, min_samples_leaf, save, dpi):
    """Do classify fct to be called from Analyser or directly

    Args:
        see decision_tree
    
    """
    decision_tree(df, features, predict, max_depth, min_samples_leaf, save, dpi)


if __name__ == '__main__':
    do_classify(merge_pers_feature(), ['Tests', 'Übungen'], 'RF_Diff',
                3, 6, True, 100)
