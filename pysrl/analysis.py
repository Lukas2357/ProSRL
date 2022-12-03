"""The analysis module to perform all analysis steps"""

import warnings
import pandas as pd

from .classifier.classify_fcts import merge_pers_feature
from .cluster.cluster_fcts import load_prep_data
from .cluster.cluster_plot_fcts import correlation_heatmap, user_heatmap
from .classifier.classify import do_classify
from .cluster.clustering import do_cluster
from .config.helper import clear_directory, get_feature_types
from .config.constants import RECENT_RESULTS_PATH
from .crawler.crawl import do_crawl
from .organizer.organize import do_orga
from .preparer.prep_plot_fcts import learn_types_lineplots
from .transformer.transform import do_trafo
from .preparer.prepare import do_prep
from .preparer.prep_fcts import get_feature_groups


class Analyst:
    """The Analyst performing all analysis steps

    Analyst comes with two public methods to be addressed from the outside.
    self.raw_lineplots can be used to generate user lineplots a priori.
    self.perform can be used to perform any other task specified by params.
    All other methods are either used by the perform method or exploited to
    set attributes of the Analyst properly.

    """

    def __init__(self):
        """Initialize an Analyst to analyse the data with

        During initialization only attributes are set as seen below.
        The settings represent a usual configuration, but should be overwritten
        by the user under any common circumstances!

        """
        self.performs = dict(crawl=0, orga=0, trafo=0, prep=0, analysis=0)

        self.df = pd.DataFrame()

        self.use_personal = True
        self.rm_incomplete = True
        self.excluded_user = []

        self.feature_types = get_feature_types()

        self.features = ['Tests']
        self.algorithm = 'k_means'
        self.scaler = 'Standard'

        self.image_quality = 1
        self.n_cluster = [2]
        self.n_dim = 1

        self.clustering = True
        self.feature_plots = True
        self.user_heatmap = True
        self.cor_heatmap = True
        self.plot_significant = False
        self.pca_plot = False
        self.user_lineplots = False
        self.classify = False

        self.color_threshold = 2
        self.learntypes = [[0], [1, 2], [3, 4, 5], [6]]
        self.n_bins = 20
        self.abline = False
        self.show_cluster_of = None
        self.clear_recent = True
        self.save_plots = True

        self.predict = 'RF_Diff'
        self.max_depth = 3
        self.min_leaves = 6

        self.formats = ('csv',)
        self.use_time = True

        self.elbow = True
        self.silhouette = True

        self.prep_feature = 'LearnType'
        self.prep_mapper = (0, 1, 1, 2, 2, 2, None)
        self.max_n_cond_entropy = 1

    @property
    def image_quality(self):
        return self._image_dpi

    @image_quality.setter
    def image_quality(self, quality):
        dpis = [50, 100, 300, 600]
        qualities = ['low', 'middle', 'high', 'perfect']
        warn_m = "Only qualities of 'low'/'middle'/'high'/'perfect' or an " \
                 "integer from 0 to 3 supported. Quality not changed."
        if isinstance(quality, str):
            if quality in qualities:
                self._image_dpi = dpis[qualities.index(quality)]
            else:
                warnings.warn(warn_m)
        elif isinstance(quality, int):
            if 0 <= quality <= 3:
                self._image_dpi = dpis[quality]
            else:
                warnings.warn(warn_m)
        else:
            warnings.warn(warn_m)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, feature_list):

        used_features = []
        for feature in feature_list:
            if feature in self.feature_types:
                used_features += get_feature_groups()[feature]
            else:
                used_features.append(feature)

        if len(used_features) == 0:
            raise ValueError('No Features selected!!!')

        self._features = used_features

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, alg):

        warn_m = "Only algorithms of 'k_means', 'meanshift', 'spectral', " \
                 "'agglomerative', 'hierarchy' or an integer from 0 to 4 " \
                 "supported. Algorithm not changed."

        existing_algorithms = ['k_means', 'meanshift', 'spectral',
                               'agglomerative', 'hierarchy']

        if isinstance(alg, int):
            if 0 <= alg <= 4:
                alg = existing_algorithms[alg]
            else:
                warnings.warn(warn_m)
                return

        if alg in existing_algorithms:
            self._algorithm = alg
        else:
            warnings.warn(warn_m)

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):

        warn_m = "Only scaler of 'MinMax', 'Standard' or None or an integer " \
                 "from 0 to 2 supported. Scaler not changed."

        existing_scaler = ['MinMax', 'Standard', None]

        if isinstance(scaler, int):
            if 0 <= scaler <= 4:
                scaler = existing_scaler[scaler]
            else:
                warnings.warn(warn_m)
                return

        if scaler in existing_scaler:
            self._scaler = scaler
        else:
            warnings.warn(warn_m)

    @property
    def n_cluster(self):
        return self._n_cluster

    @n_cluster.setter
    def n_cluster(self, n_cluster):
        if isinstance(n_cluster, int):
            n_cluster = [n_cluster]
        if all(isinstance(n, int) and n > 0 for n in n_cluster):
            self._n_cluster = n_cluster
        else:
            warnings.warn(f'Value for n_cluster of {n_cluster} invalid, '
                          f'n_clusters not changed')

    @property
    def n_dim(self):
        return self._n_dim

    @n_dim.setter
    def n_dim(self, n_dim):
        if isinstance(n_dim, int):
            n_dim = [n_dim]
        if all(isinstance(n, int) and 0 < n <= len(self._features)
               for n in n_dim):
            self._n_dim = n_dim
        else:
            self._n_dim = [len(self._features)]
            warnings.warn(f'n_dim of {n_dim} invalid, use number of features '
                          f'{self._n_dim} as fallback instead')

    def _clear_temp_results(self):
        """Delete all results saved in the temporary results folder"""
        if self.clear_recent:
            clear_directory(RECENT_RESULTS_PATH)

    def _load_data(self):
        """Load data from load_prep_data or merge_pers_features and set self.df

        """
        if self.use_personal:
            df = merge_pers_feature(self.rm_incomplete)
        else:
            df = load_prep_data(rm_first_col=False)

        self.df = df[~df.User.isin(self.excluded_user)]

    def _analyse(self):
        """Perform the actual analysis based on initialized parameters

        self._analyse uses self._clear_temp_results and self._load_data and is
        used by self.perform

        """
        # If clear_recent is True, the recent result folder is cleared:
        self._clear_temp_results()

        # Load the features from file. Will implicitly remove users if
        # self.rm_incomplete is not empty and add the personal features if
        # self.use_personal is True:
        self._load_data()

        # We perform sanity check for the set features to ensure they exist:
        skipped = [f for f in self.features if f not in self.df.columns]
        if skipped:
            warnings.warn(f"Features {skipped} not found in data, skip those!")
        self.features = [f for f in self.features if f in self.df.columns]

        # Clustering can only be performed in dimension less or equal the number
        # of features. We choose the user-friendly way and reduce the dimension
        # if it is too high and warn the user accordingly:
        n_features = len(self.features)
        for idx, dim in enumerate(self.n_dim):
            if dim > n_features:
                self.n_dim[idx] = n_features
                warnings.warn(f"Less features selected than dimension, "
                              f"reduce n_dim to {n_features}")

        # User heatmap provides easy to grasp overview of features for selected
        # users. Will be the first plot generated of option is set:
        if self.user_heatmap:
            user_heatmap(
                features=self._features,
                df=self.df,
                save=self.save_plots,
                dpi=self._image_dpi
            )

        # Correlation heatmap provides easy to grasp overview of correlations
        # between selected features. The heatmap is a colored representation
        # of the symmetric correlation matrix. If plot_significant is True there
        # will be an additional colored table with feature combinations as rows,
        # containing the correlation coefficient as well as the p-value when
        # testing for dependence. By default, all combinations with p < 0.1 will
        # be considered significant and included in this table.
        # Note that the potential number of rows in the table grows
        # quadratically in the number of features, so generating this table
        # for many features can be computational expensive and the result will
        # be cumbersome (it is saved as png):
        if self.cor_heatmap:
            correlation_heatmap(
                features=self._features,
                df=self.df,
                save=self.save_plots,
                dpi=self._image_dpi,
                significant=self.plot_significant
            )

        # Now for the actual clustering. Check the do_cluster from
        # cluster.clustering module for details on parameters and functions:
        if self.clustering:
            kwargs = {
                'n_clusters': self.n_cluster,
                'alg': self._algorithm,
                'scaler': self._scaler,
                'plot': self.feature_plots,
                'pca': self.pca_plot,
                'lineplots': self.user_lineplots,
                'abline': self.abline,
                'c_thresh': self.color_threshold,
                'learntypes': self.learntypes,
                'n_bins': self.n_bins,
                'dpi': self._image_dpi,
                'save': self.save_plots,
                'show_cluster_of': self.show_cluster_of,
                'elbow': self.elbow,
                'silhouette': self.silhouette,
                'plot_center': True,
                'verbose': True
            }
            # Clustering is performed separately with each dimensionality
            # provided by user in self.n_dim list:
            for dim in self._n_dim:
                do_cluster(self.df, self.features, dim=dim, **kwargs)

        # Classification can be performed using scikit learns decision tree
        # algorithm. The feature to classify for is set by self.predict.
        # Further parameters are explained in classifier.classify.do_classify:
        if self.classify:
            kwargs = {
                'predict': self.predict,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_leaves,
                'save': self.save_plots, 'dpi': self._image_dpi
            }
            do_classify(self.df, self.features, **kwargs)

    def raw_lineplots(self):
        """Show raw lineplots prior to the analysis

        """
        learn_types_lineplots(
            save=True,
            learntypes=self.learntypes,
            dpi=300,
            path=''
        )

    def perform(self):
        """Perform any of the functionalities crawl/orga/trafo/prep/analyse

        self.performs defines which of the functionalities is performed.
        For the sake of simplicity user can just overwrite this attribute
        directly. All other attributes of the Analyst will be exploited as well,
        mainly in self._analyse which is called by self.perform (if set).

        """
        # Below are the five functions that can be performed, zipped with there
        # required parameters:
        fcts = (
            (do_crawl, []),
            (do_orga, [self.formats]),
            (do_trafo, [self.formats]),
            (do_prep, [
                self.formats,
                self.prep_feature,
                self.prep_mapper,
                self.max_n_cond_entropy
            ]),
            (self._analyse, [])
        )
        # Any function selected by self.performs is now performed with its
        # parameters as specified above:
        [
            fct(*args)
            for selected, (fct, args) in zip(self.performs.values(), fcts)
            if selected
        ]
