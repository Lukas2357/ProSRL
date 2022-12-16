# ++++++++++++++        ProSRL Datenanalyse (PySRL)          V1.3.6 +++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---- Import Teil zum Bereitstellen von Funktionen ----------------------------
import pandas as pd

from pysrl.config.helper import set_root

set_root()
from pysrl.analysis import Analyst

all_columns = pd.read_csv('data/all/data_prep_with_personal.csv').columns[3:]

for x in range(1):

    cut = (len(all_columns) + 21) // 2
    columns = [all_columns[x:x+1].to_list() + all_columns[21:cut].to_list(),
               all_columns[x:x+1].to_list() + all_columns[cut:].to_list()]

    for column in columns[:1]:

        an = Analyst()

        # ---- Wahl der Parameter und Einstellungen ------------------------------------

        an.performs = {'crawl': 0, 'orga': 0, 'trafo': 0, 'prep': 0, 'analysis': 1}
        an.formats = ('csv',)  # ('csv', 'xlsx') or ('csv', ), latter saves ~50% time
        raw_lineplots = False  # Create and show raw user lineplots before analysis

        an.features = ['backfrac_cat', 'foolfrac_cat', 'jumpfrac_cat',
                       'backfrac_topic', 'foolfrac_topic', 'jumpfrac_topic']  # column

        an.use_personal = True
        an.rm_incomplete = False
        an.excluded_user = []

        an.predict = 'AV'
        an.algorithm = 'k_means'
        an.scaler = 'Standard'

        an.image_quality = 'high'
        an.n_cluster = [1]
        an.n_dim = 2

        an.clustering = True
        an.feature_plots = True
        an.user_heatmap = False
        an.cor_heatmap = False
        an.pca_plot = False
        an.user_lineplots = False
        an.elbow = False
        an.silhouette = False
        an.classify = False

        an.learntypes = [[0], [1], [2], [3], [4], [5]]
        an.color_threshold = 2
        an.n_bins = 20
        an.show_cluster_of = ['NSt_Pre', 'NSt_Post']
        an.abline = False
        an.clear_recent = False
        an.save_plots = True

        an.max_depth = 3
        an.min_leaves = 6

        an.use_time = True
        an.prep_feature = 'LearnType'
        an.prep_mapper = (0, 1, 1, 2, 2, 2, None)
        an.max_n_cond_entropy = 4

        # ---- Eigentliche Durchführung der Auswertung ---------------------------------

        if raw_lineplots:
            an.raw_lineplots()
        an.perform()