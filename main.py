# ++++++++++++++        ProSRL Datenanalyse (PySRL)          V1.3.6 +++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---- Import Teil zum Bereitstellen von Funktionen ----------------------------

from pysrl.config.helper import set_root

set_root()
from pysrl.analysis import Analyst

an = Analyst()

# ---- Wahl der Parameter und Einstellungen ------------------------------------

an.performs = {'crawl': 0, 'orga': 0, 'trafo': 0, 'prep': 0, 'analysis': 1}
an.formats = ('csv',)  # ('csv', 'xlsx') or ('csv', ), latter saves ~50% time
raw_lineplots = False  # Create and show raw user lineplots before analysis

an.use_personal = False
an.rm_incomplete = False
an.excluded_user = []

an.features = ['Beispiele', 'Kurzaufgaben']

an.predict = 'AV'
an.algorithm = 'k_means'
an.scaler = 'Standard'

an.image_quality = 'high'
an.n_cluster = [1]
an.n_dim = 1

an.clustering = True
an.feature_plots = False
an.user_heatmap = False
an.cor_heatmap = False
an.pca_plot = False
an.user_lineplots = True
an.elbow = False
an.silhouette = False
an.classify = False

an.learntypes = [[0], [1], [2], [3], [4], [5]]
an.color_threshold = 2
an.n_bins = 20
an.show_cluster_of = []
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
