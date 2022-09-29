# ++++++++++++++        ProSRL Datenanalyse (PySRL)          V1.3.2 +++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---- Import Teil zum Bereitstellen von Funktionen ----------------------------

from pysrl.analysis import Analyst
from pysrl.config.helper import set_root
set_root()
an = Analyst()

# ---- Übersicht der Auswahlmöglichkeiten --------------------------------------

# Mögliche Feature (vgl. features.txt), die in an.features über
# den Index in dieser Liste oder über ihren Namen ausgewählt werden können:

# ['Tests', 'Übungen', 'Kurzaufgaben', 'Beispiele', 'BK_Info', 'Ü_Info',
# 'Übersicht', 'K+B', 'K+Ü', 'K+B+BKI+ÜI', 'K+B+Ü', 'K+B+BKI+Ü+ÜI', 'K+BKI',
# 'B+BKI', 'Ü+ÜI', 'BKI+ÜI', 'K+B+BKI', 'Grund', 'Erweitert', 'Cat_0',
# 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'LearnActs', 'LearnActsPerc',
# 'InfoActs', 'InfoActsPerc', 'TotActs', 'LearnTime', 'LearnTimePerc',
# 'InfoTime', 'InfoTimePerc', 'TotTime', 'LearnTimePerAct', 'InfoTimePerAct',
# 'TotTimePerAct', 'Short_Acts', 'LongActs', 'ShortInfoActsPerc',
# 'ShortLearnActsPerc', 'LongLearnActsPerc', 'LongInfoActsPerc', 'fMean',
# 'f1', 'f2', 'f12', 'f1Amp', 'f2Amp', 'f12Amp' 'PredChance1', 'PredChance2',
# 'PredChance3', 'PredChance4', 'LearnTypeEntropy', 'CategoryEntropy',
# 'CondEntropy1', 'CondEntropy2', 'BeforeTestTime', 'AfterTestTime',
# 'TRegrFit', 'ÜRegrFit', 'BKRegrFit', 'CatTRegrFit', 'CatÜRegrFit',
# 'CatBKRegrFit', 'DiffLowRegrFit', 'DiffMedRegrFit', 'DiffHighRegrFit',
# 'KÜfRegrFit', 'KÜtwrRegrFit', 'KÜrRegrFit', 'TResQuantRegrFit',
# 'TSecSpentRegrFit', 'ÜSecSpentRegrFit', 'BKSecSpentRegrFit',
# 'nResponsTask', 'nTestResQual', 'nTestResQuant', 'ndifficulty',
# 'meanResponsTask', 'meanTestResQual', 'meanTestResQuant', 'meandifficulty',
# 'duplicatesPerc', 'nCat', 'TimePerCat', 'DidTest',
# 'NSt_Pre', 'NSt_Post', 'NSt_Diff', 'RF_Pre', 'RF_Post', 'RF_Diff',
# 'SWJK', 'Note', 'SWK', 'FSK', 'AV', 'ALZO', 'VLZO', 'LZO', 'Inte',
# 'KLO', 'KLW', 'MKL', 'RZM', 'RSA', 'RAK']

# Hier sind Feature in Gruppen zusammengefasst (vgl. feature_types.txt)
# die ebenfalls in an.features über ihren Namen ausgewählt werden können:

# ['learn', 'info', 'type', 'cat', 'level', 'freq', 'per_act',
# 'time', 'acts', 'long_short', 'agg', 'cat_test', 'regression',
# 'mean', 'count', 'other', 'rasch', 'person']

# Das sind die nutzbaren Algorithmen, die mittels an.algorithm über Ihren
# Index oder Namen ausgewählt werden können:

# ['k_means', 'meanshift', 'spectral', 'agglomerative', 'hierarchy']

# Das sind Skalierer, die über Index/Name in an.scaler gewählt werden können:

# ['MinMax', 'Standard', None]

# Das sind die Bildqualitäten, die mit an.image_quality gewählt werden können.
# Reduziere die Qualität bei großen Bildern um Zeit und Platz zu sparen:

# ['low', 'middle', 'high', 'perfect']


# ---- Wahl der Parameter und Einstellungen ------------------------------------
# !! In der Regel müssen nur in diesem Bereich Änderungen vorgenommen werden !!

# Welche pre-processing Schritte sollen durchgeführt werden?
an.performs = {'crawl': 0, 'orga': 0, 'trafo': 0, 'prep': 0, 'analyse': 0}
an.formats = ('csv', )  # ('csv', 'xlsx') or ('csv', ), latter saves ~50% time
raw_lineplots = True  # Create and show raw user lineplots before analysis
# Sollen Personenfeature zur Analyse mit einbezogen werden?
an.use_personal = True
# Sollen Personen weggelassen werden, deren Lernerfolg nicht ermittelt wurde?
an.rm_incomplete = True
# Sollen weitere Personen weggelassen werden (trage IDs hier ein)?
an.excluded_user = []

# Welche Features oder Feature Gruppen sollen berücksichtigt werden?
# Trage in [] Indices von Liste oben oder direkt entsprechende strings ein:
an.features = ['Tests', 'Übungen']

# Welcher Algorithmus und welcher Skalierer soll verwendet werden?
an.algorithm = 'k_means'
an.scaler = 'Standard'

# Setze die Anzahl der Cluster hier (ignoriert für hierarchy, meanshift und
# agglomerative clustering) als Zahl oder in einer Liste []:
an.n_cluster = [2]

# Und die Dimension, wenn diese kleiner als die Zahl der Features ist, werden
# alle Kombinationen von Features in der Anzahl an Dimensionen durchgegangen:
an.n_dim = 2

# Wähle hier welche Analysen durchgeführt werden sollen:
an.clustering = True        # Clustering der User nach gegebenen Features
an.feature_plots = True     # Featureplots (nur in Verbindung mit Clustering)
an.user_heatmap = True      # Heatmaps der Features aufgeteilt nach User
an.cor_heatmap = True       # Korrelationen Heatmap der Feature
an.pca_plot = False         # PCA und ihr Plot (nur in Verb. mit Clustering)
an.user_lineplots = False   # Linepl. zu Clustern (nur in Verb. mit Clustering)
an.elbow = True             # Elbow plots für optimale Cluster-Anzahl
an.silhouette = True        # Silhouette analysis für Güte der Cluster
an.classify = False         # Führe Klassifizierung des predict features durch

# Für das Dendrogram kann hier noch die Grenze der farblichen Cluster-Trennung
# in Einheiten von Standardabweichungen der cluster gesetzt werden:
color_threshold = 2

# Für die Lineplots welche Lerntypen gezeigt werden soll.
# Mögliche Lerntypen für lineplots:
learntypes = ['Tests', 'Ü_Info', 'Übungen', 'Kurzaufgaben', 'BK_Info',
              'Beispiele', 'Übersicht']
# Wähle hier: Eine Liste mit Listen von Indices aus den Lerntypen oben.
# Indices in einer inneren Liste werden zusammengefasst:
an.learntypes = [[0], [1, 2], [3, 4, 5], [6]]

# Für die Klassifizierung noch diese Parameter:
an.predict = 'RF_Diff'  # Feature für das der Entscheidungsbaum erzeugt wird
an.max_depth = 3        # Maximale Tiefe des Entscheidungsbaumes
an.min_leaves = 6       # Minimale Zahl der User in Blättern

# Dann noch weitere Einstellungen:
an.n_bins = 20          # Anzahl der Bins in Histogrammen
an.abline = False       # Winkelhalbierende in Scatterplots
an.clear_recent = True  # Lösche temporären Ergebnisordner vor Analyse
an.save_plots = True    # Speichere Plots der Analyse

an.use_time = True      # Verwende Zeit statt Aktivitäten bei Präparieren
# Diese Parameter werden für das Präparieren verwendet. Bei der Bestimmung von
# bedingten Entropien und Vorhersage-Wahrscheinlichkeiten wird das folgende
# Feature untersucht, mit dem darauffolgenden Mapping seiner Werte. Die maximale
# Anzahl zurück betrachteter Datenpunkte wird mit an.max_n_cond_entropy gesetzt:
an.prep_feature = 'LearnType'
an.prep_mapper = (0, 1, 1, 2, 2, 2, None)
an.max_n_cond_entropy = 1


# ---- Eigentliche Durchführung der Auswertung ---------------------------------

if raw_lineplots:
    an.raw_lineplots()
an.perform()
