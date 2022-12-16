"""Some constants for the pysrl package"""

from os import path

from ..config.get_root import get_root

# VERY IMPORTANT: Set here the year for the data to analyse:
YEAR = 'all'

ROOT = get_root()
INPUT_PATH = path.join(ROOT, 'input', YEAR)
DATA_PATH = path.join(ROOT, 'data', YEAR)
RESULTS_PATH = path.join(ROOT, 'results', YEAR)
RECENT_RESULTS_PATH = path.join(ROOT, 'results', 'recent')

LEARN_TYPES = ['Tests', 'Ü_Info', 'Übungen', 'Kurzaufgaben', 'BK_Info',
               'Beispiele', 'Übersicht']
