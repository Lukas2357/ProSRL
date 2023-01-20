"""Some constants for the pysrl package"""

from os import path

from ..config.get_root import get_root

# VERY IMPORTANT: Set here the year for the data to analyse:
YEAR = 2021

ROOT = get_root()
INPUT_PATH = path.join(ROOT, 'input', str(YEAR))
DATA_PATH = path.join(ROOT, 'data', str(YEAR))
RESULTS_PATH = path.join(ROOT, 'results', str(YEAR))
RECENT_RESULTS_PATH = path.join(ROOT, 'results', 'recent')

LEARN_TYPES = ['Tests', 'Ü_Info', 'Übungen', 'Kurzaufgaben', 'BK_Info',
               'Beispiele', 'Übersicht']
