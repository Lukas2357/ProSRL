"""Some constants for the pysrl package"""

from os import path

from .helper import get_root

ROOT = get_root()
INPUT_PATH = path.join(ROOT, 'input')
DATA_PATH = path.join(ROOT, 'data')
RESULTS_PATH = path.join(ROOT, 'results')
RECENT_RESULTS_PATH = path.join(ROOT, 'results', 'recent')
