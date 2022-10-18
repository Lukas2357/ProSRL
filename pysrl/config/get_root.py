import os
import pathlib

FILE_PATH = pathlib.Path(__file__).parent.resolve()


def get_root() -> str:
    """Get the root path from root.txt

    Return:
        str: The root path as string

    """

    m = "Could not specify ROOT. Go to /pysrl/config, open root.txt, enter" \
        " the path of the desired root folder in the first line, save and " \
        "rerun. Alternatively run main.py to specify its location as ROOT."

    with open(os.path.join(FILE_PATH, 'root.txt'), 'r') as f:
        root = f.read().splitlines()[0]

    if not os.path.isdir(root):
        raise FileNotFoundError(m)

    return root
