import os
from functools import partial

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
absolute = partial(os.path.join, ROOT_DIR)

ORIGINAL_DATASET_DIR = os.path.join(ROOT_DIR, 'dataset/original')
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
