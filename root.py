import os
from functools import partial
import datetime
import numpy as np
import torch
import random
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
absolute = partial(os.path.join, ROOT_DIR)

ORIGINAL_DATASET_DIR = os.path.join(ROOT_DIR, 'dataset/original')
RESOURCE_DIR = os.path.join(ROOT_DIR, "resource")
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
