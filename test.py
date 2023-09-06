# %%

import argparse
import copy
import os
from enum import Enum
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from config.configuration import Config
from data.dataloader import GeneralTrainerDataLoader
from data.dataset import GeneralDataset
from data.interaction import Interaction
from data.utils import data_reparation
from models import NeuMF
from root import DATASET_DIR, ROOT_DIR, absolute
from torch.nn.utils import rnn as rnn_utils
from torch.utils.data import Dataset as TorchDataset
from trainer import Trainer
from utils.logger import init_logger

# %% [markdown]
# ### TODO
# - 添加验证集
# - NGCF
# - LightGCN
# - 所有代码整体过一遍
# - tensorboard
# - 训练参数整理
# - checkpoint逻辑
# - 日志 ✅

# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d", type=str, default="wsdream-rt", help="name of datasets"
)

args, _ = parser.parse_known_args()

config = Config(model="NeuMF", dataset=args.dataset)


init_logger(config)

dataset = GeneralDataset(config)
train_data, test_data = data_reparation(config, dataset)
model = NeuMF(config, dataset)
trainer = Trainer(config, model)

trainer.fit(train_data, test_data, saved=False, show_progress=True)


# %%
