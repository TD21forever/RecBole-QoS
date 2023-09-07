# %%

import argparse

from config.configuration import Config
from data.dataset import GeneralDataset, GeneralGraphDataset
from data.utils import data_reparation
from models import NGCF, NeuMF
from recbole.utils import init_seed
from trainer import Trainer
from utils.logger import init_logger

# %% [markdown]
# ### TODO
# - 添加验证集 ✅
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

config = Config(model="NGCF", dataset=args.dataset)


init_logger(config)
init_seed(config["seed"], True)

# dataset = GeneralDataset(config)
dataset = GeneralGraphDataset(config)
train_data, test_data = data_reparation(config, dataset)
model = NGCF(config, dataset).to(config["device"])
trainer = Trainer(config, model)

trainer.fit(train_data, test_data, saved=False, show_progress=True)


# %%
