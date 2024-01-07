# %%

import argparse
import warnings
from logging import getLogger
from torch_geometric.nn import GATConv

from config.configuration import Config
from data.dataset import GeneralDataset, GeneralGraphDataset
from data.utils import data_reparation
from recbole.utils import init_seed, set_color
from trainer import Trainer
from utils.logger import init_logger
from utils.utils import get_flops, get_model

warnings.filterwarnings("ignore")

# %% [markdown]
# ### TODO
# - 添加验证集 ✅
# - NGCF ✅
# - LightGCN ✅
# - 所有代码整体过一遍
# - tensorboard ✅
# - 训练参数整理
# - checkpoint逻辑
# - 日志 ✅
# - 不同显卡切换貌似没起作用
# - 测试流程整理,增加初始化 ✅
# - loss计算方式的优化
# - wandb_project ✅

# %%

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", "-d", type=str, default="wsdream-rt", help="name of datasets"
)

parser.add_argument(
    "--model", "-m", type=str, default="XXX", help="name of models"
)

args, _ = parser.parse_known_args()

config = Config(model=args.model, dataset=args.dataset)
config["train_batch_size"], config["learning_rate"], config["weight_decay"], config["n_layers"] = 512, 0.005, 1e-07, 4

init_logger(config)
init_seed(config["seed"], True)

logger = getLogger()
logger.info(config)

dataset = GeneralGraphDataset(config)
train_data, test_data = data_reparation(config, dataset)

# 必须传入train_data, 必须使用训练集的数据建图
model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
logger.info(model)

flops = get_flops(model, dataset, config["device"], logger)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(
    train_data, test_data, saved=True, show_progress=bool(config["show_progress"]))

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

