# %%

import argparse
from logging import getLogger

from config.configuration import Config
from data.dataset import GeneralDataset, GeneralGraphDataset
from data.utils import data_reparation
from models import NGCF, XXX, LightGCN, NeuMF
from recbole.utils import init_seed, set_color
from trainer import Trainer
from utils.logger import init_logger
from utils.utils import get_model, get_flops

# %% [markdown]
# ### TODO
# - 添加验证集 ✅
# - NGCF ✅
# - LightGCN ✅
# - 所有代码整体过一遍
# - tensorboard
# - 训练参数整理
# - checkpoint逻辑
# - 日志 ✅
# - 不同显卡切换貌似没起作用
# - 测试流程整理,增加初始化
# - loss计算方式的优化

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


init_logger(config)
init_seed(config["seed"], True)

logger = getLogger()
logger.info(config)

dataset = GeneralGraphDataset(config)
train_data, test_data = data_reparation(config, dataset)

model = get_model(config["model"])(config, dataset).to(config["device"])
logger.info(model)
print("model", model)
flops = get_flops(model, dataset, config["device"], logger)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(
    train_data, test_data, saved=False, show_progress=True)

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")


# %%
