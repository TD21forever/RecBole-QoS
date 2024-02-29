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

# %%

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", "-d", type=str, default="wsdream-rt", help="name of datasets"
)

parser.add_argument(
    "--model", "-m", type=str, default="NeuMF", help="name of models"
)

args, _ = parser.parse_known_args()

config = Config(model=args.model, dataset=args.dataset)

init_logger(config)
init_seed(config["seed"], True)

logger = getLogger()
logger.info(config)

# 如果是图模型,使用图数据集
# dataset = GeneralGraphDataset(config)

dataset = GeneralDataset(config)
train_data, test_data = data_reparation(config, dataset)

model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
logger.info(model)

flops = get_flops(model, dataset, config["device"], logger)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(
    train_data, test_data, saved=True, show_progress=bool(config["show_progress"]))

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

