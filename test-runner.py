# %%

import argparse
import warnings
from logging import getLogger
import logging
from config.configuration import Config
from data.dataset import GeneralDataset, GeneralGraphDataset
from data.utils import data_reparation
from recbole.utils import init_seed, set_color
from trainer import Trainer
from utils.logger import init_logger
from utils.utils import get_flops, get_model
import hashlib
import time

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


def test(dataset_name, split_rate, use_mte, use_improved_prompt, learning_rate, train_batch_size):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", "-d", type=str, default=dataset_name, help="name of datasets"
    )

    parser.add_argument(
        "--model", "-m", type=str, default="XXX", help="name of models"
    )

    args, _ = parser.parse_known_args()

    config = Config(model=args.model, dataset=args.dataset)
    
    config["split_ratio"] = split_rate
    config["LABEL_FIELD"] = dataset_name.split("-")[1]
    config["use_mte"] = use_mte
    config["use_improved_prompt"] = use_improved_prompt
    config["learning_rate"] = learning_rate
    config["train_batch_size"] = train_batch_size
    
    init_logger(config)
    init_seed(config["seed"], True)

    logger = getLogger(str(int(time.time())))
    logger.info(f"dataset_name:{dataset_name}, split_rate:{split_rate}, use_mte:{use_mte}, use_improved_prompt:{use_improved_prompt}, lr:{learning_rate}, bs:{train_batch_size}")
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
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)




# %%
for lr in [0.005,0.0005]:
    for batch_size in [128,256,512,1024,2048]:
        test("wsdream-rt", 0.05, True, True, lr, batch_size)
