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


def test(dataset_name, split_rate, use_mte, use_improved_prompt, learning_rate, train_batch_size, weight_decay, n_layers, mte_model):
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
    config["weight_decay"] = weight_decay
    config["n_layers"] = n_layers
    config["mte_model"] = mte_model
    
    init_logger(config)
    init_seed(config["seed"], True)

    logger = getLogger(str(int(time.time())))
    logger.info(f"dataset_name:{dataset_name}, split_rate:{split_rate}, use_mte:{use_mte}, use_improved_prompt:{use_improved_prompt}, lr:{learning_rate}, bs:{train_batch_size}, wd:{weight_decay}, n_layer:{n_layers}, mte_model:{mte_model}")
    logger.info(config)
    try:
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
        
    except Exception as e:
        logger.info(f"Error: {e}")
    finally:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

# %%
weight_decay = 0
mte_model = "il"
for dataset in ["wsdream-tp"]:
    layers_lis = [4, 5]
    density_lis = [0.1]
    for layers in layers_lis:
        for density in density_lis:
            test(dataset, density, True, True, 0.005, 512, 0, layers, mte_model)
