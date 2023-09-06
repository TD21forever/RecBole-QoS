from data.dataloader import GeneralEvalDataLoader, GeneralTrainerDataLoader
from data.dataset import RecboleDataset


def data_reparation(config, dataset:RecboleDataset):
    built_dataset = dataset.build()
    train_dataset, test_dataset = built_dataset
    train_data = GeneralTrainerDataLoader(config, train_dataset)
    test_data = GeneralEvalDataLoader(config, test_dataset)
    return train_data, test_data

