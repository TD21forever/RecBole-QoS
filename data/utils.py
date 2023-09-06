from data.dataset import RecboleDataset
from data.dataloader import GeneralDataLoader

def data_reparation(config, dataset:RecboleDataset):
    built_dataset = dataset.build()
    train_dataset, test_dataset = built_dataset
    train_data = GeneralDataLoader(train_dataset, config)
    test_data = GeneralDataLoader(test_dataset, config)
    return train_data, test_data

