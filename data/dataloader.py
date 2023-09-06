import torch
from torch.utils.data import DataLoader
import numpy as np
class AbstractDataLoader(DataLoader):

    def __init__(self, config, dataset, shuffle=False):
        self.shuffle = shuffle
        self.config = config
        self.sample_size = len(dataset)
        self._dataset = dataset
        self._batch_size = self.step = self.model = None
        self._init_batch_size_and_step()
        index_sampler = None
        self.generator = torch.Generator()
        self.generator.manual_seed(config["seed"])

        super().__init__(
            dataset=self._dataset,
            batch_size=self.step,
            num_workers=config["worker"],
            shuffle=shuffle,
            sampler=index_sampler,
            generator=self.generator,
        )

    def _init_batch_size_and_step(self):
        """Initializing :attr:`step` and :attr:`batch_size`."""
        raise NotImplementedError(
            "Method [init_batch_size_and_step] should be implemented"
        )

    def update_config(self, config):
        """Update configure of dataloader, such as :attr:`batch_size`, :attr:`step` etc.

        Args:
            config (Config): The new config of dataloader.
        """
        self.config = config
        self._init_batch_size_and_step()

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        self._batch_size = batch_size


    def __iter__(self):
        global start_iter
        start_iter = True
        res = super().__iter__()
        start_iter = False
        return res

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)
    
    
class GeneralDataLoader(AbstractDataLoader):

    def __init__(self, config, dataset, shuffle=False):
        self.sample_size = len(dataset)
        super().__init__(config, dataset, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)
