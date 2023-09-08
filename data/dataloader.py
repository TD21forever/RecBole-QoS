import numpy as np
import torch
from data.dataset import RecboleDataset
from data.interaction import Interaction
from torch.utils.data import DataLoader

start_iter = False


class AbstractDataLoader(DataLoader):

    def __init__(self, config, dataset: RecboleDataset, shuffle=False):
        # 常用的有随机采样器：RandomSampler，当dataloader的shuffle参数为True时，系统会自动调用这个采样器

        self.shuffle = shuffle  # shuffle：在每个epoch中对整个数据集data进行shuffle重排，默认为False
        self.config = config
        self.sample_size = len(dataset)

        self._dataset = dataset
        self._batch_size = self.step = self.model = None
        self._init_batch_size_and_step()

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        # this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate `base_seed` for workers. (default: ``None``)
        self.generator = torch.Generator()

        self.generator.manual_seed(config["seed"])

        super().__init__(
            dataset=list(range(self.sample_size)),  # type: ignore
            batch_size=self.step,
            # num_workers：表示开启多少个线程数去加载你的数据，默认为0，代表只使用主进程。
            num_workers=config["worker"],
            shuffle=shuffle,
            generator=self.generator,
            collate_fn=self.collate_fn # type: ignore
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

    def collate_fn(self):
        """Collect the sampled index, and apply neg_sampling or other methods to get the final data."""
        raise NotImplementedError("Method [collate_fn] must be implemented.")

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


class GeneralTrainerDataLoader(AbstractDataLoader):

    def __init__(self, config, dataset, shuffle=False):
        self.sample_size = len(dataset)
        super().__init__(config, dataset, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index) :
        index = np.array(index)
        interaction = self._dataset[index]
        positive_iid = interaction[self.iid_field]
        positive_uid = interaction[self.uid_field]
        return interaction, positive_uid, positive_iid


class GeneralEvalDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, shuffle=False):
        self.sample_size = len(dataset)
        super().__init__(config, dataset, shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        interaction = self._dataset[index]
        positive_iid = interaction[self.iid_field]
        positive_uid = interaction[self.uid_field]
        return interaction, positive_uid, positive_iid
