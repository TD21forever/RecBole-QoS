
"""
recbole.model.abstract_recommender
##################################
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from data.dataset import GeneralGraphDataset, RecboleDataset
from recbole.utils import set_color
from utils.enums import FeatSource


class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def __init__(self):
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )
        
    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    def __init__(self, config, dataset: RecboleDataset):
        super(GeneralRecommender, self).__init__()

        self.config = config
        self.dataset = dataset
        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.n_users = dataset.num(self.USER_ID, FeatSource.USER)
        self.n_items = dataset.num(self.ITEM_ID, FeatSource.ITEM)

        # load parameters info
        self.device = config["device"]


class GeneralGraphRecommender(GeneralRecommender):
    """This is an abstract general graph recommender. All the general graph models should implement in this class.
    The base general graph recommender class provide the basic U-I graph dataset and parameters information.
    """

    def __init__(self, config, dataset: GeneralGraphDataset):
        super(GeneralGraphRecommender, self).__init__(config, dataset)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(
            self.device), self.edge_weight.to(self.device)
