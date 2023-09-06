
"""
recbole.model.abstract_recommender
##################################
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from recbole.model.layers import FMEmbedding, FMFirstOrderLinear, FLEmbedding
from recbole.utils import ModelType, FeatureSource, FeatureType, set_color


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


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.n_users = dataset.uid_num
        self.n_items = dataset.iid_num

        # load parameters info
        self.device = config["device"]


