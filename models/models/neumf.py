r"""
NeuMF
################################################
Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from models.abc_model import GeneralRecommender
from models.embedding import (EmbeddingHelper, EmbeddingModel, EmbeddingType,
                              TemplateType)
from models.layers import MLPLayers
from torch.nn.init import normal_


class NeuMF(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]

        # load parameters info
        self.mf_embedding_size = config["mf_embedding_size"]
        self.mlp_embedding_size = config["mlp_embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.mf_train = config["mf_train"]
        self.mlp_train = config["mlp_train"]
        self.use_pretrain = config["use_pretrain"]
        self.mf_pretrain_path = config["mf_pretrain_path"]
        self.mlp_pretrain_path = config["mlp_pretrain_path"]
        self.use_embedding = config["use_mte"]
        self.freeze_embedding = config["freeze_embedding"]
        self.apply_init = config["apply_init"]

        self.user_mf_embedding = nn.Embedding(
            self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(
            self.n_items, self.mf_embedding_size)

        # define layers and loss
        if not self.user_mf_embedding:

            self.user_mlp_embedding = nn.Embedding(
                self.n_users, self.mlp_embedding_size)

            self.item_mlp_embedding = nn.Embedding(
                self.n_items, self.mlp_embedding_size)

        else:
            self._get_pretrained_embedding()
            
        embedding_size = self.user_mlp_embedding.weight.shape[1]

        self.mlp_layers = MLPLayers(
            [2 * embedding_size] +
            self.mlp_hidden_size, self.dropout_prob
        )
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.loss = nn.L1Loss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        elif self.apply_init:
            self.apply(self._init_weights)

    def _get_pretrained_embedding(self):
        eh = EmbeddingHelper()
        user_embedding = torch.Tensor(eh.fit(
            EmbeddingType.USER, TemplateType.BASIC, EmbeddingModel.INSTRUCTOR_BGE_SMALL))
        item_embedding = torch.Tensor(eh.fit(
            EmbeddingType.ITEM, TemplateType.BASIC, EmbeddingModel.INSTRUCTOR_BGE_SMALL))
        self.user_mlp_embedding = torch.nn.Embedding.from_pretrained(
            user_embedding, self.freeze_embedding)
        self.item_mlp_embedding = torch.nn.Embedding.from_pretrained(
            item_embedding, self.freeze_embedding)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters."""
        mf = torch.load(self.mf_pretrain_path, map_location="cpu")
        mlp = torch.load(self.mlp_pretrain_path, map_location="cpu")
        mf = mf if "state_dict" not in mf else mf["state_dict"]
        mlp = mlp if "state_dict" not in mlp else mlp["state_dict"]
        self.user_mf_embedding.weight.data.copy_(
            mf["user_mf_embedding.weight"])
        self.item_mf_embedding.weight.data.copy_(
            mf["item_mf_embedding.weight"])
        self.user_mlp_embedding.weight.data.copy_(
            mlp["user_mlp_embedding.weight"])
        self.item_mlp_embedding.weight.data.copy_(
            mlp["item_mlp_embedding.weight"])

        mlp_layers = list(self.mlp_layers.state_dict().keys())
        index = 0
        for layer in self.mlp_layers.mlp_layers:
            if isinstance(layer, nn.Linear):
                weight_key = "mlp_layers." + mlp_layers[index]
                bias_key = "mlp_layers." + mlp_layers[index + 1]
                assert (
                    layer.weight.shape == mlp[weight_key].shape
                ), f"mlp layer parameter shape mismatch"
                assert (
                    layer.bias.shape == mlp[bias_key].shape
                ), f"mlp layer parameter shape mismatch"
                layer.weight.data.copy_(mlp[weight_key])
                layer.bias.data.copy_(mlp[bias_key])
                index += 2

        predict_weight = torch.cat(
            [mf["predict_layer.weight"], mlp["predict_layer.weight"]], dim=1
        )
        predict_bias = mf["predict_layer.bias"] + mlp["predict_layer.bias"]

        self.predict_layer.weight.data.copy_(predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        mf_output, mlp_output = None, None
        if self.mf_train:
            # [batch_size, embedding_size]
            mf_output = torch.mul(user_mf_e, item_mf_e)
        if self.mlp_train:
            mlp_output = self.mlp_layers(
                torch.cat((user_mlp_e, item_mlp_e), -1)
            )  # [batch_size, layers[-1]]
        if mf_output is not None and mlp_output is not None:
            output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        elif mf_output is not None:
            output = self.predict_layer(mf_output)
        elif mlp_output is not None:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError(
                "mf_train and mlp_train can not be False at the same time"
            )
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)
