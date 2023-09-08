# @Time   : 2022/3/8
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com

r"""
LightGCN
################################################
Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import torch
from models.abc_model import GeneralGraphRecommender
from models.layers import LightGCNConv, ResidualLayer
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from torch import nn
from models.layers import MLPLayers
from models.embedding import EmbeddingHelper
from utils.enums import TemplateType, EmbeddingModel, EmbeddingType
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj


class XXX(GeneralGraphRecommender):
    r"""LightGCN is a GCN-based recommender model, implemented via PyG.
    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly 
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, config, dataset):
        super(XXX, self).__init__(config, dataset)

        self.label = config["LABEL_FIELD"]

        # load parameters info
        
        # int type:the embedding size of lightGCN
        self.latent_dim = config['embedding_size']
        # int type:the layer num of lightGCN
        self.n_layers = config['n_layers']
        # float32 type: the weight decay for l2 normalization
        self.reg_weight = config['reg_weight']
        # bool type: whether to require pow when regularization
        self.require_pow = config['require_pow']
        # boll type: whether to use mte
        self.use_embedding = config["use_mte"] 
        self.freeze_embedding = config["freeze_embedding"]
        self.dropout_prob = config["dropout_prob"]
        self.use_bn = config["use_bn"]
        self.node_dropout = config["node_dropout"]

        # define layers and loss

        if not self.use_embedding:
            self.user_embedding = torch.nn.Embedding(
                num_embeddings=self.n_users, embedding_dim=self.latent_dim)
            self.item_embedding = torch.nn.Embedding(
                num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        else:
            self._get_pretrained_embedding()
        
        
        embedding_size = self.user_embedding.weight.shape[1]
        
        self.u_embedding_residual = ResidualLayer(embedding_size, 512, dropout=self.dropout_prob, bn=self.use_bn)
        self.i_embedding_residual = ResidualLayer(embedding_size, 512, dropout=self.dropout_prob, bn=self.use_bn)
        
        self.line = [embedding_size * 2] + config["line_layers"]
        self.affine = MLPLayers(self.line, dropout=self.dropout_prob, bn=self.use_bn)
        self.output_layer = nn.Linear(self.line[-1], 1)
        
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.reg_loss = EmbLoss()
        self.loss = nn.L1Loss()
        
    def _get_pretrained_embedding(self):
        eh = EmbeddingHelper()
        user_embedding = torch.Tensor(eh.fit(EmbeddingType.USER, TemplateType.BASIC, EmbeddingModel.INSTRUCTOR_BGE_SMALL))
        item_embedding = torch.Tensor(eh.fit(EmbeddingType.ITEM, TemplateType.BASIC, EmbeddingModel.INSTRUCTOR_BGE_SMALL))
        self.user_embedding = torch.nn.Embedding.from_pretrained(user_embedding, self.freeze_embedding)
        self.item_embedding = torch.nn.Embedding.from_pretrained(item_embedding, self.freeze_embedding)

    
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        
        if self.node_dropout == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:
            edge_index, edge_weight = dropout_adj(
                edge_index=self.edge_index, edge_attr=self.edge_weight, p=self.node_dropout)

        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(
                all_embeddings, edge_index, edge_weight)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.dropout_prob)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):

        uid = interaction[self.USER_ID]
        iid = interaction[self.ITEM_ID]
        label = interaction[self.label]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[uid]
        i_embeddings = item_all_embeddings[iid]
        
        u_embeddings = self.u_embedding_residual(u_embeddings)
        i_embeddings = self.i_embedding_residual(i_embeddings)
        
        u_i_embeddings = torch.cat([u_embeddings, i_embeddings], dim=1)
        x = self.affine(u_i_embeddings)
        output = self.output_layer(x).squeeze(-1)

        task_loss = self.loss(output, label)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(uid)
        i_ego_embeddings = self.item_embedding(iid)
        reg_loss = self.reg_loss(
            u_ego_embeddings, i_ego_embeddings, require_pow=self.require_pow)
        
        loss = task_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        u_embeddings = self.u_embedding_residual(u_embeddings)
        i_embeddings = self.i_embedding_residual(i_embeddings)

        u_i_embeddings = torch.cat([u_embeddings, i_embeddings], dim=1)
        x = self.affine(u_i_embeddings)
        output = self.output_layer(x).squeeze(-1)

        return output
