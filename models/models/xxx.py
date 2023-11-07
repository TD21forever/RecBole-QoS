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
import torch.nn.functional as F
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from torch import nn
from torch_geometric.utils import dropout_adj

from models.abc_model import GeneralGraphRecommender
from models.embedding import EmbeddingHelper
from models.layers import LightGCNConv, MLPLayers, ResidualLayer, LightGATConv
from utils.enums import EmbeddingModel, EmbeddingType, TemplateType


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
        self.use_improved_prompt = config["use_improved_prompt"]
        self.freeze_embedding = config["freeze_embedding"]
        self.dropout_prob = config["dropout_prob"]
        self.use_bn = config["use_bn"]
        self.node_dropout = config["node_dropout"]
        
        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']

        # define layers and loss

        if not self.use_embedding:
            self.user_embedding = torch.nn.Embedding(
                num_embeddings=self.n_users, embedding_dim=self.latent_dim)
            self.item_embedding = torch.nn.Embedding(
                num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        else:
            self._get_pretrained_embedding()

        embedding_size = self.user_embedding.weight.shape[1]

        self.u_embedding_residual = ResidualLayer(
            embedding_size * 2, 512, dropout=self.dropout_prob, bn=self.use_bn)
        self.i_embedding_residual = ResidualLayer(
            embedding_size * 2, 512, dropout=self.dropout_prob, bn=self.use_bn)

        self.line = [embedding_size * 4] + config["line_layers"]
        self.affine = MLPLayers(
            self.line, dropout=self.dropout_prob, bn=self.use_bn)
        self.output_layer = nn.Linear(self.line[-1], 1)

        # self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.gcn_conv = LightGATConv(dim=embedding_size)
        self.reg_loss = EmbLoss()
        self.loss = nn.L1Loss()

    def _get_pretrained_embedding(self):
        eh = EmbeddingHelper()
        user_invocations = {}
        item_invocations = {}
        for uid in self.dataset.uids_in_inter_feat:
            user_invocations[uid] = self.dataset.inter_data_by_type("user", uid)
        for iid in self.dataset.iids_in_inter_feat:
            item_invocations[iid] = self.dataset.inter_data_by_type("item", iid)
        if self.use_improved_prompt:
            user_embedding = torch.Tensor(eh.fit(EmbeddingType.USER, TemplateType.IMPROVED,
                                        EmbeddingModel.INSTRUCTOR_BGE_SMALL, invocations=user_invocations, auto_save=False))
            item_embedding = torch.Tensor(eh.fit(EmbeddingType.ITEM, TemplateType.IMPROVED,
                                        EmbeddingModel.INSTRUCTOR_BGE_SMALL, invocations=item_invocations, auto_save=False))
        else:
            user_embedding = torch.Tensor(eh.fit(EmbeddingType.USER, TemplateType.STATIC,
                                        EmbeddingModel.INSTRUCTOR_BGE_SMALL, invocations=user_invocations, auto_save=False))
            item_embedding = torch.Tensor(eh.fit(EmbeddingType.ITEM, TemplateType.STATIC,
                                        EmbeddingModel.INSTRUCTOR_BGE_SMALL, invocations=item_invocations, auto_save=False))
        self.user_embedding = torch.nn.Embedding.from_pretrained(
            user_embedding, self.freeze_embedding)
        self.item_embedding = torch.nn.Embedding.from_pretrained(
            item_embedding, self.freeze_embedding)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self,perturbed):
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
            if perturbed:
                random_noise = torch.rand_like(all_embeddings, device=all_embeddings.device)
                all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim = -1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()
        
    def calculate_task_loss(self, interaction):
        uid = interaction[self.USER_ID]
        iid = interaction[self.ITEM_ID]
        label = interaction[self.label]

        user_all_embeddings, item_all_embeddings = self.forward(False)

        u_embeddings = user_all_embeddings[uid]
        u_ego_embeddings = self.user_embedding(uid)
        u_final_embeddings = torch.cat([u_embeddings, u_ego_embeddings], dim = 1)
        
        i_embeddings = item_all_embeddings[iid]
        i_ego_embeddings = self.item_embedding(iid)
        i_final_embeddings = torch.cat([i_embeddings, i_ego_embeddings], dim = 1)

        u_embeddings = self.u_embedding_residual(u_final_embeddings)
        i_embeddings = self.i_embedding_residual(i_final_embeddings)

        u_i_embeddings = torch.cat([u_embeddings, i_embeddings], dim=1)
        
        
        x = self.affine(u_i_embeddings)
        output = self.output_layer(x).squeeze(-1)

        task_loss = self.loss(output, label)

        # calculate regularization Loss
        reg_loss = self.reg_loss(
            u_ego_embeddings, i_ego_embeddings, require_pow=self.require_pow)

        loss = task_loss + self.reg_weight * reg_loss

        return loss
    
    def calculate_loss(self, interaction):

        task_loss = self.calculate_task_loss(interaction)
        return task_loss
        
        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])

        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])

        return task_loss + self.cl_rate * (user_cl_loss + item_cl_loss)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(False)

        u_embeddings = user_all_embeddings[user]
        u_ego_embeddings = self.user_embedding(user)
        u_final_embeddings = torch.cat([u_embeddings, u_ego_embeddings], dim = 1)
        
        i_embeddings = item_all_embeddings[item]
        i_ego_embeddings = self.item_embedding(item)
        i_final_embeddings = torch.cat([i_embeddings, i_ego_embeddings], dim = 1)

        u_embeddings = self.u_embedding_residual(u_final_embeddings)
        i_embeddings = self.i_embedding_residual(i_final_embeddings)
        
        u_i_embeddings = torch.cat([u_embeddings, i_embeddings], dim=1)
        x = self.affine(u_i_embeddings)
        output = self.output_layer(x).squeeze(-1)

        return output
