r"""
NGCF
################################################
Reference:
    Xiang Wang et al. "Neural Graph Collaborative Filtering." in SIGIR 2019.

Reference code:
    https://github.com/xiangwang1223/neural_graph_collaborative_filtering

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.abc_model import GeneralGraphRecommender
from models.layers import BiGNNConv
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from torch_geometric.utils import dropout_adj


class NGCF(GeneralGraphRecommender):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)

        # load dataset info
        self.label = config["LABEL_FIELD"]

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size_list = config['hidden_size_list']
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.reg_weight = config['reg_weight']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.GNNlayers = torch.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.GNNlayers.append(BiGNNConv(input_size, output_size))

        self.loss = nn.L1Loss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        if self.node_dropout == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:
            edge_index, edge_weight = dropout_adj(
                edge_index=self.edge_index, edge_attr=self.edge_weight, p=self.node_dropout)

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(all_embeddings, edge_index, edge_weight)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            # storage output embedding of each layer
            embeddings_list += [all_embeddings]

        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):

        uid = interaction[self.USER_ID]
        iid = interaction[self.ITEM_ID]
        label = interaction[self.label]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[uid]
        i_embeddings = item_all_embeddings[iid]

        output = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        task_loss = self.loss(output, label)

        # L2 regularization of embeddings
        reg_loss = self.reg_loss(u_embeddings, i_embeddings)

        return task_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
