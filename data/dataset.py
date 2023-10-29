import copy
import os
from typing import Dict
from torch import Tensor
import numpy as np
import pandas as pd
import torch
from data.interaction import Interaction
from root import DATASET_DIR
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.utils import degree
from utils.enums import FeatSource, FeatType


class RecboleDataset(TorchDataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self._get_preset()
        self._get_field_from_config()
        self._load_data(DATASET_DIR)
        self._data_processing()

    def _get_field_from_config(self):
        """初始化数据集的通用字段"""
        self.dataset_name = self.config['dataset']
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]

        self.split_ratio = self.config['split_ratio']

    def _get_preset(self):
        self.field2type: Dict[str, FeatType] = {}
        self.field2num: Dict[str, Dict[FeatSource, int]] = {}

    def _data_processing(self):
        self.feat_name_list = self._build_feat_name_list()
        self._data_filtering()
        self._reset_index()

    def _load_data(self, data_dir):
        """加载数据集"""
        dataset_dir = os.path.join(data_dir, self.dataset_name)
        self._load_item_feat(dataset_dir, self.dataset_name)
        self._load_user_feat(dataset_dir, self.dataset_name)
        self._load_inter_feat(dataset_dir, self.dataset_name)

    def _data_filtering(self):
        nan_flag = self.config["nan_flag"]
        if nan_flag is None:
            return
        self.inter_feat = self.inter_feat[self.inter_feat[self.label_field] != -1]

    def _reset_index(self):
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            assert isinstance(feat, pd.DataFrame)
            if feat.empty:
                raise ValueError(
                    "feat {} is empty, please check your data".format(
                        feat_name)
                )
            feat.reset_index(drop=True, inplace=True)

    def _load_feat(self, feat_dir, feat_name, source: FeatSource):
        path = os.path.join(feat_dir, feat_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(
            path,
            header=0
        )
        new_columns = []
        for col in df.columns:
            name, dtype = col.split(":")
            dtype = FeatType.from_code(dtype)
            if dtype is None:
                raise ValueError(f"feat type {dtype} not found")
            new_columns.append(name)
            self.field2type[name] = dtype
            if name not in self.field2num:
                self.field2num[name] = {source: self._count_unique(df, col)}
            else:
                self.field2num[name][source] = self._count_unique(df, col)

        df.columns = new_columns

        return df

    def num(self, field, source):
        if field in self.field2num:
            return self.field2num[field].get(source, 0)
        raise ValueError("")

    def _count_unique(self, df: pd.DataFrame, field_name: str):
        if field_name not in df:
            return 0
        return len(df[field_name].unique())

    def _load_inter_feat(self, feat_dir, feat_prefix):
        feat_name = f"{feat_prefix}.inter"
        self.inter_feat = self._load_feat(
            feat_dir, feat_name, FeatSource.INTERACTION)

    def _load_user_feat(self, feat_dir, feat_prefix):
        feat_name = f"{feat_prefix}.user"
        self.user_feat = self._load_feat(feat_dir, feat_name, FeatSource.USER)

    def _load_item_feat(self, feat_dir, feat_prefix):
        feat_name = f"{feat_prefix}.item"
        self.item_feat = self._load_feat(feat_dir, feat_name, FeatSource.ITEM)

    def _build_feat_name_list(self):
        feat_name_list = [
            feat_name
            for feat_name in ["inter_feat", "user_feat", "item_feat"]
            if getattr(self, feat_name, None) is not None
        ]
        return feat_name_list

    def _change_feat_format(self):
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            setattr(self, feat_name, self._dataframe_to_interaction(feat))

    def build(self):
        self._change_feat_format()
        dataset = self.split_by_ratio_without_eval(self.split_ratio)
        return dataset

    def split_by_ratio_without_eval(self, split_ratio: float):
        """分割训练集和测试集"""
        assert 0 < split_ratio < 1
        total_cnt = self.__len__()
        total_ids = np.arange(total_cnt)
        train_ids = np.random.choice(total_ids, int(
            total_cnt * split_ratio), replace=False)
        test_ids = np.setdiff1d(total_ids, train_ids)
        next_index = [
            train_ids, test_ids
        ]
        next_df = [self.inter_feat[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _calcu_split_ids(self, tot, ratios):
        """Given split ratios, and total number, calculate the number of each part after splitting.

        Other than the first one, each part is rounded down.

        Args:
            tot (int): Total number.
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: Number of each part after splitting.
        """
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        for i in range(1, len(ratios)):
            if cnt[0] <= 1:
                break
            if 0 < ratios[-i] * tot < 1:
                cnt[-i] += 1
                cnt[0] -= 1
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def _dataframe_to_interaction(self, data: pd.DataFrame) -> Interaction:
        data_for_tensor = {}
        for col_name in data:
            assert isinstance(col_name, str)
            value = data[col_name].values
            ftype = self.field2type[col_name]
            if ftype == FeatType.Token:
                data_for_tensor[col_name] = torch.LongTensor(value)
            elif ftype == FeatType.Float:
                data_for_tensor[col_name] = torch.FloatTensor(value)
            else:
                raise NotImplementedError(f"feat type {ftype} not implemented")
        return Interaction(data_for_tensor)

    def __len__(self):
        return len(self.inter_feat)

    def __getitem__(self, index, join=False) -> Interaction:
        assert isinstance(self.inter_feat, Interaction)
        df = self.inter_feat[index]
        return self.join(df) if join else df
    
    def inter_data_by_type(self, type: str, id: int):
        """根据类型和id获取交互数据"""
        assert isinstance(self.inter_feat, Interaction)
        if type == "user":
            inter_data = self.inter_feat[self.inter_feat[self.uid_field] == id]
        elif type == "item":
            inter_data = self.inter_feat[self.inter_feat[self.iid_field] == id]
        else:
            raise ValueError(f"type {type} not found")
        user_ids = inter_data[self.uid_field]
        item_ids = inter_data[self.iid_field]
        labels = inter_data[self.label_field]
        assert isinstance(user_ids, Tensor) and isinstance(item_ids, Tensor) and isinstance(labels, Tensor)
        return torch.stack((user_ids, item_ids, labels), 1).tolist()

    @property
    def uids_in_inter_feat(self):
        return self.inter_feat[self.uid_field].unique().tolist()
        
    @property
    def iids_in_inter_feat(self):
        return self.inter_feat[self.iid_field].unique().tolist()

    def join(self, df) -> Interaction:
        """Given interaction feature, join user/item feature into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        return df


class GeneralDataset(RecboleDataset):
    def __init__(self, config):
        super().__init__(config)


class GeneralGraphDataset(RecboleDataset):

    def __init__(self, config):
        super().__init__(config)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        user_num = self.num(self.uid_field, FeatSource.INTERACTION)
        item_num = self.num(self.iid_field, FeatSource.INTERACTION)

        # 构建邻接矩阵 只用训练集数据
        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        deg = degree(edge_index[0], user_num + item_num)

        # 如果度=0，那么就是一个新用户，这个用户没有任何交互，所以这里的处理是将du设置为1
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight
