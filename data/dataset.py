import copy
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from data.interaction import Interaction
from root import DATASET_DIR
from torch.utils.data import Dataset as TorchDataset
from utils.enums import FeatType


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

    def _data_processing(self):
        self.feat_name_list = self._build_feat_name_list()

    def _load_data(self, data_dir):
        """加载数据集"""
        dataset_dir = os.path.join(data_dir, self.dataset_name)
        self._load_item_feat(dataset_dir, self.dataset_name)
        self._load_user_feat(dataset_dir, self.dataset_name)
        self._load_inter_feat(dataset_dir, self.dataset_name)

    def _load_feat(self, feat_dir, feat_name):
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
        df.columns = new_columns

        return df

    @property
    def uid_num(self):
        return self._count_unique(self.uid_field)

    @property
    def iid_num(self):
        return self._count_unique(self.iid_field)

    def _count_unique(self, feat_name):
        return len(self.inter_feat[feat_name].unique())

    def _load_inter_feat(self, feat_dir, feat_prefix):
        feat_name = f"{feat_prefix}.inter"
        self.inter_feat = self._load_feat(feat_dir, feat_name)

    def _load_user_feat(self, feat_dir, feat_prefix):
        feat_name = f"{feat_prefix}.user"
        self.user_feat = self._load_feat(feat_dir, feat_name)

    def _load_item_feat(self, feat_dir, feat_prefix):
        feat_name = f"{feat_prefix}.item"
        self.item_feat = self._load_feat(feat_dir, feat_name)

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
        split_ids = self._calcu_split_ids(
            total_cnt, [split_ratio, 1 - split_ratio])
        next_index = [
            range(start, end)
            for start, end in zip([0] + split_ids, split_ids + [total_cnt])
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
