import os
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from root import DATASET_DIR, ORIGINAL_DATASET_DIR
from utils.enums import WSDreamDataType

# 把原始的wsdream数据转成原子形式
# https://recbole.io/cn/atomic_files.html


ALL_USER_FIELD = ["[User ID]", "[IP Address]", "[Country]", "[IP No.]", "[AS]", "[Latitude]", "[Longitude]"]
ALL_ITEM_FIELD = ["[Service ID]","[WSDL Address]","[Service Provider]","[IP Address]","[Country]","[IP No.]","[AS]","[Latitude]","[Longitude]"]

class BasicDataConvert:
    def load_user_data(self):
        raise NotImplementedError
    
    def loda_item_data(self):
        raise NotImplementedError
    
    def load_inter_data(self):
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError
    
    
class WSDreamDataConvert(BasicDataConvert):
    
    def __init__(self, wsdream_type:WSDreamDataType) -> None:
        super().__init__()
        
        self.origin_user_field = ["[User ID]", "[Country]", "[AS]"]
        self.user_field = ["user_id", "country", "AS"]
        
        self.origin_item_field = ["[Service ID]", "[Country]", "[AS]"]
        self.item_field = ["item_id", "country", "AS",]
        
        self.inter_field = ["user_id", "item_id"]
        if wsdream_type == WSDreamDataType.RT_ONLY: self.inter_field.append("rt")
        elif wsdream_type == WSDreamDataType.TP_ONLY: self.inter_field.append("tp")
        else: self.inter_field.extend(["rt", "tp"])
            
        self.upath = os.path.join(ORIGINAL_DATASET_DIR, "userlist.txt")
        self.ipath = os.path.join(ORIGINAL_DATASET_DIR, "wslist.txt")
        self.rt_inter = os.path.join(ORIGINAL_DATASET_DIR, "rtMatrix.txt")
        self.tp_inter = os.path.join(ORIGINAL_DATASET_DIR, "tpMatrix.txt")
        
        self.wstype = wsdream_type
        self.output_dir = os.path.join(DATASET_DIR, wsdream_type.value[1])
        
        self.dataset_name = wsdream_type.value[1]
        
        self._load_data()
        
    def _load_data(self):
        self.user_data = self.load_user_data()
        self.item_data = self.loda_item_data()
        self.inter_data = self.load_inter_data()
        for name in ["[Country]", "[AS]"]:
            self._deal_categorical_feat(name)
        
    def _deal_categorical_feat(self, name:str):
        if self.item_data is None and self.user_data is None: self._load_data()
        feat_kinds = []
        if name in self.user_data: feat_kinds.extend(self.user_data[name].unique().tolist())
        if name in self.item_data: feat_kinds.extend(self.item_data[name].unique().tolist())
        feat_kinds = list(set(feat_kinds))
        map_ = {
            feat:idx for idx, feat in enumerate(feat_kinds)
        }
        if name in self.user_data: self.user_data.replace({name:map_}, inplace=True)
        if name in self.item_data: self.item_data.replace({name:map_}, inplace=True)

        
    def _feat_type_wrap(self, type_:str):
        feat_types = []
        if type_ == "user":
            feat_types = [0, 0, 0]
            return list(map(lambda x,y:f'{x}:{y}', self.user_field, feat_types))
        elif type_ == "item":
            feat_types = [0, 0, 0]
            return list(map(lambda x,y:f'{x}:{y}', self.item_field, feat_types))
        else:
            if self.wstype == WSDreamDataType.RT_ONLY or self.wstype == WSDreamDataType.TP_ONLY:
                feat_types = [0, 0, 2]
            else:
                feat_types = [0, 0, 2, 2]
            return list(map(lambda x,y:f'{x}:{y}', self.inter_field, feat_types))
    

    def load_inter_data(self) -> pd.DataFrame:
        rt_path, tp_path = None, None
        if self.wstype == WSDreamDataType.RT_ONLY: rt_path = self.rt_inter
        elif self.wstype == WSDreamDataType.TP_ONLY: tp_path = self.tp_inter
        else: rt_path, tp_path = self.rt_inter, self.tp_inter
        if rt_path and tp_path:
            rt_data = np.loadtxt(rt_path, dtype=np.float64)
            tp_data = np.loadtxt(tp_path, dtype=np.float64)
            rows, cols = np.nonzero(rt_data)
            inter_data = pd.DataFrame({self.inter_field[0]:rows, self.inter_field[1]:cols, self.inter_field[2]:rt_data[rows, cols], self.inter_field[3]: tp_data[rows, cols]})
        else:
            path = self.rt_inter if rt_path else self.tp_inter
            inter_data = np.loadtxt(path, dtype=np.float64)
            rows, cols = np.nonzero(inter_data)
            inter_data = pd.DataFrame({self.inter_field[0]:rows, self.inter_field[1]:cols, self.inter_field[2]:inter_data[rows, cols]})
        return inter_data

    def load_user_data(self) -> pd.DataFrame:
        return pd.read_csv(self.upath, sep="\t", header=0)[self.origin_user_field]
    
    def loda_item_data(self):
        return pd.read_csv(self.ipath, sep="\t", header=0)[self.origin_item_field]
    
    def _convert(self, type_:str):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if type_ == "user":
            data = self.user_data
        elif type_ == "item":
            data = self.item_data
        else:
            data = self.inter_data
        data.columns = self._feat_type_wrap(type_)
        data.to_csv(os.path.join(self.output_dir, f"{self.dataset_name}.{type_}"), index=False)
        
    def fit(self):
        for type_ in ["user", "item", "inter"]:
            self._convert(type_)
        
    
if __name__ == "__main__":
    wc = WSDreamDataConvert(WSDreamDataType.TP_ONLY)
    wc.fit()