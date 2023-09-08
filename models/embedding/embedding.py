import hashlib
import os

import numpy as np
import pandas as pd
from langchain.embeddings import (HuggingFaceEmbeddings,
                                  HuggingFaceInstructEmbeddings)

from models.embedding.template import BasicTempalte
from root import ORIGINAL_DATASET_DIR, RESOURCE_DIR
from utils.enums import *
from typing import List

embedding_models = {
    "il":(HuggingFaceInstructEmbeddings, "hkunlp/instructor-large"),
    "e5":(HuggingFaceInstructEmbeddings, "intfloat/e5-large-v2"),
    "ixl":(HuggingFaceInstructEmbeddings, "hkunlp/instructor-xl"),
    "bge-small":(HuggingFaceInstructEmbeddings, "BAAI/bge-small-en")
}


class EmbeddingHelper:
    
    def __init__(self) -> None:
        
        self.upath = os.path.join(ORIGINAL_DATASET_DIR, "userlist.txt")
        self.ipath = os.path.join(ORIGINAL_DATASET_DIR, "wslist.txt")
        self._load_user_and_item()   
            
    @property
    def _user_info_header(self):
        return ["user_id", "ip_address", "counrty", "ip_number", "AS", "latitude", "longitude"]
    
    @property
    def _item_info_header(self):
        return ["service_id", "wsdl_address", "provider", "ip_address", "country", "ip_number", "AS", "latitude", "longitude"]
    
    def _load_user_and_item(self):
        self.user_info = pd.read_csv(self.upath, sep="\t", header=0, names=self._user_info_header)
        self.item_info = pd.read_csv(self.ipath, sep="\t", header=0, names=self._item_info_header)
        
    def info2template(self, type_:EmbeddingType, template_type:TemplateType)->List[str]:
        if type_ == EmbeddingType.USER:
            info = self.user_info
        else:
            info = self.item_info
        if template_type == TemplateType.BASIC:
            template_func = BasicTempalte
        else:
            raise NotImplementedError
        res = []
        for row_dict in info.to_dict(orient="records"):
            template = template_func(row_dict)  # type: ignore
            res.append(str(template))
        return res
    
    @property
    def embedding_path(self):
        embedding_path = os.path.join(RESOURCE_DIR, "embedding")
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)
        return embedding_path
            
    def get_models(self, type_:EmbeddingModel) -> Union[HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings]:
        model, model_name = embedding_models[type_.value]
        return model(model_name = model_name)
    
    def save_embedding(self, embed_data, embed_name):
        saved_path = os.path.join(self.embedding_path, embed_name)
        if not os.path.exists(saved_path):
            np.save(saved_path, embed_data)
        
    def load_embedding(self, embed_name):
        saved_path = os.path.join(self.embedding_path, embed_name)
        if not os.path.exists(saved_path):
            raise FileNotFoundError
        return np.load(saved_path)
    
    def fit(self, type_:EmbeddingType, template_type: TemplateType, model_type:EmbeddingModel, auto_save = True):
        combined_string = f"{type_.value}_{template_type.value}_{model_type.value}"
        file_name = hashlib.md5(combined_string.encode()).hexdigest()[:6]
        try:
            return self.load_embedding(file_name)
        except FileNotFoundError:
            pass
        model = self.get_models(model_type)
        embeddings = model.embed_documents(self.info2template(type_, template_type))
        if auto_save:
            self.save_embedding(embeddings, file_name)
        return embeddings
    

if __name__ == "__main__":
    eh = EmbeddingHelper()
    eh.fit(EmbeddingType.USER, TemplateType.BASIC, EmbeddingModel.INSTRUCTOR_LARGE)