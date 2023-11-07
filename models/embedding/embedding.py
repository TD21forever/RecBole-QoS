import hashlib
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from langchain.embeddings import (HuggingFaceEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models.embedding.template import BasicTempalte, ImprovedTemplate, StaticTemplate
from root import ORIGINAL_DATASET_DIR, RESOURCE_DIR
from utils.enums import *
from tqdm import tqdm

embedding_models = {
    "il": (HuggingFaceInstructEmbeddings, "hkunlp/instructor-large"),
    "e5": (HuggingFaceInstructEmbeddings, "intfloat/e5-large-v2"),
    "ixl": (HuggingFaceInstructEmbeddings, "hkunlp/instructor-xl"),
    "bge-small": (HuggingFaceInstructEmbeddings, "BAAI/bge-small-en"),
    "bge-large": (HuggingFaceInstructEmbeddings, "BAAI/bge-large-en-v1.5"),
    "bge-base": (HuggingFaceInstructEmbeddings, "BAAI/bge-base-en-v1.5")
}


class EmbeddingHelper:

    def __init__(self) -> None:

        self.upath = os.path.join(ORIGINAL_DATASET_DIR, "userlist.txt")
        self.ipath = os.path.join(ORIGINAL_DATASET_DIR, "wslist.txt")
        self.suffix = ".npy"
        self._load_user_and_item()

    @property
    def _user_info_header(self):
        return ["user_id", "ip_address", "country", "ip_number", "AS", "latitude", "longitude"]

    @property
    def _item_info_header(self):
        return ["service_id", "wsdl_address", "provider", "ip_address", "country", "ip_number", "AS", "latitude", "longitude"]

    def _load_user_and_item(self):
        self.user_info = pd.read_csv(
            self.upath, sep="\t", header=0, names=self._user_info_header)
        self.item_info = pd.read_csv(
            self.ipath, sep="\t", header=0, names=self._item_info_header)

    def info2template(self, type_: EmbeddingType, template_type: TemplateType, invocations: Dict[str, List]) -> List[List[str]]:
        if type_ == EmbeddingType.USER:
            info = self.user_info
            id_label = "user_id"
        else:
            info = self.item_info
            id_label = "service_id"

        if template_type == TemplateType.BASIC:
            template_func = BasicTempalte
        elif template_type == TemplateType.IMPROVED:
            template_func = ImprovedTemplate
        elif template_type == TemplateType.STATIC:
            template_func = StaticTemplate
        else:
            raise ValueError
        
        res:List[List[str]] = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=30)
        
        for row_dict in info.to_dict(orient="records"):
            id_ = row_dict[id_label]
            if issubclass(template_func, BasicTempalte):
                template = template_func(row_dict)
            else:
                if type_ == EmbeddingType.USER:
                    template = template_func(
                        type="user", invocations=invocations.get(id_, []), content=row_dict)  # type: ignore
                else:
                    template = template_func(type="item", invocations=invocations.get(id_, []), content=row_dict)
            splits = text_splitter.split_text(str(template))
            res.append(splits)

        return res

    @property
    def embedding_path(self):
        embedding_path = os.path.join(RESOURCE_DIR, "embedding")
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)
        return embedding_path

    def get_models(self, type_: EmbeddingModel) -> Union[HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings]:
        model, model_name = embedding_models[type_.value]
        return model(model_name=model_name)

    def save_embedding(self, embed_data, embed_name):
        saved_path = os.path.join(self.embedding_path, embed_name)
        if not os.path.exists(saved_path):
            np.save(saved_path, embed_data)

    def load_embedding(self, embed_name):
        saved_path = os.path.join(
            self.embedding_path, embed_name + self.suffix)
        if not os.path.exists(saved_path):
            raise FileNotFoundError
        return np.load(saved_path)

    def fit(self, type_: EmbeddingType, template_type: TemplateType, model_type: EmbeddingModel, auto_save=True, *arg, **kwarg):
        combined_string = f"{type_.value}_{template_type.value}_{model_type.value}"
        file_name = hashlib.md5(combined_string.encode()).hexdigest()[:6]
        try:
            return self.load_embedding(file_name)
        except FileNotFoundError as e:
            pass
        model = self.get_models(model_type)
        embeddings = []
        for invocation_text in tqdm(self.info2template(type_, template_type, kwarg["invocations"]), 
              desc="Processing", 
              ncols=75, 
              colour='green', 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            
            embedding = model.embed_documents(invocation_text)
            # embedding = np.mean(model.embed_documents(invocation_text), axis=0)
            
            # embeddings是一个包含所有嵌入的列表
            number_of_embeddings = len(embedding)
            # print(number_of_embeddings)
            # 首个嵌入的权重
            first_weight = 0.4
            # 其余嵌入的总权重
            remaining_weight_total = 0.2
            # 如果只有一个嵌入，它将获得所有的权重
            if number_of_embeddings == 1:
                weights = [1.0]
            elif number_of_embeddings == 2:
                weights = [0.5, 0.5]
            else:
                # 其余每个嵌入的权重
                remaining_weights = [remaining_weight_total / (number_of_embeddings - 2)] * (number_of_embeddings - 2)

                # 构造权重列表
                weights = [first_weight] + remaining_weights + [0.4]
            embedding = np.average(embedding, axis=0, weights=weights)
            embeddings.append(embedding)
        if auto_save:
            self.save_embedding(embeddings, file_name)
        return embeddings


if __name__ == "__main__":
    eh = EmbeddingHelper()
    eh.fit(EmbeddingType.USER, TemplateType.BASIC,
           EmbeddingModel.INSTRUCTOR_LARGE)
