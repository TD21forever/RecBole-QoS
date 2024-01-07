from enum import Enum
from typing import Union


class FeatType(Enum):
    Token = (0, "单个离散特征序列")
    TokenSeq = (1, "多个离散特征序列")
    Float = (2, "单个连续特征序列")
    FloatSeq = (3, "多个连续特征序列")

    @classmethod
    def from_code(cls, code: Union[str, int]):
        if isinstance(code, str):
            code = int(code)
        for feat_type in FeatType:
            if feat_type.value[0] == code:
                return feat_type
        return None


class WSDreamDataType(Enum):
    TP_ONLY = (1, "wsdream-tp")
    RT_ONLY = (2, "wsdream-rt")
    TP_AND_RT = (3, "wsdream-all")

    @classmethod
    def from_code(cls, code: int):
        for wsdream_type in WSDreamDataType:
            if wsdream_type.value[0] == code:
                return wsdream_type
        return None


class FeatSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter``
    - ``USER``: Features from ``.user`` 
    - ``ITEM``: Features from ``.item`` 
    """

    INTERACTION = "inter"
    USER = "user"
    ITEM = "item"


class EmbeddingType(Enum):
    USER = "user"
    ITEM = "item"

    @classmethod
    def from_code(cls, code: str):
        for model in cls:
            if model.value == code:
                return model
        return None


class EmbeddingModel(Enum):
    INSTRUCTOR_LARGE = "il"
    INSTRUCTOR_E5 = "e5"
    INSTRUCTOR_XL = "ixl"
    INSTRUCTOR_BGE_SMALL = "bge-small"
    INSTRUCTOR_BGE_LARGE = "bge-large"
    INSTRUCTOR_BGE_BASE = "bge-base"
    BERT = "bert"
    GTE_BASE = "gte-base"
    MINILM = "minilm"

    @classmethod
    def from_code(cls, code: str):
        for model in cls:
            if model.value == code:
                return model
        return None


class TemplateType(Enum):
    BASIC = "basic"
    IMPROVED = "improved"
    STATIC = "static"

    @classmethod
    def from_code(cls, code: str):
        for model in cls:
            if model.value == code:
                return model
        return None
