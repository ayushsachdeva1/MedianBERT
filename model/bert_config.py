import torch

from torch import nn
from transformers import BertConfig
from typing import Union


class HashBertConfig(BertConfig):
    def __init__(self, hash_seeds, pool_size, agg_func, **kwargs):
        super().__init__(**kwargs)
        self.hash_seeds = hash_seeds
        self.pool_size = pool_size
        self.agg_func = agg_func