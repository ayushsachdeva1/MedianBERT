import torch

from torch import nn
from transformers import BertConfig
from typing import Union

from model.bert_config import HashBertConfig
from model.embedding import StandardEmbeddings, HashEmbeddings

class BertModel(nn.Module):
    def __init__(self, config: Union[BertConfig, HashBertConfig]):
        super().__init__()
        self.embeddings = HashEmbeddings(config) if type(config) == HashBertConfig else StandardEmbeddings(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size,
            activation=config.hidden_act, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        return self.encoder(x, src_key_padding_mask=~attention_mask.bool())


class BertForMLM(BertModel):
    def __init__(self, config: Union[BertConfig, HashBertConfig]):
        super().__init__(config)
        shared_weights = self.embeddings.tok_embeds.weight
        self.linear = nn.Linear(config.hidden_size, shared_weights.shape[0])
        self.linear.weight = shared_weights

    def forward(self, input_ids, attention_mask):
        return self.linear(super().forward(input_ids, attention_mask))


class HashBertForMLM(BertModel):
    def __init__(self, config: HashBertConfig):
        super().__init__(config)
        # shared_weights = self.embeddings.tok_embeds.weight
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        # self.linear.weight = shared_weights

    def forward(self, input_ids, attention_mask):
        return self.linear(super().forward(input_ids, attention_mask))

class BertForSequenceClassification(BertModel):
    def __init__(self, config: BertConfig, num_labels):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.linear(super().forward(input_ids, token_type_ids, attention_mask)[:, 0, :])  # use CLS token