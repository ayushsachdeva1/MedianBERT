import torch

from torch import nn
from transformers import BertConfig
from typing import Union

from model.bert_config import HashBertConfig


class BaseEmbeddings(nn.Module):
    def __init__(self, config: Union[BertConfig, HashBertConfig]):
        super().__init__()

        assert type(self) != BaseEmbeddings  # this class should not be used directly

        self.seg_embeds = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pos_embeds = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self._position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(dim=0)

    def forward(self, token_embeddings, token_type_ids):
        batch_size, seq_len = token_type_ids.shape
        seg = self.seg_embeds(token_type_ids)
        pos = self.pos_embeds(self._position_ids[:, :seq_len].expand(batch_size, -1).to(token_type_ids.device))

        return self.dropout(self.layer_norm(token_embeddings + seg + pos))


class StandardEmbeddings(BaseEmbeddings):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.tok_embeds = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(self, input_ids, token_type_ids):
        return super().forward(self.tok_embeds(input_ids), token_type_ids)


class HashEmbeddings(BaseEmbeddings):
    def __init__(self, config: HashBertConfig):
        super().__init__(config)
        rows_count = len(config.hash_seeds) * config.pool_size
        # add one row at the end which will always represent padding tokens
        self.tok_embeds = nn.Embedding(rows_count + 1, config.hidden_size, padding_idx=rows_count)
        self.agg_func = getattr(torch, config.agg_func)

    def forward(self, input_ids, token_type_ids):
        tok = self.agg_func(self.tok_embeds(input_ids), dim=-2)
        if self.agg_func == torch.median:
            tok = tok.values  # for some reason torch.median returns a namedtuple
        return super().forward(tok, token_type_ids)