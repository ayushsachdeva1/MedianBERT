import torch

from model.bert_config import HashBertConfig
from sklearn.utils import murmurhash3_32

def build_lookup_tables(config: HashBertConfig, tokenizer):
    """Map each token ID in the vocabulary to a tuple of h values (coordinates) from h independent hash functions."""
    h, p = len(config.hash_seeds), config.pool_size  # the number of hash functions and the range they map to
    max_token_id = max(tokenizer.get_vocab().values())
    ids_to_coords = torch.zeros(max_token_id + 1, h, dtype=torch.long)
    coords_to_ids = {}
    for token, id in tokenizer.get_vocab().items():
        if id == tokenizer.pad_token_id:
            continue
        coord = tuple((murmurhash3_32(id, seed=x) % p) + (i * p) for i, x in enumerate(config.hash_seeds))
        assert coord not in coords_to_ids  # ensure no collisions
        ids_to_coords[id] = torch.tensor(coord)
        coords_to_ids[coord] = id
    # the padding token has its own coordinate corresponding to the last row in the embedding matrix
    # see HashEmbeddings in models.py
    pad_token_coord = (h * p,) * h
    ids_to_coords[tokenizer.pad_token_id] = torch.tensor(pad_token_coord)
    coords_to_ids[pad_token_coord] = tokenizer.pad_token_id
    return ids_to_coords, coords_to_ids