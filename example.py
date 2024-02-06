from transformers import AutoTokenizer, BertConfig

from model.bert import BertForMLM, HashBertConfig
from utils.lookup import build_lookup_tables


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="./")
    inputs = tokenizer(["hello world, my name is xyz.", "hi"], ["hello", "cheese"],
                       return_tensors="pt", padding=True, truncation=True)
    for row in inputs.input_ids:
        print(tokenizer.convert_ids_to_tokens(row))
    
    print(inputs.input_ids.shape, inputs.attention_mask.shape)

    config1 = BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=128,
    )
    model1 = BertForMLM(config1)
    print("output shape of standard BERT:", model1(**inputs).shape)

    config2 = HashBertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=128,

        hash_seeds=[0, 1, 2, 3],
        pool_size=500,
        agg_func="median",
    )

    model2 = BertForMLM(config2)
    ids_to_coords, coords_to_ids = build_lookup_tables(config2, tokenizer)
    x = ids_to_coords[inputs.input_ids]
    output = model2(x, inputs.attention_mask)
    print("output shape of hash emb BERT:", output.shape)