from torch.utils.data import Dataset
import torch
import random

MODEL_MAX_SEQ_LEN = 512

# Simplified masking of input
def mask_input(input_ids):

    # create random array of floats with equal dims to input_ids
    rand = torch.rand(input_ids.shape)
    
    # mask random 15% where token is not 0 [PAD], 101 [CLS], 102 [SEP] pr 103 [MASK]
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 101) * (input_ids != 102) * (input_ids != 103)

    # Select non-zero elements to be masked
    selection = torch.flatten(mask_arr.nonzero()).tolist()

    # Mask selected elements in input
    input_ids[selection] = 103

    return input_ids

class TokenizedDataset(torch.utils.data.Dataset):
    "This wraps the dataset and tokenizes it, ready for the model"

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.tokenizer(self.dataset[i]["text"], return_tensors="pt", truncation=True, padding="max_length")
        
        labels = data['input_ids'][0].detach().clone()
        attention_mask = data['attention_mask'][0].detach().clone()
        token_type_ids = data['token_type_ids'][0].detach().clone()
        
        input_ids = mask_input(labels.detach().clone())
        
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": labels}