import argparse
import numpy as np
import torch

from torch.utils import tensorboard
from tqdm import tqdm
import random

from transformers import AutoTokenizer, BertConfig
from datasets import load_dataset


from model import BertModel, BertForMLM
from utils.checkpoint import Checkpointer
from dataset.tokenized_dataset import TokenizedDataset

if __name__ == "__main__":
    # Experiment setup
    parser = argparse.ArgumentParser(description="Bert MLM Training")

    parser.add_argument("--train_log_every_n", type=int, default=100)
    parser.add_argument("--val_log_every_n", type=int, default=1000)
    parser.add_argument("--log_path", type=str, default="./logs/bert_mlm")
    parser.add_argument("--pretrained_file", type=str, default=None)
    parser.add_argument("--ckpt_kept_n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=0.2)

    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=1e-4)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
   
    checkpointer = Checkpointer(path=args.log_path, top_k=args.ckpt_kept_n, keep_min=True)
    writer = tensorboard.SummaryWriter(log_dir=args.log_path)
    
    # Data
    data = load_dataset("wikipedia", "20220301.simple", split="train")
    data = data.remove_columns([col for col in data.column_names if col != "text"])  # only keep the 'text' column
    
    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="./")
    dataset = TokenizedDataset(data, tokenizer)

    train_set, val_set = torch.utils.data.random_split(dataset, [1-args.val_size, args.val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model & Optimizer
    config = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
        )

    model = BertForMLM(config)
    model = checkpointer.load(model, args.pretrained_file)
    
    # Become slower using DataParallel
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)    
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=args.epoch)
    
    losses = []

    for n_epoch in range(args.epoch):
        # Training (GLDv2)
       
        progress_bar = tqdm(train_loader, ncols=120, desc=f"Epoch {n_epoch}", postfix={"loss": ""})
        
        model.train()
        for n_iter, data in enumerate(progress_bar, 1):
            n_data_read = n_epoch * len(train_set) + (n_iter-1) * train_loader.batch_size 
            n_step = n_epoch * len(train_loader) + n_iter

            # Input & Output
            input_ids = data['input_ids'].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids, token_type_ids, attention_mask)
            outputs = outputs.transpose(1, 2)

            # Loss
            loss = loss_fn(outputs, labels)

            # Log
            loss_val = float(loss.detach())

            progress_bar.set_postfix(loss=round(loss_val, 2))

            losses.append(loss_val)

            # Adding to summary
            if n_iter % args.train_log_every_n == 0 or n_iter == len(train_loader):
                writer.add_scalar("Train/loss", sum(losses)/len(losses), n_step)

                losses = []

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation + Save checkpoints
            if n_iter % args.val_log_every_n == 0 or n_iter == len(train_loader):
                val_losses = []

                # Iterate through val_loader
                with torch.no_grad():
                    for n_iter_val, data in enumerate(val_loader, 1):
                        print(f"Validation {n_iter_val}/{len(val_loader)}...", end="\r")

                        input_ids = data['input_ids'].to(device)
                        token_type_ids = data["token_type_ids"].to(device)
                        attention_mask = data['attention_mask'].to(device)
                        labels = data['labels'].to(device)

                        outputs = model(input_ids, token_type_ids, attention_mask)
                        outputs = outputs.transpose(1, 2)
                        
                        loss = loss_fn(outputs, labels)
                        loss_val = float(loss.detach())

                        val_losses.append(loss_val)

                # Log
                writer.add_scalar("Val/loss", sum(val_losses)/len(val_losses), n_step)

                # Save (criterion based on desc_loss of validation set)
                num_digits = len(str(args.epoch * len(train_loader)))
                filename = "delg_" + str(n_data_read).zfill(num_digits) + ".pt"
                checkpointer.save(model, filename, criterion=sum(val_losses)/len(val_losses))

        progress_bar.close()

        # Learning rate schedule
        writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], n_epoch)
        # scheduler.step()