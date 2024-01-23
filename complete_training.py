import logging
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from transformers import AutoTokenizer

from torch.nn import Sequential, Linear, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv

from dataloader import AllGraphDataset, AugmentGraphTextDataset, GraphTextInMDataset
from augment import RWSample, UniformSample
from losses import infoNCE

from Model import get_model

from tqdm import tqdm


# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss
contrastive_loss = infoNCE()

def get_loader(config, tokenizer, model_name):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

    train_dataset = AugmentGraphTextDataset(root="./data/", gt=gt, split="train", transforms=[RWSample(), UniformSample()], tokenizer=tokenizer, model_name=model_name)
    val_dataset = AugmentGraphTextDataset(root="./data/", gt=gt, split="val", tokenizer=tokenizer, model_name=model_name)
                                        #   )#, transforms=[RWSample(), UniformSample()])

    # batch_size = config["batch_size"]
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_anchor', 'x_pos'])
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_anchor', 'x_pos'])

    # return train_loader, val_loader
    return None, None



def step(model, loader, optimizer, type='train'):
    if type == 'train': model.train()
    else: model.eval()

    losses = []
    progress_bar = tqdm(loader)
    for batch in progress_bar:
        batch = batch.to(device)

        graph_embedding, text_embedding = model(batch)

        loss = contrastive_loss(graph_embedding, text_embedding)

        if type == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Training",
        description="Launch training of the model",
        epilog="have to provide a correct model name",
    )
    parser.add_argument("config_yaml", help="path to the config yaml file")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_yaml, "r", encoding="utf-8"))

    model_config = config["model"]
    model_name = model_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])

    train_loader, val_loader = get_loader(config, tokenizer, model_name)