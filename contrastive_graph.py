import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from torch.nn import Sequential, Linear, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv

from dataloader import AllGraphDataset, AugmentGraphDataset
from augment import RWSample, UniformSample
from losses import infoNCE

from Model import get_model

from tqdm import tqdm


# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss
contrastive_loss = infoNCE()

def get_loader():
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    dataset = AllGraphDataset(root="./data/", gt=gt)
    augmented_dataset = AugmentGraphDataset(dataset, transforms=[RWSample(), UniformSample()])

    train_size = int(0.9 * len(augmented_dataset))
    test_size = len(augmented_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(augmented_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_anchor', 'x_pos'])
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_anchor', 'x_pos'])

    return train_loader, val_loader


def step(model, loader, optimizer, type='train'):
    if type == 'train': model.train()
    else: model.eval()

    losses = []
    progress_bar = tqdm(loader)
    for batch in progress_bar:
        batch = batch.to(device)

        # print(batch)

        batch.edge_index_anchor = batch.edge_index_anchor.to(torch.int64)
        batch.edge_index_pos = batch.edge_index_pos.to(torch.int64)

        data_anchor = Data(x=batch.x_anchor, edge_index=batch.edge_index_anchor, batch=batch.x_anchor_batch)
        data_pos = Data(x=batch.x_pos, edge_index=batch.edge_index_pos, batch=batch.x_pos_batch)

        readout_anchor = model(data_anchor)
        readout_pos = model(data_pos)

        loss = contrastive_loss(readout_anchor, readout_pos)

        if type == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    return np.mean(losses)


if __name__ == "__main__":

    # Hyperparameters
    decay = 0.01
    lr = 1e-4
    batch_size = 128
    epochs = 100

    train_loader, val_loader = get_loader()
    print("Data loaded")

    # Model
    model_type = "gin"
    model = get_model("nlpie/distil-biobert", model_type).graph_encoder
    model.to(device)

    optimizer_model = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    best_train_loss = np.inf
    best_val_loss = np.inf

    print("Start graph pretraining")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss = step(model, train_loader, optimizer_model, type='train')
        print(f"Train loss: {train_loss}")

        val_loss = step(model, val_loader, optimizer_model, type='val')
        print(f"Validation loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"{model_type}_pretrained_{epoch+1}.pt")
            print("Saved model")