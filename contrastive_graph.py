import logging
import time

import numpy as np
import torch
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from augment import RWSample, UniformSample
from dataloader import AllGraphDataset, AugmentGraphDataset
from losses import infoNCE
from Model import get_model

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss

# contrastive_loss = infoNCE()

CE = torch.nn.CrossEntropyLoss()


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def get_loader():
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    dataset = AllGraphDataset(root="./data/", gt=gt)
    train_dataset = AugmentGraphDataset(
        dataset, transforms=[RWSample(), UniformSample()]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        follow_batch=["x_anchor", "x_pos"],
        num_workers=16,
    )

    return train_loader


def step(model, loader, optimizer, type="train"):
    if type == "train":
        model.train()
    else:
        model.eval()

    losses = []
    progress_bar = tqdm(loader)
    for batch in progress_bar:
        batch = batch.to(device)

        # logging.info(batch)

        batch.edge_index_anchor = batch.edge_index_anchor.to(torch.int64)
        batch.edge_index_pos = batch.edge_index_pos.to(torch.int64)

        data_anchor = Data(
            x=batch.x_anchor,
            edge_index=batch.edge_index_anchor,
            batch=batch.x_anchor_batch,
        )
        data_pos = Data(
            x=batch.x_pos, edge_index=batch.edge_index_pos, batch=batch.x_pos_batch
        )

        readout_anchor = model(data_anchor)
        readout_pos = model(data_pos)

        loss = contrastive_loss(readout_anchor, readout_pos)

        if type == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    return np.mean(losses)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Hyperparameters
    decay = 0.01
    lr = 2e-4
    batch_size = 128
    epochs = 200

    train_loader = get_loader()
    logging.info("Data loaded")

    # Model
    model_type = "gin"
    model = get_model("nlpie/distil-biobert", model_type).graph_encoder
    model.to(device)

    optimizer_model = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    logging.info("Start graph pretraining")
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")

        train_loss = step(model, train_loader, optimizer_model, type="train")
        logging.info(f"Train loss: {train_loss}")

        torch.save(model, f"graph_models/CE_{model_type}_pretrained_{epoch+1}.pt")
        logging.info("Saved model")
