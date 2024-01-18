import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader

from dataloader import AllGraphDataset
from Model import get_model


def process_batch(batch, mask_rate: float = 0.15):
    batch_len = len(batch.x)
    nb_nodes_to_mask = int(np.round(batch_len * mask_rate))
    indices = torch.randperm(batch_len, device=batch.x.device)[:nb_nodes_to_mask]
    batch.mask_indices = indices
    batch.masked_nodes = batch.x[indices]
    batch.x[indices] = torch.zeros_like(batch.x[indices], device=batch.x.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Load models and data")
    lr = 1e-5
    batch_size = 128
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    dataset = AllGraphDataset(root="./data/", gt=gt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear_model = nn.Sequential(
        nn.Linear(300, 300), nn.ReLU(), nn.Linear(300, 300)
    ).to(device)
    model = get_model("nlpie/distil-biobert", "gin").graph_encoder
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_model = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    optimizer_linear_model = optim.AdamW(
        linear_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    logging.info("Start graph pretraining")
    for epoch in range(100):
        total_loss = 0
        n_iter = 0
        for batch in loader:
            process_batch(batch)
            optimizer_model.zero_grad()
            optimizer_linear_model.zero_grad()
            batch = batch.to(device)
            md = model.forward_gnn(batch)
            out = linear_model(md[batch.mask_indices])
            loss = torch.nn.functional.mse_loss(out, batch.masked_nodes)
            loss.backward()
            optimizer_model.step()
            optimizer_linear_model.step()
            total_loss += loss.item()
            n_iter += 1
            if n_iter % 50 == 0:
                logging.info(f"loss: {total_loss/n_iter}")
        logging.info(f"Epoch: {epoch}, total_loss: {total_loss/n_iter}")
