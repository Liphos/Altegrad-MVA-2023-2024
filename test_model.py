import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader import GraphDatasetInM, GraphTextInMDataset, TextDataset
from Model import get_model, load_tokenizer

CE = torch.nn.CrossEntropyLoss()


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def load_datasets(tokenizer: AutoTokenizer, model_name: str):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextInMDataset(
        root="./data/", gt=gt, split="val", tokenizer=tokenizer, model_name=model_name
    )
    train_dataset = GraphTextInMDataset(
        root="./data/", gt=gt, split="train", tokenizer=tokenizer, model_name=model_name
    )
    return val_dataset, train_dataset


def prepare_graph_batch(batch):
    input_ids = batch.input_ids
    batch.pop("input_ids")
    attention_mask = batch.attention_mask
    batch.pop("attention_mask")
    graph_batch = batch
    return input_ids, attention_mask, graph_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Test Model",
        description="Compute score for the last model of a folder",
        epilog="have to provide a correct folder",
    )
    parser.add_argument("output_folder", help="path to the experiment folder")
    args = parser.parse_args()
    load_folder = args.output_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    config = yaml.safe_load(
        open(os.path.join(load_folder, "training.yaml"), "r", encoding="utf-8")
    )
    model_config = config["model"]

    hyperparameters = config["hyperparameters"]

    tokenizer = load_tokenizer(model_config["model_name"])
    val_dataset, train_dataset = load_datasets(
        tokenizer=tokenizer, model_name=model_config["model_name"]
    )
    checkpoint = torch.load(os.path.join(load_folder, "last_model.pt"))

    model = get_model(model_config["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters["batch_size"], shuffle=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True
    )

    train_loss = 0

    text_embeddings = []
    graph_embeddings = []

    for batch in tqdm(train_loader):
        input_ids, attention_mask, graph_batch = prepare_graph_batch(batch)
        x_graph, x_text = model(
            graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
        )
        current_loss = contrastive_loss(x_graph, x_text)
        train_loss += current_loss.item()

        text_embeddings.append(x_text.tolist())
        graph_embeddings.append(x_graph.tolist())

    text_embeddings = np.concatenate(text_embeddings)
    graph_embeddings = np.concatenate(graph_embeddings)
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    y_true = np.zeros(similarity.shape)
    for i in range(similarity.shape[0]):
        y_true[i, i] = 1

    score = label_ranking_average_precision_score(y_true, similarity)
    print("train loss: ", train_loss)
    print("train score: ", score)

    val_loss = 0

    text_embeddings = []
    graph_embeddings = []

    for batch in tqdm(val_loader):
        input_ids, attention_mask, graph_batch = prepare_graph_batch(batch)
        x_graph, x_text = model(
            graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
        )
        current_loss = contrastive_loss(x_graph, x_text)
        val_loss += current_loss.item()

        text_embeddings.append(x_text.tolist())
        graph_embeddings.append(x_graph.tolist())

    text_embeddings = np.concatenate(text_embeddings)
    graph_embeddings = np.concatenate(graph_embeddings)
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    y_true = np.zeros(similarity.shape)
    for i in range(similarity.shape[0]):
        y_true[i, i] = 1

    score = label_ranking_average_precision_score(y_true, similarity)
    print("val loss: ", val_loss)
    print("val score: ", score)
