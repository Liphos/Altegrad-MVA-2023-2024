import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import DataLoader

from dataloader import GraphDatasetInM, TextDataset
from Model import get_model
from tools import load_tokenizer


def load_dataset(tokenizer):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    test_cids_dataset = GraphDatasetInM(root="./data/", gt=gt, split="test_cids")
    test_text_dataset = TextDataset(
        file_path="./data/test_text.txt", tokenizer=tokenizer
    )
    return test_cids_dataset, test_text_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Create Submission",
        description="Create submission from the folder of experiment",
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
    config_model = config["model"]
    hyperparameters = config["hyperparameters"]
    print("loading last model...")
    checkpoint = torch.load(os.path.join(load_folder, "last_model.pt"))

    model = get_model(config_model["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = load_tokenizer(config_model["model_name"])

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    test_cids_dataset, test_text_dataset = load_dataset(tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(
        test_cids_dataset, batch_size=hyperparameters["batch_size"], shuffle=False
    )

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(
        test_text_dataset, batch_size=hyperparameters["batch_size"], shuffle=False
    )
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ):
            text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    solution.to_csv("submission.csv", index=False)
