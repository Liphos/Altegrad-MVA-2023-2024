import argparse
import datetime
import logging
import os
import shutil
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataloader import GraphTextInMDataset
from losses import infoNCE
from Model import get_model, load_tokenizer

CE = torch.nn.CrossEntropyLoss()


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def load_datasets(tokenizer: AutoTokenizer, model_name: str, training_on_val: bool):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextInMDataset(
        root="./data/", gt=gt, split="val", tokenizer=tokenizer, model_name=model_name
    )
    train_dataset = GraphTextInMDataset(
        root="./data/", gt=gt, split="train", tokenizer=tokenizer, model_name=model_name
    )
    if training_on_val:
        logging.info("Training on val set")
        data_list_train = [data for data in train_dataset]
        data_list_val = [data for data in val_dataset]
        data_list_train.extend(data_list_val)
        return val_dataset, data_list_train
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
        prog="Training",
        description="Launch training of the model",
        epilog="have to provide a correct model name",
    )
    parser.add_argument("config_yaml", help="path to the config yaml file")
    args = parser.parse_args()
    # Load configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    config = yaml.safe_load(open(args.config_yaml, "r", encoding="utf-8"))
    model_config = config["model"]
    hyperparameters = config["hyperparameters"]
    debug_config = config["debug"]
    training_on_val = (
        True
        if ("training_on_val" in config["debug"] and config["debug"]["training_on_val"])
        else False
    )
    if "loss" not in hyperparameters:
        loss_function = contrastive_loss
    else:
        if hyperparameters["loss"] == "NCE":
            print("Use NCE loss function")
            loss_function = infoNCE()
        else:
            raise ValueError("Loss is not implemented/doesn't exist")
    # Load model and datasets
    tokenizer = load_tokenizer(model_config["model_name"])
    val_dataset, train_dataset = load_datasets(
        tokenizer=tokenizer,
        model_name=model_config["model_name"],
        training_on_val=training_on_val,
    )

    # Create log file
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    output_path = "./outputs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(output_path)
    logging.basicConfig(
        filename=output_path + "log.txt",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )
    shutil.copy(args.config_yaml, output_path + "training.yaml")
    logging.info("device: {}".format(device))

    model = get_model(model_config["model_name"], model_config["gnn_type"])
    # Load pretrained model if specified
    if "gnn_pretrained" in model_config:
        try:
            model.graph_encoder = torch.load(model_config["gnn_pretrained"])
        except:
            model.graph_encoder.load_state_dict(
                torch.load(model_config["gnn_pretrained"])
            )
            logging.info("loaded pretrained gnn")

    if "bert_pretrained" in model_config:
        model.text_encoder.bert.from_pretrained(model_config["bert_pretrained"])
        logging.info("loaded pretrained bert")
    model.train()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters["batch_size"], shuffle=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True
    )

    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    time1 = time.time()
    best_validation_loss = np.inf

    print("Start training...")
    train = False
    for i in range(hyperparameters["nb_epochs"]):
        logging.info(f"-----EPOCH{i + 1}-----")
        model.train()

        for batch in train_loader:
            input_ids, attention_mask, graph_batch = prepare_graph_batch(batch)

            x_graph, x_text = model(
                graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
            )
            current_loss = contrastive_loss(x_graph, x_text)
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            loss += current_loss.item()

            count_iter += 1
            if count_iter % debug_config["print_every"] == 0:
                time2 = time.time()
                logging.info(
                    "Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                        count_iter, time2 - time1, loss / debug_config["print_every"]
                    )
                )
                losses.append(loss)
                loss = 0

        model.eval()
        val_loss = 0

        text_embeddings = []
        graph_embeddings = []

        for batch in val_loader:
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

        y_true = np.diag(np.ones(similarity.shape[0]))

        score = label_ranking_average_precision_score(y_true, similarity)
        logging.info(f"validation score: {score:.4f}")
        print(f"validation score: {score:.4f}")

        logging.info(
            f"-----EPOCH + {i+1} + ----- done.  Validation loss: {val_loss / len(val_loader):.4f}"
        )

        if best_validation_loss > val_loss:
            best_validation_loss = val_loss
            logging.info("validation loss improved saving checkpoint...")
            save_path = os.path.join(output_path, "model" + str(i) + ".pt")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": val_loss,
                    "loss": loss,
                },
                save_path,
            )
            shutil.copy(save_path, output_path + "last_model.pt")
            logging.info("checkpoint saved to: {}".format(save_path))
