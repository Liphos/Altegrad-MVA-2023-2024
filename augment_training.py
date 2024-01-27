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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from augment import EdgePerturbation, NodeDrop, Subgraph, AttributeMask
from dataloader import AugmentGraphTextDataset, GraphTextInMDataset, MergeDataset
from losses import infoNCE
from Model import get_model, load_tokenizer

CE = torch.nn.CrossEntropyLoss()


class AugmentData(Data):
    def __init__(
        self,
        x,
        edge_index,
        x_augment,
        edge_index_augment,
        input_ids,
        attention_mask,
        description,
    ):
        super().__init__()
        self.x = x
        self.edge_index = edge_index

        self.x_augment = x_augment
        self.edge_index_augment = edge_index_augment

        self.input_ids = input_ids
        self.attention_mask = attention_mask

        self.description = description

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.x.size(0)
        if key == "edge_index_augment":
            return self.x_augment.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


transforms = [NodeDrop(), Subgraph(), AttributeMask(), EdgePerturbation(ratio=0.05)]


def transform_augment(sample):
    tmp = sample.clone()
    if tmp.x.shape[0] > 6:
        choice = np.random.randint(len(transforms))
        transform = transforms[choice]
        tmp = transform(tmp)

    data = AugmentData(
        x=sample.x,
        edge_index=sample.edge_index,
        x_augment=tmp.x,
        edge_index_augment=tmp.edge_index,
        input_ids=sample.input_ids,
        attention_mask=sample.attention_mask,
        description=sample.description,
    )
    return data


def load_datasets(tokenizer: AutoTokenizer, model_name: str, training_on_val: bool):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = AugmentGraphTextDataset(
        root="./data/",
        gt=gt,
        split="val",
        tokenizer=tokenizer,
        model_name=model_name,
        transform=transform_augment,
    )
    train_dataset = AugmentGraphTextDataset(
        root="./data/",
        gt=gt,
        split="train",
        tokenizer=tokenizer,
        model_name=model_name,
        transform=transform_augment,
    )
    if training_on_val:
        logging.info("Training on val set")

        merge_val_dataset = AugmentGraphTextDataset(
            root="./data/",
            gt=gt,
            split="train_on_val",
            tokenizer=tokenizer,
            model_name=model_name,
            transform=transform_augment,
        )

        merged_dataset = MergeDataset(train_dataset, merge_val_dataset)

        return val_dataset, merged_dataset
    return val_dataset, train_dataset


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

    model = get_model(
        model_config["model_name"],
        model_config["gnn_type"],
        model_config["use_lora"] if "use_lora" in model_config else False,
    )
    if "use_lora" in model_config and model_config["use_lora"]:
        logging.info("using lora")

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
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=hyperparameters["step_size"],
        gamma=hyperparameters["gamma"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=16,
        follow_batch=["x", "x_augment"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        follow_batch=["x", "x_augment"],
        num_workers=16,
    )

    print("train")

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

        # train_bar = tqdm(train_loader)
        for batch in train_loader:
            graph_original = Data(
                x=batch.x, edge_index=batch.edge_index, batch=batch.x_batch
            )
            graph_augment = Data(
                x=batch.x_augment,
                edge_index=batch.edge_index_augment,
                batch=batch.x_augment_batch,
            )

            input_ids_1 = batch.input_ids[::2]
            attention_mask_1 = batch.attention_mask[::2]
            input_ids_2 = batch.input_ids[1::2]
            attention_mask_2 = batch.attention_mask[1::2]

            # print(tokenizer.batch_decode(input_ids_1, skip_special_tokens=True))
            # print(tokenizer.batch_decode(input_ids_2, skip_special_tokens=True))

            # print('Graph original:', graph_original)
            # print('Graph augment:', graph_augment)
            # print('Input ids 1:', input_ids_1.shape)
            # print('Input ids 2:', input_ids_2.shape)
            # print('Attention mask 1:', attention_mask_1.shape)
            # print('Attention mask 2:', attention_mask_2.shape)

            graph_embeddings_original = model.graph_encoder(graph_original.to(device))
            # print('Graph embeddings original:', graph_embeddings_original.shape)
            graph_embeddings_augment = model.graph_encoder(graph_augment.to(device))
            # print('Graph embeddings augment:', graph_embeddings_augment.shape)

            text_embeddings_1 = model.text_encoder(
                input_ids_1.to(device), attention_mask_1.to(device)
            )
            # print('Text embeddings 1:', text_embeddings_1.shape)
            text_embeddings_2 = model.text_encoder(
                input_ids_2.to(device), attention_mask_2.to(device)
            )
            # print('Text embeddings 2:', text_embeddings_2.shape)

            loss_1 = contrastive_loss(graph_embeddings_original, text_embeddings_1)
            loss_3 = contrastive_loss(graph_embeddings_original, text_embeddings_2)
            loss_2 = contrastive_loss(graph_embeddings_augment, text_embeddings_1)
            loss_4 = contrastive_loss(graph_embeddings_augment, text_embeddings_2)
            loss_5 = contrastive_loss(
                graph_embeddings_original, graph_embeddings_augment
            )

            # print('Loss 1:', loss_1)
            # print('Loss 2:', loss_2)
            # print('Loss 3:', loss_3)
            # print('Loss 4:', loss_4)
            # print('Loss 5:', loss_5)

            current_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

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

        scheduler.step()
        logging.info(f"scheduler current lr, {scheduler.get_last_lr()}")
        model.eval()
        val_loss = 0

        text_embeddings_list = []
        graph_embeddings_list = []

        for k, batch in enumerate(val_loader):
            graph_original = Data(
                x=batch.x, edge_index=batch.edge_index, batch=batch.x_batch
            )
            graph_augment = Data(
                x=batch.x_augment,
                edge_index=batch.edge_index_augment,
                batch=batch.x_augment_batch,
            )
            input_ids = batch.input_ids[2::3]
            attention_mask = batch.attention_mask[2::3]
            input_ids_1 = batch.input_ids[::3]
            attention_mask_1 = batch.attention_mask[::3]
            input_ids_2 = batch.input_ids[1::3]
            attention_mask_2 = batch.attention_mask[1::3]

            graph_embeddings_original = model.graph_encoder(graph_original.to(device))
            # print('Graph embeddings original:', graph_embeddings_original.shape)
            graph_embeddings_augment = model.graph_encoder(graph_augment.to(device))
            # print('Graph embeddings augment:', graph_embeddings_augment.shape)

            text_embeddings = model.text_encoder(
                input_ids.to(device), attention_mask.to(device)
            )
            text_embeddings_1 = model.text_encoder(
                input_ids_1.to(device), attention_mask_1.to(device)
            )
            # print('Text embeddings 1:', text_embeddings_1.shape)
            text_embeddings_2 = model.text_encoder(
                input_ids_2.to(device), attention_mask_2.to(device)
            )
            # print('Text embeddings 2:', text_embeddings_2.shape)

            loss_1 = contrastive_loss(graph_embeddings_original, text_embeddings_1)
            loss_3 = contrastive_loss(graph_embeddings_original, text_embeddings_2)
            loss_2 = contrastive_loss(graph_embeddings_augment, text_embeddings_1)
            loss_4 = contrastive_loss(graph_embeddings_augment, text_embeddings_2)
            loss_5 = contrastive_loss(
                graph_embeddings_original, graph_embeddings_augment
            )

            current_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

            val_loss += current_loss.item()

            text_embeddings_list.append(text_embeddings.tolist())
            graph_embeddings_list.append(graph_embeddings_original.tolist())

        text_embeddings_list = np.concatenate(text_embeddings_list)
        graph_embeddings_list = np.concatenate(graph_embeddings_list)

        similarity = cosine_similarity(text_embeddings_list, graph_embeddings_list)

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
