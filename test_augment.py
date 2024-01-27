import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from augment import EdgePerturbation, Subgraph, NodeDrop, AttributeMask
from dataloader import AugmentGraphTextDataset, MergeDataset
from Model import get_model, load_tokenizer


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
        print("Training on val set")

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
    # Load configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    # Load model and datasets
    tokenizer = load_tokenizer("nlpie/distil-biobert")
    val_dataset, train_dataset = load_datasets(
        tokenizer=tokenizer,
        model_name="nlpie/distil-biobert",
        training_on_val=True,
    )
    print("device: {}".format(device))

    val_loader = DataLoader(
        val_dataset,
        batch_size=3,
        shuffle=True,
        follow_batch=["x", "x_augment"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=3,
        shuffle=True,
        follow_batch=["x", "x_augment"],
    )

    for batch in tqdm(train_loader):
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

    for k, batch in tqdm(enumerate(val_loader)):
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
