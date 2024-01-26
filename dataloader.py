import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm


class GraphTextDataset(Dataset):
    def __init__(
        self, root, gt, split, tokenizer=None, transform=None, pre_transform=None
    ):
        self.tokenizer_name = type(tokenizer).__name__
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(
            os.path.join(self.root, split + ".tsv"), sep="\t", header=None
        )
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ["data_{}.pt".format(cid) for cid in self.cids]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.tokenizer_name}/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:  # get mol2vec features:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            try:
                # On linux
                cid = int(raw_path.split("/")[-1][:-6])
            except:
                # On windows
                cid = int(raw_path.split("\\")[-1][:-6])
            text_input = self.tokenizer(
                [self.description[1][cid]],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length",
                add_special_tokens=True,
            )
            edge_index, x = self.process_graph(raw_path)
            data = Data(
                x=x,
                edge_index=edge_index,
                input_ids=text_input["input_ids"],
                attention_mask=text_input["attention_mask"],
            )

            torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data_{}.pt".format(self.idx_to_cid[idx]))
        )
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(cid)))
        return data


class GraphTextInMDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer=None,
        model_name=None,
        transform=None,
        pre_transform=None,
    ):
        self.tokenizer_name = type(tokenizer).__name__
        self.model_name = model_name
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(
            os.path.join(self.root, split + ".tsv"), sep="\t", header=None
        )
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextInMDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.model_name}/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:  # get mol2vec features:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0
        data_list = []
        for raw_path in self.raw_paths:
            try:
                # On linux
                cid = int(raw_path.split("/")[-1][:-6])
            except:
                # On windows
                cid = int(raw_path.split("\\")[-1][:-6])
            text_input = self.tokenizer(
                [self.description[1][cid]],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length",
                add_special_tokens=True,
            )
            edge_index, x = self.process_graph(raw_path)
            data = Data(
                x=x,
                edge_index=edge_index,
                input_ids=text_input["input_ids"],
                attention_mask=text_input["attention_mask"],
            )
            data_list.append(data)
            i += 1
        self.save(data_list, self.processed_paths[0])


class GraphDataset(Dataset):
    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer=None,
        transform=None,
        pre_transform=None,
        model_name=None,
    ):
        self.tokenizer_name = type(tokenizer).__name__
        self.model_name = model_name
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(
            os.path.join(self.root, split + ".txt"), sep="\t", header=None
        )
        self.cids = self.description[0].tolist()

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ["data_{}.pt".format(cid) for cid in self.cids]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.model_name}/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            try:
                # On linux
                cid = int(raw_path.split("/")[-1][:-6])
            except:
                # On windows
                cid = int(raw_path.split("\\")[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data_{}.pt".format(self.idx_to_cid[idx]))
        )
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(cid)))
        return data

    def get_idx_to_cid(self):
        return self.idx_to_cid


class AllGraphDataset(InMemoryDataset):
    def __init__(self, root, gt, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        super(AllGraphDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if file.endswith(".graph")]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_allgraphs")

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    # edge = np.array(edge)
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            edge_index = np.array(edge_index, dtype=int)
            x = np.array(x)
            return torch.from_numpy(edge_index).T, torch.from_numpy(x)
            # return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        data_list = []
        for raw_path in tqdm(self.raw_paths):
            try:
                # On linux
                cid = int(raw_path.split("/")[-1][:-6])
            except:
                # On windows
                cid = int(raw_path.split("\\")[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        self.save(data_list, osp.join(self.processed_dir, "data.pt"))


class GraphDatasetInM(InMemoryDataset):
    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer=None,
        transform=None,
        pre_transform=None,
        model_name=None,
    ):
        self.model_name = model_name
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(
            os.path.join(self.root, split + ".txt"), sep="\t", header=None
        )
        self.cids = self.description[0].tolist()

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDatasetInM, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.model_name}/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0
        data_list = []
        for raw_path in tqdm(self.raw_paths):
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
            i += 1
        self.save(data_list, osp.join(self.processed_dir, "data.pt"))

    def get_idx_to_cid(self):
        return self.idx_to_cid


class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


class PairData(Data):
    """
    Utility function to return a pair of graphs in dataloader.
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    """

    def __init__(
        self, edge_index_anchor=None, x_anchor=None, edge_index_pos=None, x_pos=None
    ):
        super().__init__()
        self.edge_index_anchor = edge_index_anchor
        self.x_anchor = x_anchor

        self.edge_index_pos = edge_index_pos
        self.x_pos = x_pos

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_anchor":
            return self.x_anchor.size(0)
        if key == "edge_index_pos":
            return self.x_pos.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class AugmentGraphDataset(Dataset):
    def __init__(self, dataset: AllGraphDataset, transforms=None):
        super(AugmentGraphDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        graph_anchor = self.dataset[idx]
        graph_anchor.edge_index = graph_anchor.edge_index.to(torch.int64)
        graph_positive = self.get_positive(graph_anchor)

        # print(graph_positive)

        return PairData(
            graph_anchor.edge_index,
            graph_anchor.x,
            graph_positive.edge_index,
            graph_positive.x,
        )

    def get_positive(self, anchor):
        tmp = anchor.clone()
        if tmp.x.shape[0] > 6:
            # Randomly select a transformation without using np.random.choice
            choice = np.random.randint(len(self.transforms))
            transform = self.transforms[choice]
            tmp = transform(tmp)
        return tmp

class AugmentGraphTextDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer=None,
        model_name=None,
        transform=None,
        pre_transform=None,
    ):
        self.tokenizer_name = type(tokenizer).__name__
        self.model_name = model_name
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(
            os.path.join(self.root, split + "_split.tsv"), sep="\t", header=None
        )
        self.description = self.description.set_index(0).to_dict()

        self.base_description = pd.read_csv(
            os.path.join(self.root, split + ".tsv"), sep="\t", header=None
        )
        self.base_description = self.base_description.set_index(0).to_dict()
        self.cids = list(self.base_description[1].keys())

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(AugmentGraphTextDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"complete_processed_{self.split}_{self.model_name}/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:  # get mol2vec features:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0
        data_list = []
        for raw_path in tqdm(self.raw_paths):
            try:
                # On linux
                cid = int(raw_path.split("/")[-1][:-6])
            except:
                # On windows
                cid = int(raw_path.split("\\")[-1][:-6])
            # print(cid)
            texts = [self.description[1][cid]]
            if isinstance(self.description[2][cid], str):
                texts.append(self.description[2][cid])
            else:
                texts.append(self.base_description[1][cid])

            if self.split == "val":
                texts.append(self.base_description[1][cid])

            text_input = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length",
                add_special_tokens=True,
            )
            edge_index, x = self.process_graph(raw_path)
            data = Data(
                x=x,
                edge_index=edge_index,
                input_ids=text_input["input_ids"],
                attention_mask=text_input["attention_mask"],
                description=texts
            )
            data_list.append(data)
            i += 1

        self.save(data_list, self.processed_paths[0])

class MergeDataset(Dataset):

    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2

        super(MergeDataset, self).__init__()

    def len(self):
        return len(self.ds1) + len(self.ds2)

    def get(self, idx):
        if idx < len(self.ds1):
            return self.ds1[idx]
        else:
            return self.ds2[idx - len(self.ds1)]