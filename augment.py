import random

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph


class Augment:
    """Global class for all augmentation methods"""

    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, data):
        """Call the augmentation method and differentiates between batch and single data"""
        if isinstance(data, Batch):
            dlist = [self.augmentation(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.augmentation(data)

    def augmentation(self, data):
        """Augmentation method to be implemented in child classes"""
        raise NotImplementedError


class NodeDrop(Augment):
    """Node drop augmentation method.
    Randomly drops a percentage of nodes from the graph.
    """

    def augmentation(self, data):
        node_num, _ = data.x.size()
        keep_num = int(node_num * (1 - self.ratio))

        idx_nondrop = torch.randperm(node_num)[:keep_num]
        mask_nondrop = (
            torch.zeros_like(data.x[:, 0]).scatter_(0, idx_nondrop, 1.0).bool()
        )

        edge_index, _ = subgraph(
            mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
        )
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)


class Subgraph(Augment):
    """Subgraph augmentation method.
    Randomly samples a subgraph of the original graph using a random walk.
    An adjacency matrix is created to updqte the reachable neighbors with low computational cost.
    """

    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def augmentation(self, data):
        node_num, _ = data.x.size()
        sub_num = int(node_num * self.ratio * (1 + random.uniform(-0.1, 0.1)))
        edge_index = data.edge_index.detach().clone()

        adj_list = [set() for _ in range(node_num)]
        for i in range(edge_index.size(1)):
            adj_list[edge_index[0][i].item()].add(edge_index[1][i].item())

        init_node = np.random.randint(node_num, size=1)[0]
        idx_sub = set([init_node])
        idx_neigh = adj_list[init_node]

        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                break
            if len(idx_neigh) == 0:
                break

            sample_idx = np.random.randint(len(idx_neigh), size=1)[0]
            sample_node = list(idx_neigh)[sample_idx]
            idx_sub.add(sample_node)
            idx_neigh = idx_neigh.union(adj_list[sample_node])
            idx_neigh = idx_neigh - set(idx_sub)

        idx_sub = torch.LongTensor(list(idx_sub)).to(data.x.device)
        mask_nondrop = torch.zeros_like(data.x[:, 0]).scatter_(0, idx_sub, 1.0).bool()
        edge_index, _ = subgraph(
            mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
        )
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)


class EdgePerturbation(Augment):
    """Edge perturbation augmentation method.
    Randomly adds or removes edges from the graph.
    """

    def __init__(self, ratio=0.05, add=True, drop=True):
        super().__init__(ratio)
        self.add = add
        self.drop = drop

    def augmentation(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * self.ratio)

        edge_index = data.edge_index.detach().clone()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(2, -1).long()

        if self.drop:
            idx_remain = edge_index[
                :, np.random.choice(edge_num, edge_num - perturb_num, replace=False)
            ]

        if self.add:
            idx_add = torch.randint(node_num, (2, perturb_num))
            # Remove self-loop
            idx_add = idx_add[:, idx_add[0] != idx_add[1]]

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return Data(x=data.x, edge_index=new_edge_index)


class AttributeMask(Augment):
    """Attribute mask augmentation method.
    Randomly masks a percentage of node attributes with attributes from other nodes.
    """

    def __init__(self, ratio=0.1, gt=None):
        super().__init__(ratio)
        self.gt = (
            gt
            if gt
            else np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
        )
        self.gt_keys = list(self.gt.keys())

    def augmentation(self, data):
        node_num, _ = data.x.size()
        x = data.x.detach().clone()

        mask_num = int(node_num * self.ratio)
        if mask_num == 0:
            return Data(x=x, edge_index=data.edge_index)
        idx_mask = torch.randperm(node_num)[:mask_num]
        rand_embeddings = torch.randint(len(self.gt_keys), size=(mask_num,))
        x[idx_mask] = torch.tensor(
            [
                self.gt[self.gt_keys[rand_embedding]]
                for rand_embedding in rand_embeddings
            ],
            dtype=torch.float32,
        )

        return Data(x=x, edge_index=data.edge_index)
