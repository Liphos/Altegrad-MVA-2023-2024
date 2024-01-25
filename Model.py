import torch
from peft import LoftQConfig, LoraConfig, get_peft_model
from torch import nn
from torch_geometric.nn import BatchNorm, GCNConv, GINConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer


def create_mlp_gin(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 2 * output_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * output_dim, output_dim),
    )


class GINEncoder(nn.Module):
    def __init__(
        self, num_node_features, nout, nhid, graph_hidden_channels, num_layers=5
    ):
        super(GINEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            GINConv(create_mlp_gin(num_node_features, graph_hidden_channels))
        )
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                GINConv(create_mlp_gin(graph_hidden_channels, graph_hidden_channels))
            )
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.norm_layers.append(BatchNorm(graph_hidden_channels))
        self.mol_hidden1 = nn.Linear(num_layers * graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward_gnn(self, graph_batch, apply_global_mean_pool=False):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        xs = []
        for incr, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            x = self.norm_layers[incr](x)
            x = x.relu()
            if apply_global_mean_pool:
                xs.append(global_mean_pool(x, batch))
            else:
                xs.append(x)
        return xs

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        xs = []
        for incr, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            x = self.norm_layers[incr](x)
            x = x.relu()
            xs.append(global_mean_pool(x, batch))
        x = self.mol_hidden1(torch.cat(xs, axis=1)).relu()
        x = self.mol_hidden2(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv4 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv5 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward_gnn(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        return x

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name, use_lora=False):
        super(TextEncoder, self).__init__()
        bert = AutoModel.from_pretrained(model_name)
        if use_lora:
            loftq_config = LoftQConfig(loftq_bits=4)
            lora_config = LoraConfig(
                r=16,
                target_modules=[
                    "dense",
                    "value",
                    "query",
                    "key",
                ],
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                init_lora_weights="loftq",
                loftq_config=loftq_config,
            )
            self.bert = get_peft_model(bert, lora_config)
        else:
            self.bert = bert

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        # print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:, 0, :]


class Model(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        num_layers,
        gnn_type,
        use_lora,
    ):
        super(Model, self).__init__()
        if gnn_type == "gin":
            self.graph_encoder = GINEncoder(
                num_node_features, nout, nhid, graph_hidden_channels, num_layers
            )
        elif gnn_type == "gcn":
            self.graph_encoder = GraphEncoder(
                num_node_features, nout, nhid, graph_hidden_channels
            )
        else:
            raise NotImplementedError("GNN type unknwon")
        self.text_encoder = TextEncoder(model_name, use_lora)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder


def get_model(model_name, gnn_type, use_lora=False):
    return Model(
        model_name=model_name,
        num_node_features=300,
        nout=768,
        nhid=300,
        graph_hidden_channels=300,
        num_layers=5,
        gnn_type=gnn_type,
        use_lora=use_lora,
    )


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
