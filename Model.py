from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


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
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        # print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:, 0, :]


class Model(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, graph_hidden_channels
    ):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(
            num_node_features, nout, nhid, graph_hidden_channels
        )
        self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder


def get_model(model_name):
    if model_name == "distilbert-base-uncased":
        return Model(
            model_name=model_name,
            num_node_features=300,
            nout=768,
            nhid=300,
            graph_hidden_channels=300,
        )
    elif model_name == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":
        return Model(
            model_name=model_name,
            num_node_features=300,
            nout=768,
            nhid=300,
            graph_hidden_channels=300,
        )
    else:
        raise NotImplementedError


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
