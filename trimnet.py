import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as gnn
import torch_geometric
from torch_geometric.utils import degree
import numpy as np
import sys
sys.path.insert(1, "../BenchCanc/")
from PyDRP.Models.encoders.drugs import TrimNet, RandomSignal


config_model2d = {"init_dim": 79,
                  "edge_dim" : 10,
                  "num_layers" : 6}
config_model3d = {"init_dim": 79}

class Model2D(nn.Module):
    def __init__(self,
                 init_dim = 0,
                 embed_dim = 64,
                 hidden_dim = 1024,
                 num_layers = 1,
                 num_blocks = 1,
                 num_heads=1,
                 edge_dim = 10,
                 random_signal_dim=1,
                 dropout = 0.05,
                **kwargs):
        super().__init__()
        self.gnn_net = TrimNet(in_dim = init_dim, hidden_dim = embed_dim, outdim = 1, depth = num_layers, heads=num_heads, edge_in_dim = edge_dim)
    def forward(self, data):
        return self.gnn_net(data, embeddings="nodes")
    
class Model3D(nn.Module):
    def __init__(self,
                 init_dim = 0,
                 embed_dim = 64,
                 hidden_dim = 1024,
                 num_layers = 1,
                 dropout = 0.01,
                 random_signal_dim=1,
                **kwargs):
        super().__init__()
        # Compute the maximum in-degree in the training data.
        self.r = RandomSignal(random_signal_dim)
        module_list  = [(gnn.GINEConv(nn = nn.Sequential(nn.Linear(init_dim + random_signal_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)), edge_dim=1), "x, edge_index, edge_attr -> x0")]
        for l in range(num_layers -1):
            module_list += [(nn.Dropout(p=dropout), f"x{l} -> x{l+1}")]
            module_list += [(gnn.GINEConv(nn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)), edge_dim=1), f"x{l+1}, edge_index, edge_attr -> x{l+1}")]
            module_list   += [(lambda xi, xii: (F.leaky_relu(xii) + xi)/2, f'x{l}, x{l+1} -> x{l+1}'),]
        self.gnn_net = gnn.Sequential('x, edge_index, edge_attr', module_list)
        self.pool = gnn.aggr.MeanAggregation()
    def forward(self, data, return_embed=False):
        embed = self.gnn_net(x = self.r(data["x"]),
                                     edge_index = data["edge_index"],
                                     edge_attr=data["edge_attr"])
        return embed