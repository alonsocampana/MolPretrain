import torch
from torch import nn
from torch_geometric import nn as gnn
import torch_geometric
from torch.nn import functional as F

class AttnDropout(nn.Module):
    def __init__(self,
                 p_dropout = 0.1,
                 **kwargs):
        super().__init__()
        self.id = nn.Identity()
        self.bern = torch.distributions.bernoulli.Bernoulli(torch.Tensor([p_dropout]))
    def forward(self, x):
        x = self.id(x)
        if self.training:
            mask = self.bern.sample([x.shape[0]]).squeeze()
            x[mask.bool()] = float("-inf")
        return x

class MeanAggr(nn.Module):
    def __init__(self,
                 embed_dim,
                 *args,
                 **kwargs):
        super().__init__()
        self.lin_c = nn.LazyLinear(embed_dim)
        self.pool = gnn.aggr.MeanAggregation()
    def forward(self, x, y=None,batch=None):
        if y is None:
            pass
        else:
            x = x + self.lin_c(y).unsqueeze(-2)
        return self.pool(x, batch)

class MaxAggr(nn.Module):
    def __init__(self,
                 embed_dim,
                 *args,
                 **kwargs):
        super().__init__()
        self.lin_c = nn.LazyLinear(embed_dim)
        self.pool = gnn.aggr.MaxAggregation()
    def forward(self, x, y=None,batch=None):
        if y is None:
            pass
        else:
            x = x + self.lin_c(y).unsqueeze(-2)
        return self.pool(x, batch) 

class AttnAggr(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim= 1024,
                 dropout = 0.1,
                 node_dropout = 0.1,
                 use_layernorm = True,
                 *args,
                 **kwargs):
        super().__init__()
        if use_layernorm:
            norm = nn.LayerNorm
        else:
            norm = nn.Identity
        self.lin_c = nn.LazyLinear(embed_dim)
        self.pool = gnn.aggr.AttentionalAggregation(gate_nn = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout1d(dropout),
                                                                            nn.Linear(hidden_dim, 1),
                                                                            AttnDropout(node_dropout)),
                                                    nn=nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout1d(node_dropout),
                                                                            nn.Linear(hidden_dim, embed_dim)))
    def forward(self, x, y=None,batch=None):
        if y is None:
            pass
        else:
            x = x + self.lin_c(y).unsqueeze(-2)
        return self.pool(x, batch)

class DeepsetAggr(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim= 1024,
                 dropout = 0.1,
                 node_dropout = 0.1,
                 use_layernorm = True,
                 *args,
                 **kwargs ):
        super().__init__()
        if use_layernorm:
            norm = nn.LayerNorm
        else:
            norm = nn.Identity
        self.lin_c = nn.LazyLinear(embed_dim)
        self.pool = gnn.aggr.DeepSetsAggregation(local_nn = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout(dropout),
                                                                            nn.Linear(hidden_dim, embed_dim),
                                                                            nn.Dropout1d(node_dropout)),
                                                    global_nn=nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout(dropout),
                                                                            nn.Linear(hidden_dim, embed_dim)))
    def forward(self, x, y = None, batch=None):
        if y is None:
            pass
        else:
            x = (x + self.lin_c(y).unsqueeze(-2))
        return self.pool(x, batch)

class DeepersetAggr(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim= 1024,
                 dropout = 0.1,
                 node_dropout = 0.1,
                 use_layernorm = True, 
                 num_layers = 3,
                 *args,
                 **kwargs):
        super().__init__()
        if use_layernorm:
            norm = nn.LayerNorm
        else:
            norm = nn.Identity
        self.pools = nn.ModuleList()
        self.lin_c = nn.LazyLinear(embed_dim)
        self.pools.append(gnn.aggr.DeepSetsAggregation(local_nn = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout(dropout),
                                                                            nn.Linear(hidden_dim, embed_dim),
                                                                            nn.Dropout1d(node_dropout)),
                                                    global_nn=nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout(dropout),
                                                                            nn.Linear(hidden_dim, embed_dim))))
        for l in range(num_layers-1):
            self.pools.append(gnn.aggr.DeepSetsAggregation(local_nn = nn.Sequential(nn.Linear(embed_dim*2, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout(dropout),
                                                                            nn.Linear(hidden_dim, embed_dim),
                                                                            nn.Dropout1d(node_dropout)),
                                                    global_nn=nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                            norm(hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Dropout(dropout),
                                                                            nn.Linear(hidden_dim, embed_dim))))
    def forward(self, x, y = None, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0)).long()
        for i in range(len(self.pools)):
            if i == 0:
                x_global = self.pools[i](x, batch)
                if y is not None:
                    x_global += self.lin_c(y)
            else:
                x_ = torch.cat([x, x_global.repeat_interleave(torch.bincount(batch), 0)], -1)
                x_global = self.pools[i](x_, batch)
        return x_global
    

class MultiHeadAttnPooling(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_heads = 2,
                 dropout_nodes=0.0,
                 use_layernorm=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.q = nn.Linear(embed_dim, embed_dim*n_heads)
        self.k = nn.Linear(embed_dim, embed_dim*n_heads)
        self.v = nn.Linear(embed_dim, embed_dim*n_heads)
        self.lin = nn.Linear(embed_dim*n_heads, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout_attn = dropout_nodes
    def forward(self, x, y = None, batch=None):
        drugs, nodes_mask = torch_geometric.utils.to_dense_batch(x  =x, batch=batch)
        B, L, _ = drugs.shape
        if y is None:
            cell_lines = x
            S = L
        else:
            cell_lines = y
            S = 1
        q = self.q(cell_lines).view(-1, S, self.n_heads, self.embed_dim).transpose(2, 1)
        nodes_mask = nodes_mask.unsqueeze(1).repeat(1, self.n_heads, 1)
        k = self.k(drugs).view(-1, L, self.n_heads, self.embed_dim).transpose(2, 1)
        v = self.v(drugs).view(-1, L, self.n_heads, self.embed_dim).transpose(2, 1)
        x = F.scaled_dot_product_attention(q, k, v, nodes_mask.unsqueeze(-2), dropout_p = self.dropout_attn)
        x = x.transpose(1, 2).reshape(-1, S, self.n_heads*self.embed_dim).mean(1)
        return self.lin(x)
    
class MultiHeadAttnFFPooling(nn.Module):
    def __init__(self, embed_dim,
                 hidden_dim = 1024,
                 n_heads = 4,
                 dropout = 0.1,
                 dropout_nodes=0.1,
                 use_layernorm=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.q = nn.Linear(embed_dim, embed_dim*n_heads)
        self.k = nn.Linear(embed_dim, embed_dim*n_heads)
        self.v = nn.Linear(embed_dim, embed_dim*n_heads)
        self.lin = nn.Linear(embed_dim*n_heads, embed_dim)
        if use_layernorm:
            norm = nn.LayerNorm
        else:
            norm = nn.Identity
        self.norm = norm(embed_dim)
        self.dropout_attn = dropout_nodes
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                norm(hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, embed_dim))
    def forward(self, x, y = None,batch=None):
        drugs, nodes_mask = torch_geometric.utils.to_dense_batch(x  =x, batch=batch)
        B, L, _ = drugs.shape
        if y is None:
            cell_lines = x
            S = L
        else:
            cell_lines = y
            S = 1
        q = self.q(cell_lines).view(-1, S, self.n_heads, self.embed_dim).transpose(2, 1)
        nodes_mask = nodes_mask.unsqueeze(1).repeat(1, self.n_heads, 1)
        k = self.k(drugs).view(-1, L, self.n_heads, self.embed_dim).transpose(2, 1)
        v = self.v(drugs).view(-1, L, self.n_heads, self.embed_dim).transpose(2, 1)
        x = F.scaled_dot_product_attention(q, k, v, nodes_mask.unsqueeze(-2), dropout_p = self.dropout_attn).transpose(1, 2).reshape(-1, S, self.n_heads*self.embed_dim)
        return self.ff(self.norm(self.lin(x).mean(1)))