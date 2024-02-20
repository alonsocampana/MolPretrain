import torch
from torch import nn
from torch_geometric import nn as gnn
import torch_geometric
from torch_geometric.utils import degree
import numpy as np
from torch.nn import functional as F
import math
from trainers import InfoMax, Av3D, MolGan, MolProp, MolProp3d, MDS
from pytorch_metric_learning import losses, miners
import torchmetrics

def collate_batch(b, sample_ds = 4):
    gs = []
    gs_3d = []
    ds = []
    for g_ in b:
        g = g_.clone()
        ds += [g["dm"].mean(0).flatten()]
        for sample in range(sample_ds):
            el_3d, ea_3d = torch_geometric.utils.coalesce(*torch_geometric.utils.to_undirected(*torch_geometric.utils.remove_self_loops(*torch_geometric.utils.dense_to_sparse(choice(g["dm"], 1)))))
            ea_3d = ea_3d.unsqueeze(1)
            gs_3d += [torch_geometric.data.Data(x = g["x"], edge_index = el_3d, edge_attr=ea_3d)]
    batch2 = torch_geometric.data.Batch.from_data_list(gs_3d, exclude_keys = ["dm"])
    batch2["dense_distance"] = torch.cat(ds, 0)
    return torch_geometric.data.Batch.from_data_list(b, exclude_keys = ["dm"]), batch2

def corrupt_distance_matrix(mat, fraction =0.05):
    M = mat.clone().flatten()
    entries = M.shape[0]
    n_row = int(math.sqrt(entries))
    diagonal = (torch.arange(n_row) * n_row) + torch.arange(n_row)
    sample = torch.distributions.Bernoulli(probs = torch.ones([entries])*fraction).sample().bool()
    sample[diagonal.long()] = 0
    index = torch.arange(entries)[sample]
    col = index%n_row
    row = (index - col)//n_row
    sym = (n_row*col + row).long()
    M[index] = -1
    M[sym] = -1
    corruption_idx = torch.zeros([M.shape[0]])
    corruption_idx[index] = 1
    return M.reshape(n_row, n_row), corruption_idx.bool()

def collate_batch(b, sample_ds = 1, corruption_ds = 0.05):
    gs = []
    gs_3d = []
    ds = []
    ds_ave = []
    masks = []
    for g_ in b:
        g = g_.clone()
        ds_ave += [g["dm"].mean(0).flatten()]
        for sample in range(sample_ds):
            rand_dist = choice(g["dm"], 1)
            ds += [rand_dist.flatten()]
            corrupted, mask = corrupt_distance_matrix(rand_dist, fraction = corruption_ds)
            el_3d, ea_3d = torch_geometric.utils.coalesce(*torch_geometric.utils.to_undirected(*torch_geometric.utils.remove_self_loops(*torch_geometric.utils.dense_to_sparse(corrupted)), reduce = "mean"), reduce="mean")
            ea_3d = ea_3d.unsqueeze(1)
            gs_3d += [torch_geometric.data.Data(x = g["x"], edge_index = el_3d, edge_attr=ea_3d)]
            masks += [mask]
    batch2 = torch_geometric.data.Batch.from_data_list(gs_3d, exclude_keys = ["dm"])
    batch2["dense_distance"] = torch.cat(ds, 0)
    batch2["dense_distance_ave"] = torch.cat(ds_ave, 0)
    batch2["dense_mask"] = torch.cat(masks, 0)
    return torch_geometric.data.Batch.from_data_list(b, exclude_keys = ["dm"]), batch2

def choice(vct, k):
    perm = torch.randperm(vct.size(0))
    idx = perm[:k]
    return  vct[idx]