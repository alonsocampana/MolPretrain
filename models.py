import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as gnn
import torch_geometric
from torch_geometric.utils import degree
import numpy as np

class RandomSignal(nn.Module):
    def __init__(self, signal_dim):
        super().__init__()
        self.signal_dim = signal_dim
    def forward(self, x):
        return torch.cat([x, x.new_tensor(np.random.randn(x.shape[0], self.signal_dim))], 1)
class GNNDimRed(nn.Module):
    def __init__(self, random_signal_dim=16, init_dim = 0, embed_dim=256, n_components = 2):
        self.r = RandomSignal(random_signal_dim)
        self.gru = torch.nn.GRUCell(embed_dim, embed_dim)
        module_list  = [(gnn.GATv2Conv(in_channels=init_dim+random_signal_dim,
                out_channels=embed_dim,
                edge_dim=edge_dim,normalize=False), "x, edge_index, edge_attr -> x0")]
        for l in range(num_layers -1):
            module_list += [(nn.LeakyReLU(), f"x{l} -> x{l+1}")]
            module_list += [(nn.Dropout(p=dropout), f"x{l+1} -> x{l+1}")]
            module_list += [(gnn.GATv2Conv(in_channels=embed_dim,
                out_channels=embed_dim,
                edge_dim=edge_dim,normalize=False), f"x{l+1}, edge_index, edge_attr -> x{l+1}")]
            module_list   += [(lambda xi, xii: (F.leaky_relu(self.gru(xii, xi)) + xi)/2, f'x{l}, x{l+1} -> x{l+1}'),]
        module_list += [(nn.LeakyReLU(), f"x{l} -> x{l+1}")]
        module_list += [(nn.Dropout(p=dropout), f"x{l+1} -> x{l+1}")]
        module_list += [(gnn.GATv2Conv(in_channels=embed_dim,
                               out_channels = n_components,
                                edge_dim=edge_dim,normalize=False), f"x{l+1}, edge_index, edge_attr -> x{l+1}")]
        self.gnn_net = gnn.Sequential('x, edge_index, edge_attr', module_list)
        self.optim = None
    def forward(self, data):
        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            return self.gnn_net(x = self.r(data["x"]),
                                     edge_index = data["edge_index"],
                                     edge_attr=data["edge_attr"])
    

    
class Diffusor(nn.Module):
    def __init__(self, embed_dim, steps=1, num_layers=1, nhead=2):
        super().__init__()
        self.steps = steps
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = embed_dim,
                                                 nhead=nhead,
                                                 batch_first=True), num_layers=num_layers)
        
    def forward(self, x, mask):
        for s in range(self.steps):
            x = (self.layers(x, src_key_padding_mask=mask) + x)/2
        return x

class DistanceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, node_embeddings, batch):
        x_dense, mask = torch_geometric.utils.to_dense_batch(node_embeddings, batch)
        dm = torch.cdist(x_dense, x_dense)
        mask_extended = dm.new_ones(dm.shape)
        mask_extended[~mask] = 0
        mask_extended.transpose(1, 2)[~mask] = 0
        mask_extended = mask_extended.bool()
        return dm, mask_extended
    
class Sampler(nn.Module):
    def __init__(self, embed_dim, eps = 0.0001):
        super().__init__()
        self.eps = eps
        self.mu = nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.std = nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, embed_dim), nn.Softplus())
        
    def forward(self, x, n_samples  = 4):
        mu = self.mu(x)
        std = self.std(x)
        return torch.distributions.Normal(mu, std + self.eps).rsample([n_samples])

class MultiAggr(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_aggr = gnn.aggr.MeanAggregation()
        self.max_aggr = gnn.aggr.MaxAggregation()
        self.min_aggr = gnn.aggr.MinAggregation()
    def forward(self, x, batch):
        return torch.cat([self.mean_aggr(x, batch), self.max_aggr(x, batch), self.min_aggr(x, batch)], 1)

class MolGAN(nn.Module):
    def __init__(self,
                 model2D,
                 model3D,
                 init_dim = 0,
                 embed_dim = 400,
                 hidden_dim = 1024,
                 num_layers_encoder = 8,
                 num_layers_discriminator = 1,
                 edge_dim = 10,
                 n_diffusion_steps=1,
                 n_diffusion_layers = 2,
                 n_diffusion_heads=2,
                 random_signal_dim=32):
        super().__init__()
        self.atom_embedder = model2D
        self.sampler = Sampler(embed_dim = embed_dim)
        self.diffusor = Diffusor(embed_dim,
                                nhead=n_diffusion_heads,
                                num_layers = n_diffusion_layers,
                                steps=n_diffusion_steps)
        self.distance_decoder = DistanceDecoder()
        self.molecular_decoder = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(hidden_dim, 22))
        self.project3d = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(hidden_dim, embed_dim))
        self.mds = nn.Sequential(nn.Linear(embed_dim, 3))
        self.atom3d_embedder = model3D
        self.atom3d_embedder_mlp = nn.Sequential(nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.pooling2d = gnn.aggr.MeanAggregation()
        self.pooling3d = gnn.aggr.MeanAggregation()
        
    def forward(self, data1, data2, 
                n_samples = 4, 
                return_fake=False,
                generate_fake = False,
                predict_features = False,
                predict_features3d = False,
                predict_3d = False,
                return_embeddings = False,
                return_mds = False):
        out = {"logits_fake_graphs":None,
               "logits_real_graphs":None,
               "embeddings_3d":None,
                "embeddings_2d":None,
                "predicted_properties":None,
                "predicted_properties3d":None,
                "predicted_distances":None,
                "reconstructed_geometry": None}
        node_embeddings = self.atom_embedder(data1)
        if return_embeddings:
            out["embeddings_2d"] = self.pooling2d(node_embeddings, data1["batch"])
        #### TODO: REFACTOR FOR BETTER PARAL
        if generate_fake:
            samples = self.sampler(node_embeddings, n_samples)
            fake_graphs = []
            for s in range(n_samples):
                new_s, mask = torch_geometric.utils.to_dense_batch(samples[s], data1["batch"])
                new_s = self.diffusor(new_s, ~mask)
                dist, mask_dist = self.distance_decoder(new_s[mask], data1["batch"])
                for i_g in range(len(dist)):
                    mask_graph = mask_dist[i_g]
                    n_atoms = mask_graph[0].sum()
                    el_3d, ea_3d = torch_geometric.utils.coalesce(
                        *torch_geometric.utils.to_undirected(
                            *torch_geometric.utils.remove_self_loops(
                                *torch_geometric.utils.dense_to_sparse(dist[i_g][mask_graph].reshape(n_atoms, n_atoms)))))
                    ea_3d = ea_3d.unsqueeze(1)
                    fake_graphs += [torch_geometric.data.Data(x = data1[i_g]["x"], edge_index = el_3d, edge_attr=ea_3d)]
            if return_fake:
                return fake_graphs
            out["logits_fake_graphs"] = self.atom3d_embedder_mlp(self.atom3d_embedder(torch_geometric.data.Batch.from_data_list(fake_graphs)))
        if generate_fake or return_embeddings or return_mds or predict_features3d:
            embed3d = self.atom3d_embedder(data2)
            gs3d_embed = self.pooling3d(embed3d, data2["batch"])
            out["logits_real_graphs"] = self.atom3d_embedder_mlp(gs3d_embed)
            out["embeddings_3d"] = gs3d_embed
            out["predicted_properties3d"] = self.molecular_decoder(gs3d_embed)
            out["reconstructed_geometry"] = self.distance_decoder(self.mds(embed3d), data2["batch"])
        if predict_features:
            out["predicted_properties"] = self.molecular_decoder(self.pooling2d(node_embeddings, data1["batch"]))
        if predict_3d:
            out["predicted_distances"] = self.distance_decoder(node_embeddings, data1["batch"])
            #### TODO: REFACTOR FOR BETTER PARAL
        return out
    def discriminator_parameters(self):
        return list(self.atom3d_embedder.parameters()) + list(self.atom3d_embedder_mlp.parameters()) + list(self.pooling3d.parameters())
    def generator_parameters(self):
        return list(self.atom_embedder.parameters()) + list(self.sampler.parameters()) + list(self.diffusor.parameters()) + list(self.distance_decoder.parameters()) + list(self.pooling2d.parameters())
    def im_parameters(self):
        return list(self.atom_embedder.parameters()) + list(self.atom3d_embedder.parameters()) + list(self.pooling3d.parameters()) + list(self.pooling2d.parameters())
    def mp_parameters(self):
        return list(self.atom_embedder.parameters()) + list(self.molecular_decoder.parameters()) + list(self.pooling2d.parameters())
    def mp3d_parameters(self):
        return list(self.atom3d_embedder.parameters()) + list(self.molecular_decoder.parameters()) + list(self.pooling3d.parameters())
    def av3d_parameters(self):
        return list(self.atom_embedder.parameters()) + list(self.distance_decoder.parameters()) + list(self.project3d.parameters()) + list(self.pooling2d.parameters())
    def mds_parameters(self):
        return list(self.atom3d_embedder.parameters()) + list(self.distance_decoder.parameters()) + list(self.mds.parameters()) + list(self.pooling3d.parameters())