from pytorch_metric_learning import losses, miners
import torch
from torch import nn


def choice(vct, k):
    perm = torch.randperm(vct.size(0))
    idx = perm[:k]
    return  vct[idx]

class InfoMax():
    def __init__(self, model, scaler, lr, grad_norm = 1):
        self.model = model
        self.l = losses.NTXentLoss()
        self.miner = miners.BatchEasyHardMiner()
        self.optimizer = torch.optim.Adam(model.im_parameters(), lr)
        self.scaler = scaler
        self.scheduler = None
        self.grad_norm = grad_norm
    def _compute_loss(self, embeddings, labels):
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            miner_output = self.miner(embeddings, labels)
            return self.l(embeddings, labels, miner_output)
    def train_step(self, embeddings_3d, embeddings_2d, retain_graph = False):
        examples_per_molecule = embeddings_3d.shape[0]//embeddings_2d.shape[0]
        labels3d = torch.arange(embeddings_3d.shape[0]//examples_per_molecule).unsqueeze(0).repeat(4, 1).T.flatten()
        labels2d = torch.arange(embeddings_2d.shape[0])
        labels = torch.cat([labels3d, labels2d], 0)
        embeddings = torch.cat([embeddings_3d, embeddings_2d], 0)
        self.optimizer.zero_grad()
        l = self._compute_loss(embeddings, labels)
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.im_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return l.item()
    def eval_step(self, embeddings_3d, embeddings_2d, retain_graph = False):
        examples_per_molecule = embeddings_3d.shape[0]//embeddings_2d.shape[0]
        labels3d = torch.arange(embeddings_3d.shape[0]//examples_per_molecule).unsqueeze(0).repeat(4, 1).T.flatten()
        labels2d = torch.arange(embeddings_2d.shape[0])
        labels = torch.cat([labels3d, labels2d], 0)
        embeddings = torch.cat([embeddings_3d, embeddings_2d], 0)
        with torch.no_grad():
            l = self._compute_loss(embeddings, labels)
        return l.item()
    
class MolProp():
    def __init__(self, model, scaler, lr, grad_norm = 1):
        self.model = model
        self.l = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.mp_parameters(), lr)
        self.scaler = scaler
        self.scheduler = None
        self.grad_norm = grad_norm
    def _compute_loss(self, predictions, labels):
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            return self.l(predictions, labels)
    def train_step(self, predictions, labels, retain_graph = False):
        self.optimizer.zero_grad()
        l = self._compute_loss(predictions, labels)
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.mp_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return l.item()
    def eval_step(self, embeddings, labels):
        with torch.no_grad():
            l = self._compute_loss(embeddings, labels)
        return l.item()
    
class Av3D():
    def __init__(self, model, scaler, lr, grad_norm = 1):
        self.model = model
        self.l = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.av3d_parameters(), lr)
        self.scaler = scaler
        self.scheduler = None
        self.grad_norm = grad_norm
    def _compute_loss(self, predictions, labels):
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            return self.l(predictions, labels)
    def train_step(self, predictions, labels, retain_graph = False):
        self.optimizer.zero_grad()
        l = self._compute_loss(predictions, labels)
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.av3d_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return l.item()
    def eval_step(self, embeddings, labels):
        with torch.no_grad():
            l = self._compute_loss(embeddings, labels)
        return l.item()
    
class MolGan():
    def __init__(self, model, scaler, lr, grad_norm = 1, clamp_ps = None):
        self.model = model
        self.l = nn.BCEWithLogitsLoss()
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator_parameters(), lr)
        self.optimizer_generator = torch.optim.Adam(self.model.generator_parameters(), lr)
        self.scaler = scaler
        self.scheduler_discriminator = None
        self.scheduler_generator = None
        self.grad_norm = grad_norm
        self.clamp_ps = clamp_ps
    def _compute_loss(self, predictions, labels):
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            return self.l(predictions, labels)
    def discriminator_train_step(self, real, fake, retain_graph = False):
        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()
        l = self._compute_loss(real, real.new_zeros(real.shape[0])) + self._compute_loss(fake, real.new_ones(real.shape[0]))
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer_discriminator)
        torch.nn.utils.clip_grad_norm_(self.model.discriminator_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer_discriminator)
        self.scaler.update()
        if self.scheduler_discriminator is not None:
            self.scheduler_discriminator.step()
        if self.clamp_ps is not None:
            with torch.no_grad():
                    for p in self.model.discriminator.parameters():
                            p.data.clamp_(-self.clamp_ps, self.clamp_ps)
        return l.item()
    def discriminator_eval_step(self, real, fake):
        with torch.no_grad():
            l = self._compute_loss(real, real.new_zeros(real.shape[0])) + self._compute_loss(fake, real.new_ones(real.shape[0]))
        return l.item()
    def generator_train_step(self, real, fake, retain_graph = False):
        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()
        l = self._compute_loss(fake, fake.new_zeros(fake.shape[0]))
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer_generator)
        torch.nn.utils.clip_grad_norm_(self.model.generator_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer_generator)
        self.scaler.update()
        if self.scheduler_generator is not None:
            self.scheduler_generator.step()
        return l.item()
    def generator_eval_step(self, real, fake):
        with torch.no_grad():
            l = self._compute_loss(fake, real.new_zeros(real.shape[0]))
        return l.item()
    
class MolProp3d():
    def __init__(self, model, scaler, lr, grad_norm = 1):
        self.model = model
        self.l = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.mp3d_parameters(), lr)
        self.scaler = scaler
        self.scheduler = None
        self.grad_norm = grad_norm
    def _compute_loss(self, predictions, labels):
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            return self.l(predictions, labels)
    def train_step(self, predictions, labels, retain_graph = False):
        self.optimizer.zero_grad()
        l = self._compute_loss(predictions, labels)
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.mp_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return l.item()
    def eval_step(self, embeddings, labels):
        with torch.no_grad():
            l = self._compute_loss(embeddings, labels)
        return l.item()

class MDS():
    def __init__(self, model, scaler, lr, grad_norm = 1):
        self.model = model
        self.l = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.mds_parameters(), lr)
        self.scaler = scaler
        self.scheduler = None
        self.grad_norm = grad_norm
    def _compute_loss(self, predictions, labels):
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            return self.l(predictions, labels)
    def train_step(self, predictions, dense_distance, dense_mask, retain_graph = False):
        distances, mask = predictions
        n_nodes = dense_mask.sum()
        non_removed = choice(torch.arange(n_nodes), n_nodes//10)
        dense_mask[non_removed] = True
        self.optimizer.zero_grad()
        l = self._compute_loss(distances[mask][dense_mask], dense_distance[dense_mask])
        self.scaler.scale(l).backward(retain_graph = retain_graph)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.av3d_parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return l.item()
    def eval_step(self, predictions, dense_distance, dense_mask, retain_graph = False):
        distances, mask = predictions
        n_nodes = dense_mask.sum()
        non_removed = choice(torch.arange(n_nodes), n_nodes//10)
        dense_mask[non_removed] = True
        with torch.no_grad():
            l = self._compute_loss(distances[mask][dense_mask], dense_distance[dense_mask])
        return l.item()