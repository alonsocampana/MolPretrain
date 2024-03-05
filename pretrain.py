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
import argparse
from models import *
from utils import *
from functools import lru_cache
import optuna

@lru_cache(maxsize = None)
def get_data():
    train = torch.load("/mnt/mlshare/alonsocampana/molecules/compiled_train.pt")
    test = torch.load("/mnt/mlshare/alonsocampana/molecules/compiled_test.pt")
    return train, test


def pretrain_model(config, final_callback = None, epoch_callback = None):
    model_name = config["model_file"].replace('.py', '')
    target_model = __import__(model_name)
    train, test = get_data()
    lr = config["learning_rate"]
    pretraining = config["pretraining"]
    config_model2d = target_model.config_model2d
    config_model3d = target_model.config_model3d
    config_model2d = {**config, **config_model2d}
    config_model3d = {**config, **config_model3d}
    
    model = MolGAN(model2D = target_model.Model2D(train_dataset= train,**config_model2d),
                   model3D = target_model.Model3D(**config_model3d),
                   init_dim=79,
                   embed_dim=config["embed_dim"],
                   edge_dim = 10,
                   hidden_dim = config["hidden_dim"],
                   n_diffusion_steps = 4)
    device = torch.device(config["device"])
    model.to(device)
    if "properties3d" in pretraining:
        batch_size = 640
        n_samples = 1
    if "mds" in pretraining:
        batch_size = 640
        n_samples = 1
    if "infomax" in pretraining:
        batch_size = 260
        n_samples = 4
    if "properties2d" in pretraining:
        batch_size = 320
        n_samples = 1
    if "dist" in pretraining:
        batch_size = 640
        n_samples = 1
    if "gan" in pretraining:
        batch_size = 160
        n_samples = 4

    
    train_loader = torch.utils.data.DataLoader(train,
                                               collate_fn=lambda x: collate_batch(x, sample_ds=n_samples),
                                               batch_size = batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test,
                                              collate_fn=lambda x: collate_batch(x, sample_ds=n_samples),
                                              batch_size = batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers = 4)
    scaler = torch.cuda.amp.GradScaler()
    infomax = InfoMax(model, scaler, lr=lr)
    molprop = MolProp(model, scaler, lr=lr)
    molprop3D = MolProp(model, scaler, lr=lr)
    av3d = Av3D(model, scaler, lr=lr)
    molgan = MolGan(model, scaler, lr=lr)
    mds = MDS(model, scaler, lr=lr)
    pretraining = config["pretraining"]
    ls_train = {}
    ls_test = {}
    for epoch in range(config["num_epochs"]):
        model.train()
        ls_train_epoch = {}
        ls_test_epoch = {}
        for x, (b1, b2) in enumerate(train_loader):
            
            if "properties3d" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out = model(b1.to(device), b2.to(device), n_samples = 4,
                                predict_features3d = True)
                l = molprop.train_step(out["predicted_properties3d"], b1["y"])
                if "properties3d" not in ls_train_epoch.keys():
                    ls_train_epoch["properties3d"] = []
                ls_train_epoch["properties3d"] += [l]
            if "mds" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out = model(b1.to(device), b2.to(device), n_samples = 4, 
                                return_mds=True)
                l = mds.train_step(out["reconstructed_geometry"], b2["dense_distance"], b2["dense_mask"])
                if "mds" not in ls_train_epoch.keys():
                    ls_train_epoch["mds"] = []
                ls_train_epoch["mds"] += [l]
            if "infomax" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out = model(b1.to(device), b2.to(device), n_samples = 1,
                                return_embeddings = True)
                l = infomax.train_step(out["embeddings_3d"], out["embeddings_2d"])
                if "infomax" not in ls_train_epoch.keys():
                    ls_train_epoch["infomax"] = []
                ls_train_epoch["infomax"] += [l]
            if "properties2d" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out = model(b1.to(device), b2.to(device), n_samples = 4,
                                predict_features = True)
                l = molprop.train_step(out["predicted_properties"], b1["y"])
                if "properties2d" not in ls_train_epoch.keys():
                    ls_train_epoch["properties2d"] = []
                ls_train_epoch["properties2d"] += [l]
            if "dist" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out = model(b1.to(device), b2.to(device), n_samples = 4,predict_3d = True)
                    y_pred = out["predicted_distances"]
                l = av3d.train_step(y_pred[0][y_pred[1]], b2["dense_distance"])
                if "dist" not in ls_train_epoch.keys():
                    ls_train_epoch["dist"] = []
                ls_train_epoch["dist"] += [l]
            if "gan" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out = model(b1.to(device), b2.to(device), n_samples = 4, generate_fake = True)
                    y_pred = out["predicted_distances"]
                if x%1 == 0:
                    l1 = molgan.discriminator_train_step(out["logits_real_graphs"].squeeze(), out["logits_fake_graphs"].squeeze(), retain_graph=False)
                    if "gan_discriminator" not in ls_train_epoch.keys():
                        ls_train_epoch["gan_discriminator"] = []
                    ls_train_epoch["gan_discriminator"] += [l1]
                else:
                    l2 = molgan.generator_train_step(out["logits_fake_graphs"].squeeze(), retain_graph=False)
                    if "gan_generator" not in ls_train_epoch.keys():
                        ls_train_epoch["gan_generator"] = []
                    ls_train_epoch["gan_generator"] += [l1]
        model.eval()
        for x, (b1, b2) in enumerate(test_loader):
            if "properties3d" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    with torch.no_grad():
                        out = model(b1.to(device), b2.to(device), n_samples = 4,
                                predict_features3d = True)
                l = molprop.eval_step(out["predicted_properties3d"], b1["y"])
                if "properties3d" not in ls_test_epoch.keys():
                    ls_test_epoch["properties3d"] = []
                ls_test_epoch["properties3d"] += [l]
            if "mds" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    with torch.no_grad():
                        out = model(b1.to(device), b2.to(device), n_samples = 4, 
                                return_mds=True)
                l = mds.eval_step(out["reconstructed_geometry"], b2["dense_distance"], b2["dense_mask"])
                if "mds" not in ls_test_epoch.keys():
                    ls_test_epoch["mds"] = []
                ls_test_epoch["mds"] += [l]
            if "infomax" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    with torch.no_grad():
                        out = model(b1.to(device), b2.to(device), n_samples = 4,
                                return_embeddings = True)
                l = infomax.eval_step(out["embeddings_3d"], out["embeddings_2d"])
                if "infomax" not in ls_test_epoch.keys():
                    ls_test_epoch["infomax"] = []
                ls_test_epoch["infomax"] += [l]
            if "properties2d" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    with torch.no_grad():
                        out = model(b1.to(device), b2.to(device), n_samples = 4,
                                predict_features = True)
                l = molprop.eval_step(out["predicted_properties"], b1["y"])
                if "properties2d" not in ls_test_epoch.keys():
                    ls_test_epoch["properties2d"] = []
                ls_test_epoch["properties2d"] += [l]
            if "dist" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    with torch.no_grad():
                        out = model(b1.to(device), b2.to(device), n_samples = 4,predict_3d = True)
                    y_pred = out["predicted_distances"]
                l = av3d.eval_step(y_pred[0][y_pred[1]], b2["dense_distance"])
                if "dist" not in ls_test_epoch.keys():
                    ls_test_epoch["dist"] = []
                ls_test_epoch["dist"] += [l]
            if "gan" in pretraining:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    with torch.no_grad():
                        out = model(b1.to(device), b2.to(device), n_samples = 4, generate_fake = True)
                    y_pred = out["predicted_distances"]
                if x%1 == 0:
                    l1 = molgan.discriminator_eval_step(out["logits_real_graphs"].squeeze(), out["logits_fake_graphs"].squeeze(), retain_graph=False)
                    if "gan_discriminator" not in ls_test_epoch.keys():
                        ls_test_epoch["gan_discriminator"] = []
                    ls_test_epoch["gan_discriminator"] += [l1]
                else:
                    l2 = molgan.generator_eval_step(out["logits_fake_graphs"].squeeze(), retain_graph=False)
                    if "gan_generator" not in ls_test_epoch.keys():
                        ls_test_epoch["gan_generator"] = []
                    ls_test_epoch["gan_generator"] += [l2]
        for key in ls_test_epoch:
            if key not in ls_test.keys():
                ls_test[key] = []
            ls_test[key] += [np.mean(ls_test_epoch[key])]
        for key in ls_train_epoch:
            if key not in ls_train.keys():
                ls_train[key] = []
            ls_train[key] += [np.mean(ls_train_epoch[key])]
        val_loss = np.mean(list(ls_test_epoch.values()))
        if epoch_callback is not None:
            epoch_callback(val_loss, epoch,)
        else:
            print(epoch, val_loss)
    if final_callback is None:
        final_callback = lambda x, y, z: [x, y, z]
    return final_callback(model, ls_train, ls_test)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pre-training of GNNs for cancer tasks')

    parser.add_argument('--model_file', type=str, required=True, help='Path to the file containing the configurations and models.')
    parser.add_argument('--pretraining', type=str, required=True, help='pretraining strategy to follow.')
    parser.add_argument('--pooling', type=str, required=True, help='pretraining strategy to follow.')
    parser.add_argument('--study', type=str, default = "none", required=False, help='Study used to load hyperparameters')
    parser.add_argument('--device', type=str, required=False,default = "cpu", help='Path to the file containing the configurations and models.')

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_arguments()
    config = {"model_file":args.model_file,
                  "pooling":args.pooling,
                  "num_epochs":50,
                  "pretraining":args.pretraining,
                 "device":args.device}
    if args.study == "none":
        config_model = {"embed_dim":250,
                  "hidden_dim":1024,
                  "learning_rate":0.0001,
                  "num_layers":4}
    else:
        study_name = args.study
        storage_name = "sqlite:///studies/{}.db".format(study_name)
        study = optuna.load_study(study_name, storage_name)
        config_model = study.best_params
        config_model["embed_dim"]*=60
    config = {**config_model, **config}
    model_name = config["model_file"].replace('.py', '')
    def save_model(model, *args):
        torch.save(model.state_dict(), f'models/{model_name}_{config["pretraining"]}_{config["pooling"]}.pt')
    pretrain_model(config, final_callback=save_model)