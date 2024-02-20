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
import optuna
from pretrain import pretrain_model

def objective(trial):
    try:
        def optuna_epoch_callback(val_loss, step):
            trial.report(value=val_loss, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        def optuna_final_callback(model, ls_train, ls_test):
            lss = []
            for k in ls_test.keys():
                lss += [ls_test[k][-1]]
            return np.mean(lss)
        config = {"embed_dim":trial.suggest_int("embed_dim", 1, 10),
                  "hidden_dim":trial.suggest_int("hidden_dim", 128, 1024),
                  "num_layers":trial.suggest_int("num_layers", 1, 8),
                  "learning_rate":trial.suggest_float("learning_rate", 0.000001, 0.01, log=True),
                  "num_epochs":20,
                  "model_file":args.model_file,
                  "pretraining":args.pretraining,
                  "random_signal_dim":trial.suggest_int("random_signal_dim", 1, 200),
                  "dropout":trial.suggest_float("dropout", 0.0, 0.2),
                 "device":args.device}
        config["embed_dim"]*=60
        return pretrain_model(config, epoch_callback = optuna_epoch_callback, final_callback = optuna_final_callback, )
    except Exception as e:
        print(e)
        return 1000

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pre-training of GNNs for cancer tasks')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the file containing the configurations and models.')
    parser.add_argument('--pretraining', type=str, required=True, help='pretraining strategy to follow.')
    parser.add_argument('--device', type=str, required=False,default = "cpu", help='Path to the file containing the configurations and models.')
    parser.add_argument('--study_name', type=str, required=False,default = "0", help='Name of the study')
    parser.add_argument('--n_trials', type=int, required=False,default = 10, help='Number of trials')

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model_file.replace('.py', '')
    study_name = f"{args.pretraining}_{model_name}_{args.study_name}"
    storage_name = f"sqlite:///studies/{study_name}.db"
    study = optuna.create_study(storage=storage_name,
                                study_name = study_name,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=1, interval_steps=3
    ),
)
    study.optimize(objective, n_trials=args.n_trials)