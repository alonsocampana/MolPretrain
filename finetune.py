from models import Finetuner
import optuna
import sys
sys.path.insert(1, "../BenchCanc/")
from PyDRP.Benchmarks import BenchCanc
import argparse

def finetune(config):
    ps = []
    config_model = config["model"]
    benchmark = BenchCanc(config["benchmark"],
                        minmax_target=False,
                        line_features = "expression",
                        setting = config["benchmark"]["env"]["setting"],
                        residualize=False,
                        dataset = config["benchmark"]["env"]["dataset"],)
    for i in range(10):
        model = Finetuner(model2D = target_model.Model2D(train_dataset= None,**config["model2d"]),
                   init_dim=79,
                   embed_dim=config_model["embed_dim"],
                   edge_dim = 10,
                   pooling = config_model["pooling"],
                    dropout = config_model["dropout_fc"],
                    dropout_genes=config_model["dropout_genes"],
                    norm_embeddings = config_model["norm_embeddings"],
                   pooling_kwargs = {"node_dropout":config_model["node_dropout"],
                                       "dropout":config_model["dropout"]},
                   hidden_dim = config_model["hidden_dim"])

        model.load_from_pretrained(f'models/{config_model["model_file"]}_{config_model["pretraining"]}_{config_model["pooling"]}_{config_model["pretraining_epoch"]}.pt',
                                   load_pooling=config_model["transfer_pooling"],
                                   load_gnn=config_model["transfer_gnn"])
        if config_model["transfer_decoder"]:
            model.transfer_mlp()
        if not config_model["freeze"]:
            pass
        else:
            if "gnn" in config_model["freeze"]:
                model.freeze_gnn()
            if "pooling" in config_model["freeze"]:
                model.pooling.freeze()
            if "partial" in config_model["freeze"]:
                model.partial_unfreeze()
        out = benchmark.train_model(model, i)
        ps += [out]
    return ps
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning of GNNs for cancer tasks')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the file containing the configurations and models.')
    parser.add_argument('--pretraining', type=str, required=True, help='pretraining strategy to follow.')
    parser.add_argument('--dataset', type=str, required=True, help='dataset to finetune on')
    parser.add_argument('--setting', type=str, required=True, help='setting to finetune')
    parser.add_argument('--device', type=str, required=False,default = "cpu", help='Path to the file containing the configurations and models.')
    args = parser.parse_args()
    MODEL = args.model_file
    PRETRAINING = args.pretraining
    DATASET = args.dataset
    DEVICE = args.device
    SETTING = args.setting
    config = {"model":{}, "model2d":{}, "benchmark":{}}
    config["model"]= {"model_file":MODEL,
              "transfer_decoder":True,
              "transfer_pooling":True,
            "transfer_gnn":True,
              "pooling":"deepset",
              "pretraining_epoch":0,
              "freeze":"gnn",
              "num_epochs":100,
              "dropout_fc":0.1,
              "pretraining":PRETRAINING,
              "learning_rate":0.0001,
              "node_dropout":0.1,
              "dropout_genes":0.4,
              "norm_embeddings":"batchnorm",
              "dropout":0.1}
    study_name = f"{PRETRAINING}_{MODEL}_0"
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.load_study(study_name, storage_name)
    config_model = study.best_params
    config_model["embed_dim"]*=60
    config["model"]["embed_dim"] = config_model["embed_dim"]
    config["model"]["hidden_dim"] = config_model["hidden_dim"]
    model_name = config["model"]["model_file"].replace('.py', '')
    target_model = __import__(model_name)
    config_model2d = target_model.config_model2d
    config_model2d = {**config_model, **config_model2d}
    config["model2d"] = config_model2d
    config["benchmark"] = {"optimizer":{"batch_size":256,
                            "learning_rate":0.0001,
                             "max_epochs":100,
                             "patience":4,
                             "clip_norm":10,
                             "kwargs":{}},
                "env":{"device":DEVICE,
                       "setting":SETTING,
                       "dataset":DATASET,
                       "mixed_precision":True}}
    finetune(config)