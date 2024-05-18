import os
import sys

sys.path.append(".")
from mlc.logging import XP

import torch
import optuna
from optuna.trial import TrialState
from mlc.datasets.dataset_factory import load_dataset
from mlc.models.model_factory import load_model
from argparse import ArgumentParser, Namespace
from mlc.transformers.tab_scaler import TabScaler
from mlc.metrics.compute import compute_metric, compute_metrics
from mlc.metrics.metric_factory import create_metric
from mlc.dataloaders import get_custom_dataloader
import pandas as pd
import itertools
# Torch config to avoid crash on HPC
torch.multiprocessing.set_sharing_strategy('file_system')

CUSTOM_DATALOADERS = ["default", "subset", "madry", "dist"]


def run(
    dataset_name: str , 
    model_name: str, 
    custom_dataloader
) -> None:
    print("Train hyperparameter optimization for {} on {}".format(model_name, dataset_name))
    
    print(f"Evaluating {dataset_name}, {model_name}, {custom_dataloader}")
    dataset = load_dataset(dataset_name)
    metadata = dataset.get_metadata(only_x=True)

    x, y = dataset.get_x_y()
    splits = dataset.get_splits()

    x_test = x.iloc[splits["train"]]
    y_test = y[splits["train"]]
    
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )
    
    model_class = load_model(model_name)
    
    weight_path = f"../models/mlc/best_models/{model_name}_{dataset_name}_{custom_dataloader}.model"
    if not os.path.exists(weight_path):
        print(f"{dataset_name}, {model_name}, {custom_dataloader}: {weight_path} Not found!")
        return {}
    model = model_class.load_class(
        weight_path,
        x_metadata=metadata,
        scaler=scaler,
        force_device="cpu",
    )
    
    metric_names =  ["auc", "accuracy", "precision", "recall", "mcc"]
    metrics = [create_metric(e) for e in metric_names]
    metric_vals = compute_metrics(model, metrics, x_test, y_test)
    out = {
        "dataset": dataset_name,
        "model": model_name,
        "training": custom_dataloader,
    }
    
    
    for metric_name, metric_val in zip(metric_names, metric_vals):
        out[metric_name] = metric_val
        
    
    return out


    

if __name__ == "__main__":
    # parser = ArgumentParser(
    #     description="Training with Hyper-parameter optimization"
    # )
    # parser.add_argument("--dataset_name", type=str, default="lcld_v2_iid",
    #                     )
    # parser.add_argument("--model_name", type=str, default="tabtransformer",
    #                     )
    # parser.add_argument("--val_batch_size", type=int, default=2048)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--custom_dataloaders", type=str, default="default")
    # args = parser.parse_args()
    
    # datasets = ["url", "lcld_v2_iid", "ctu_13_neris", "malware", "wids"]
    datasets = ["lcld_v2_iid", "url","wids"]
    models = ["tabtransformer", "stg", "tabnet", "torchrln", "vime"]
    dataloaders = ["default"]
    metric_list = []
    for dataset, model, dataloader in itertools.product(datasets, models, dataloaders):
        metric_dict = run(
            dataset,
            model,
            dataloader
        )
        metric_list.append(metric_dict)
    df = pd.DataFrame(metric_list)
    df.to_csv("performance_summary_newbug2.csv")
    
    