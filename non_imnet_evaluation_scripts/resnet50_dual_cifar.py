""""
This script is inspired by zipit_concept_merging.py
It combines two resnet50 models and computes the activates with cifar-10
"""

import os
import random
from copy import deepcopy
from time import time
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from model_merger import ModelMerge

from utils import *
from utils import prepare_data, get_config_from_name

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def merge_models(
    zipit_params: Dict, experiment_config: Dict, device: str, csv_fname: str
):
    """
    Args:
        zipit_params (Dict): Contains stop_nodes and hparams. (previously node_config)
        experiment_config (Dict): Read from configuration file.
        device (str): "cuda" or "cpu"
        csv_fname (str): file name for the csv results

    Returns:
        results (Dict): results of the merge
    """

    # prepare the configuration file
    # config = prepare_experiment_config(raw_config)

    config = raw_config

    config["data"] = prepare_data(config["dataset"], device=config["device"])

    # Get the graphs
    from ..graphs import resnet_graph

    config["graph"] = resnet_graph.resnet50

    Grapher = config["graph"]
    graphs = [
        Grapher(deepcopy(base_model)).graphify() for base_model in base_models
    ]  # create graph for each base model

    config["models"] = prepare_models(config["model"], device=config["device"])
    config["merging_fn"] = get_merging_fn(config["merging_fn"])
    config["metric_fns"] = get_metric_fns(config["merging_metrics"])

    # get train loader
    train_loader = config["data"]["train"]["full"]

    # reset batch norm stats for base models
    base_models = [
        reset_bn_stats(base_model, train_loader)
        for base_model in config["models"]["bases"]
    ]
    config["node"] = zipit_params

    Merge = ModelMerge(*graphs, device=device)

    Merge.transform(
        deepcopy(config["models"]["new"]),
        train_loader,
        transform_fn=config["merging_fn"],
        metric_classes=config["metric_fns"],
        stop_at=zipit_params["stop_node"],
        **zipit_params["params"],
    )

    reset_bn_stats(Merge, train_loader)

    #### here
    # results = evaluate_model(experiment_config["eval_type"], Merge, config)
    # for idx, split in enumerate(pair):
    #     results[f"Split {CONCEPT_TASKS[idx]}"] = split
    # results["Time"] = Merge.compute_transform_time
    # results["Merging Fn"] = config["merging_fn"].__name__
    # results["Model Name"] = config["model"]["name"]
    # results.update(flatten_nested_dict(zipit_params, sep=" "))
    # write_to_csv(results, csv_file=csv_fname)
    # print(results)
    # pdb.set_trace()

    print(f"Results of {zipit_params}: {results}")
    return results


if __name__ == "__main__":
    # Set these params
    device = "cuda"
    assert torch.cuda.is_available(), "GPU not available"
    config_name = "nick_cifar10_resnet50"
    zipit_params = {"stop_node": 21, "hparams": {"a": 0.0001, "b": 0.0875}}

    # Gets config files and injects device
    raw_config = get_config_from_name(config_name, device=device)

    # Unnecessary
    # model_dir = raw_config["model"]["dir"]
    # model_name = raw_config["model"]["name"]
    # print(raw_config["merging_fn"])

    # CSV file name
    csv_fname = os.path.join(
        "./csvs",
        raw_config["dataset"]["name"],
        raw_config["model"]["name"],
        raw_config["eval_type"],
        "zipit_configurations.csv",
    )

    with torch.no_grad():
        # Run the merging function

        # raw_config["dataset"].update(node_config.get("dataset", {}))  # Might need this

        merge_models(
            zipit_params=zipit_params,
            experiment_config=raw_config,
            device=device,
            csv_fname=csv_fname,
        )
