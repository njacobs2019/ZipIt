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

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import pandas as pd


# def dict_update(d, u):
#     for k, v in u.items():
#         if isinstance(v, list):
#             if k not in d:
#                 d[k] = []
#             d[k] += v
#         else:
#             d[k] = v
#     return d


# def create_df(search_results):
#     base = {}
#     for _, results in search_results.items():
#         base = dict_update(base, results)

#     numbers = np.array(list(base.values())).T
#     cols = list(base.keys())

#     df = pd.DataFrame(numbers, columns=cols)
#     return df


# def get_task_mapping(labels, splits):
#     task_mapping = []
#     for i, label in enumerate(labels):
#         for j, split in enumerate(splits):
#             if label in split:
#                 task_mapping.append(j)
#     return torch.from_numpy(np.array(task_mapping))


def run_node_experiment(
    node_config: Dict, experiment_config: Dict, pairs: List, device, csv_file: str
):
    assert len(pairs) > 0, "pairs is empty"

    for pair in tqdm(pairs, desc="Evaluating Pairs..."):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(raw_config)
        train_loader = config["data"]["train"]["full"]
        base_models = [
            reset_bn_stats(base_model, train_loader)
            for base_model in config["models"]["bases"]
        ]
        config["node"] = node_config

        Grapher = config["graph"]
        graphs = [
            Grapher(deepcopy(base_model)).graphify() for base_model in base_models
        ]

        Merge = ModelMerge(*graphs, device=device)

        Merge.transform(
            deepcopy(config["models"]["new"]),
            train_loader,
            transform_fn=config["merging_fn"],
            metric_classes=config["metric_fns"],
            stop_at=node_config["stop_node"],
            **node_config["params"],
        )

        reset_bn_stats(Merge, train_loader)

        results = evaluate_model(experiment_config["eval_type"], Merge, config)
        for idx, split in enumerate(pair):
            results[f"Split {CONCEPT_TASKS[idx]}"] = split
        results["Time"] = Merge.compute_transform_time
        results["Merging Fn"] = config["merging_fn"].__name__
        results["Model Name"] = config["model"]["name"]
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)
        print(results)
        # pdb.set_trace()

    print(f"Results of {node_config}: {results}")
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU isn't working"
    config_name = "nick_cifar10_resnet50"
    skip_pair_idxs = [0]

    experiment_configs = [
        {"stop_node": 21, "params": {"a": 0.0001, "b": 0.075}},
    ]

    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config["model"]["dir"]
    model_name = raw_config["model"]["name"]
    run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    print(raw_config["merging_fn"])
    csv_file = os.path.join(
        "./csvs",
        raw_config["dataset"]["name"],
        raw_config["model"]["name"],
        raw_config["eval_type"],
        "zipit_configurations.csv",
    )

    with torch.no_grad():
        for node_config in experiment_configs:
            raw_config["dataset"].update(node_config.get("dataset", {}))

            run_node_experiment(
                node_config=node_config,
                experiment_config=raw_config,
                pairs=run_pairs,
                device=device,
                csv_file=csv_file,
            )
