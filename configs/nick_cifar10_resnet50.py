config = {
    "dataset": {"name": "cifar10", "train_fraction": 0.1},
    "model": {
        "name": "resnet50",
        "bases": [
            "./checkpoints/resnet50-19c8e357.pth",
            "./checkpoints/moco_v1_200ep_pretrain.pth",
        ],
    },
    "merging_fn": "match_tensors_zipit",
    "eval_type": "logits",
    "merging_metrics": ["covariance", "mean"],
}
