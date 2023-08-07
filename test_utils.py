from utils import *
import pytest


@pytest.mark.parametrize("dev", ["cuda", None])
def test_get_config_from_name(dev):
    """
    Checks that the config (without device defined) is properly retrieved
    Note: ignores case where device is None and "device" in out
    """

    config_name = "cifar5_vgg"
    config = {
        "dataset": {"name": "cifar5", "shuffle_train": True},
        "model": {
            "name": "vgg11_w1",
            "dir": "./checkpoints/cifar5_trainlogithead/vgg11_w1/pairsplits_ourinit_epochs100",
            "bases": [],
        },
        "eval_type": "logits",
        "merging_fn": "match_tensors_zipit",
        "merging_metrics": ["covariance", "mean"],
        "device": "cuda",
    }

    assert get_config_from_name(config_name, device=dev) == config
