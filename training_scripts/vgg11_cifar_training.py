import os
import pdb
from copy import deepcopy

import clip
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from models.vgg import vgg11
from utils import *

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN) / 255, np.array(CIFAR_STD) / 255)
denormalize = T.Normalize(
    -np.array(CIFAR_MEAN) / np.array(CIFAR_STD), 255 / np.array(CIFAR_STD)
)

# CIFAR_MEAN = [0.485, 0.456, 0.406]
# CIFAR_STD = [0.229, 0.224, 0.225]
# normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
# denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), np.array(CIFAR_STD))


def load_clip_features(class_names, device):
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in class_names]
    ).to(device)
    model, preprocess = clip.load("ViT-B/32", device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10

    split_runs = 5
    models_per_run = 1
    # data_dir = '/nethome/gstoica3/research/pytorch-cifar100/data/cifar-100-python' #'/tmp'
    data_dir = "./data/cifar-10-python"
    model_dir = f"/srv/share2/gstoica3/checkpoints/cifar{num_classes//2}_trainlogits/"
    wrapper = torchvision.datasets.CIFAR10
    batch_size = 500
    width = 8
    model_name = f"vgg16_w{width}"
    epochs = 100

    model_dir = os.path.join(
        model_dir, model_name, f"pairsplits_ourinit_epochs{epochs}"
    )
    print(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # train_transform = T.Compose([T.RandomHorizontalFlip(), T.Resize(256), T.RandomCrop(227), T.ToTensor(), normalize])
    # test_transform = T.Compose([T.Resize(256), T.CenterCrop(227), T.ToTensor(), normalize])
    train_transform = T.Compose(
        [T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize]
    )
    test_transform = T.Compose([T.ToTensor(), normalize])
    train_dset = wrapper(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dset = wrapper(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    if "clip" in model_dir:
        clip_features = load_clip_features(test_dset.classes, device=device)
        out_dim = 512
    else:
        out_dim = num_classes  # // 2

    for _ in range(split_runs):
        splits = train_test_split(np.arange(num_classes), train_size=(num_classes // 2))
        split_trainers = [
            torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    train_dset,
                    [i for i, label in enumerate(train_dset.targets) if label in split],
                ),
                batch_size=500,
                shuffle=True,
                num_workers=8,
            )
            for split in splits
        ]

        split_testers = [
            torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    test_dset,
                    [i for i, label in enumerate(test_dset.targets) if label in split],
                ),
                batch_size=500,
                shuffle=False,
                num_workers=8,
            )
            for split in splits
        ]
        split1, split2 = splits
        label_remapping = np.zeros(num_classes, dtype=int)
        label_remapping[split1] = np.arange(len(split1))
        label_remapping[split2] = np.arange(len(split2))
        label_remapping = torch.from_numpy(label_remapping)
        print("label remapping: {}".format(label_remapping))
        print(f"{split1}, {split2}")
        for j in range(models_per_run):
            for i in range(len(splits)):
                model = vgg11(w=width, num_classes=out_dim).cuda().train()
                if "clip" in model_dir:
                    class_vectors = [clip_features[split] for split in splits]
                    model, final_acc = train_cliphead(
                        model=model,
                        train_loader=split_trainers[i],
                        test_loader=split_testers[i],
                        class_vectors=class_vectors[i],
                        remap_class_idxs=label_remapping,
                        epochs=epochs,
                    )
                else:
                    model, final_acc = train_logits(
                        model=model,
                        train_loader=split_trainers[i],
                        test_loader=split_testers[i],
                        epochs=epochs,
                        # remap_class_idxs=label_remapping
                    )

                print(f"Base model on {splits[i]} Acc: {final_acc}")
                print("Saving Base Model")
                idxs = [str(k) for k in splits[i]]

                save_dir = os.path.join(model_dir, "_".join(idxs))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, f"{model_name}_v{len(os.listdir(save_dir))}.pth.tar"
                )
                save_model(model, save_path)

    print("Done!")
