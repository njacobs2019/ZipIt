#!/bin/bash

# Download and unzip
gdown "https://drive.google.com/uc?id=1407oe8AooagrHNRZNLrbvFxzM_z2D1Wg"  # Note gdown is a pip package
unzip pretrained_models.zip

# Cleanup
mv ./pretrained_models/* .
mv moco_v1_200ep_pretrain.pth.tar moco_v1_200ep_pretrain.pth
rm pretrained_models.zip
rmdir pretrained_models
