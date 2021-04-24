#!/usr/bin/env python3

import torch
from src.Preprocessing.preprocessing import *
from src.models.VanillaVae import VanillaVAE
from train import train
import numpy as np

# # add or remove objects here
# objects = ['mouse', 'airplane']

# data_location = "./data/"

data_sample = "./data_sample.npy"
data_names = "./data_names.npy"

# # load data
# data = load_mult_data(objects,data_location)

# # get merged datasets
# num_samples = 2
# train_dataset, train_label = create_dataset(data, objects, num_samples)

train_dataset = np.load(data_sample)
train_label = np.load(data_names)

if torch.cuda.is_available():
    vanilla_vae = VanillaVAE().to(torch.device("cuda"))
else:
    vanilla_vae = VanillaVAE()
train(vanilla_vae, train_dataset, 100, 256, "./vanilla_latent4_samedataset.pth")