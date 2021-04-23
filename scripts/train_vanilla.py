#!/usr/bin/env python3

import torch
from src.Preprocessing.preprocessing import *
from src.models.VanillaVae import VanillaVAE
from train import train

# add or remove objects here
objects = ['mouse', 'airplane']

data_location = "./data/"

# load data
data = load_mult_data(objects,data_location)

# get merged datasets
num_samples = 2
train_dataset, train_label = create_dataset(data, objects, num_samples)

if torch.cuda.is_available():
    vanilla_vae = VanillaVAE().to(torch.device("cuda"))
else:
    vanilla_vae = VanillaVAE()
train(vanilla_vae, train_dataset, 10, 128, "./vanilla.pth")