#!/usr/bin/env python3

import torch
from src.Preprocessing.preprocessing import *
from src.models.BetaVae import BetaVAE
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
    beta_vae = BetaVAE().to(torch.device("cuda"))
else:
    beta_vae = BetaVAE()
train(beta_vae, train_dataset, 10, 128, "./beta.pth")