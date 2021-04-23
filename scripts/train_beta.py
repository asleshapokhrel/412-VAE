#!/usr/bin/env python3

import torch
import src/PreProcessing/preprocessing import *
from src.models import *

# add or remove objects here
objects = ['mouse', 'airplane']


# load data
data = load_mult_data(objects, data_location)

# get merged datasets
num_samples = 2
train_dataset, train_label = create_dataset(data, objects, num_samples)

beta_vae = BetaVae()
train(beta_vae, train_dataset, 1, 128)