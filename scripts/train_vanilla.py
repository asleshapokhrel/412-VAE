#!/usr/bin/env python3

import torch
from src.Data_PreProcessing import *
from src.models import *
from train import train

# add or remove objects here
objects = ['mouse', 'airplane']

# load data
data = load_mult_data(objects)

# get merged datasets
num_samples = 2
train_dataset, train_label = create_dataset(data, objects, num_samples)

vanilla_vae = VanillaVae()
train(vanilla_vae, train_dataset, 10, 128)