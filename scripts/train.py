#!/usr/bin/env python3

import torch
from torch.utils.data import random_split

def train(model, dataset, epochs, batch_size, filename, optimizer=None):
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())    
    training_data, val_data = random_split(dataset, [3*len(dataset)//4, (len(dataset)+3)//4])
    training_data = torch.reshape(torch.tensor(training_data, dtype=torch.float), (-1, 1, 28, 28))
    val_data = torch.reshape(torch.tensor(val_data, dtype=torch.float), (-1, 1, 28, 28))

    if torch.cuda.is_available():
        training_data = training_data.cuda()
        val_data = val_data.cuda()

    batches = torch.split(training_data, batch_size)

    for i in range(epochs):
        model.train()
        for batch in batches:
            optimizer.zero_grad()
            loss = model.loss_function(model.forward(batch), M_N = 0.005)['loss']
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            print(f"Epoch {i} {model.loss_function(model.forward(val_data), M_N = 0.005)['loss']}")
    
    torch.save(model, filename)