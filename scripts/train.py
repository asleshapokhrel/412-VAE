#!/usr/bin/env python3

import torch
from torch.utils.data import random_split

def train(model, dataset, epochs, batch_size, filename, optimizer=None):
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())    
    training_data, val_data = random_split(dataset, [9*len(dataset)//10, (len(dataset)+9)//10])
    training_data = torch.reshape(torch.tensor(training_data, dtype=torch.float), (-1, 1, 28, 28))
    val_data = torch.reshape(torch.tensor(val_data, dtype=torch.float), (-1, 1, 28, 28))

    if torch.cuda.is_available():
        training_data = training_data.cuda()
        val_data = val_data.cuda()

    batches = torch.split(training_data, batch_size)

    total_losses = []
    for i in range(epochs):
        model.train()
        total_loss = 0
        for batch in batches:
            optimizer.zero_grad()
            loss = model.loss_function(model.forward(batch), M_N = 1)['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(total_loss)
        total_losses.append(total_loss.tolist())

        model.eval()
        with torch.no_grad():
            print(f"Epoch {i} {model.loss_function(model.forward(val_data), M_N = 1)['loss']}")
   
    # same training losses
    with open("./Results/losses/total_losses_per_epoch"+filename[2:-3]+"txt", "w") as f:
        f.write(f"{total_losses}")
        f.close()

    # save models
    torch.save(model, "./Results/saved_models/" + filename)