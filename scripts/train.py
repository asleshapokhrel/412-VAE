#!/usr/bin/env python3

import torch

def train(model, dataset, epochs, batch_size, optimizer=torch.optim.Adam(model.parameters())):    
        training_data, val_data = data.random_split(dataset, [3*length(dataset)/4, length(dataset)/4])

        if torch.cuda.is_available():
            training_data = training_data.cuda()
            val_data = val_data.cuda()

        batches = torch.split(training_data)

        for i in range(epochs):
            for batch in batches:
                optimizer.zero_grad()
                loss = model.loss_function(model.forward(batch))
                loss.backward()
                optim.step()
            print("Epoch {i} {model.loss_function(model.forward(val_data))")