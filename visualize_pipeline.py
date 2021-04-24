"""Script for running main visualizations."""

import numpy as np
import torch

import src.visualize as visualize

if __name__ == "__main__":
    # Import data.
    data_sample = "./data_sample.npy"
    data_names = "./data_names.npy"
    train_dataset = np.load(data_sample)
    train_label = np.load(data_names)

    # Set up the model.
    model = torch.load('Results/saved_models/beta_4_latent_dim4.pth', map_location=torch.device('cpu'))

    # TSNE
    input_data = torch.reshape(torch.tensor(train_dataset, dtype=torch.float), (-1, 1, 28, 28))

    #Remove this on gpu
    input_data = input_data[:10, :, :, :]
    input_label = torch.Tensor(train_label)
    print(input_label.shape)
    print(torch.argmax(input_label, 1)[:10])
    visualize.create_tsne(input_data, model, input_label, "./beta_vae_")
