"""Script for running main visualizations."""
import copy

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
    visualize.create_tsne(input_data, model, input_label, "./beta_vae_")

    # Select an image, produce it's mean latent variable
    select_img = input_data[0,:,:,:].unsqueeze(0)

    # Visualize the selected image
    pil_array = copy.deepcopy(select_img.squeeze().cpu().detach())
    pil_array[pil_array < 0] = 0
    visualize.generate_image(pil_array, "./beta_vae_single_image.jpg")
    mu, log_var = model.encode(select_img)
    mu = mu.view(-1)

    # Do some interpolation with this latent vector.
    visualize.interpolation_lattice(model, "./beta_vae_0_1", mu, [0,1], num_samples=25)
    visualize.interpolation_lattice(model, "./beta_vae_0_2", mu, [0,2], num_samples=25)
    visualize.interpolation_lattice(model, "./beta_vae_0_3", mu, [0,3], num_samples=25)
    visualize.interpolation_lattice(model, "./beta_vae_1_2", mu, [1,2], num_samples=25)
    visualize.interpolation_lattice(model, "./beta_vae_1_3", mu, [1,3], num_samples=25)
    visualize.interpolation_lattice(model, "./beta_vae_2_3", mu, [2,3], num_samples=25)