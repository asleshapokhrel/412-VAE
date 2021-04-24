"""Script for running main visualizations."""
import copy
import argparse

import numpy as np
import torch

import src.visualize as visualize
from sklearn.utils import shuffle

if __name__ == "__main__":

  # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_prefix",
                        help="The path to the input CSV.")

    args = parser.parse_args()
    # Import data.
    data_sample = "./data_sample.npy"
    data_names = "./data_names.npy"
    train_dataset = np.load(data_sample)
    train_label = np.load(data_names)

    model_prefix = args.model_prefix

    # Set up the model.
    model = torch.load('Results/saved_models/{}.pth'.format(model_prefix))

    # TSNE
    input_data = torch.reshape(torch.tensor(train_dataset, dtype=torch.float), (-1, 1, 28, 28))

    #Remove this on gpu
    input_data = input_data.cuda()
    train_label[train_label == 'dog'] = 0
    train_label[train_label == 'cat'] = 1
    train_label[train_label == 'car'] = 2
    train_label[train_label == 'airplane'] = 3
    train_label[train_label == 'couch'] = 4
    train_label[train_label == 'chair'] = 5
    train_label = train_label.astype(np.int)
    input_label = torch.Tensor(train_label)

    # Shuffle input data and labels
    shuffle_data , shuffle_labels = shuffle(input_data, input_label, random_state=1, n_samples=200)

    visualize.create_tsne(shuffle_data, model, shuffle_labels, "./Results/visualize_no_dog/{}".format(model_prefix))

    # Select an image, produce it's mean latent variable
    select_img = input_data[50,:,:,:].unsqueeze(0)

    # Visualize the selected image
    pil_array = copy.deepcopy(select_img.squeeze().cpu().detach())
    pil_array[pil_array < 0] = 0
    visualize.generate_image(pil_array, "./Results/visualize_no_dog/{}_single_image.jpg".format(model_prefix))
    mu, log_var = model.encode(select_img)
    mu = mu.view(-1)

    # Do some interpolation with this latent vector.
    visualize.interpolation_lattice(model, "./Results/visualize_no_dog/{}_0_1".format(model_prefix), mu, [0,1], num_samples=25)
    visualize.interpolation_lattice(model, "./Results/visualize_no_dog/{}_0_2".format(model_prefix), mu, [0,2], num_samples=25)
    visualize.interpolation_lattice(model, "./Results/visualize_no_dog/{}_0_3".format(model_prefix), mu, [0,3], num_samples=25)
    visualize.interpolation_lattice(model, "./Results/visualize_no_dog/{}_1_2".format(model_prefix), mu, [1,2], num_samples=25)
    visualize.interpolation_lattice(model, "./Results/visualize_no_dog/{}_1_3".format(model_prefix), mu, [1,3], num_samples=25)
    visualize.interpolation_lattice(model, "./Results/visualize_no_dog/{}_2_3".format(model_prefix), mu, [2,3], num_samples=25)


    # Select another random sample for polar interpolation
    select_img_2 = input_data[20000,:,:,:].unsqueeze(0)

    # Visualize the selected image
    pil_array = copy.deepcopy(select_img_2.squeeze().cpu().detach())
    pil_array[pil_array < 0] = 0
    visualize.generate_image(pil_array, "./Results/visualize_no_dog/{}_second_image.jpg".format(model_prefix))
    mu_2, log_var_2 = model.encode(select_img_2)
    mu_2 = mu_2.view(-1)
    visualize.polar_interpolation(mu, mu_2, model, "./Results/visualize_no_dog/{}".format(model_prefix))


