"""Methods for visualizing the variational posterior."""
import copy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import torch

import src.models


def generate_image(img_tensor, save_dir):
    """ Helper function for generating images from tensors.

    Parameters:
    img_tensor: torch.Tensor. The tensor we'll be converting into
        an image.
    save_dir: str. The path to save the image to.
    """
    # Convert the tensor to a numpy array.
    img_np_array = img_tensor.cpu().detach().numpy()
    image = Image.fromarray(np.uint8(img_np_array)).convert('L')
    image.save(save_dir)


def interpolation_lattice(model, save_prefix, base_latent, interp_dims, num_samples=25):
    """ Create an array of points to interpolate on. 

    model: BaseVae object. The model we're visualizing
    save_prefix: str. Path prefix for saving the image.
    base_latent: torch.Tensor. The initial latent vector that we
        should interpolate on. Note that its dimension should match that of the
        latent space of the model.
    interp_dims: list of 2 ints. The two dimensions to interpolate on.
    num_samples: int. The number of samples to take between -1 and 1.
    """
    # First, copy the base latent vector num_samples times
    first_copy = torch.stack(num_samples*[base_latent])

    # Create a vector that interpolates between -1 and 1 in num_samples steps.
    linspace = torch.linspace(-3, 3, num_samples).cuda()

    # Replace the first interpretation dim with the linspace
    # TODO: EXPERIMENT WITH ADDING THE LINSPACE INSTEAD
    first_copy[:, interp_dims[0]] += linspace

    # Now make another dimension to interpolate upon
    second_copy = torch.stack(num_samples*[first_copy])

    # Now replace the second interpolation dimension with the linspace
    # TODO: EXPERIMENT WITH ADDING THE LINSPACE INSTEAD
    second_copy.transpose(0,1)[:, :, interp_dims[1]] += linspace


    # Reformat this as a list of num_samples^2 latent vectors.
    interp_vecs = second_copy.view(num_samples**2, -1)

    # Pass these through the decoder
    interp_imgs = model.decode(interp_vecs)

    # Finally, we'll reformat this [num_samples^2, 1, 28, 28] array
    # into a [1, num_samples*28, num_samples*28] image such that the
    # interpolation dimensions are kept correctly
    row_list = []
    for row_idx in range(num_samples):
        row_list.append(
            torch.cat([interp_imgs[row_idx*num_samples + col_idx,0, :, :] for col_idx in range(num_samples)])
        )
    final_array = torch.cat(row_list, dim=1)

    img_np_array = final_array.cpu().detach().numpy()

    # Multiply by 255 to get better fidelity (Potentially?).
    # Temporarily also save a numpy array
    plt.imshow(img_np_array, cmap='gray')
    plt.savefig("{}_pyplot.png".format(save_prefix))
    pil_array = copy.deepcopy(final_array.cpu().detach())
    pil_array[pil_array < 0] = 0
    generate_image(pil_array, "{}_pil.jpg".format(save_prefix))


def create_tsne(image_data, model, labels, save_prefix):
    """ Map the training data's latent space into two dimensions

    Paramters:
    image_data: torch.Tensor. The training data.
    model: BaseVae object. The trained VAE model, based on the object
        defined in "methods".
    labels: torch.Tensor. The labels for the training data.
    save_prefix: str. Path prefix for saving the image.

    """
    # Encode the data and fit the TSNE model.
    mu, log_var = model.encode(image_data)
    embedded_data = TSNE(n_components=2).fit_transform(mu.detach().cpu().numpy())

    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels)
    plt.savefig("{}_tsne_plot.png".format(save_prefix))


def polar_interpolation(latent_1, latent_2, model, save_prefix, num_samples = 25):
    """ Perform polar interpolation between two latent vectors.

    Parameters:
    latent_1, latent_2: torch.Tensor. Latent vectors for interpolation.
    num_samples: int. Grainularity of the interpolation.
    """
    # Create a lin space to base our interpolation on.
    linspace = torch.linspace(0, 1, num_samples).cuda()
    linspace = linspace.unsqueeze(1)
    linspace_1 = torch.sqrt(linspace)
    linspace_2 = torch.sqrt(1 - linspace)
    latent_1_lin = latent_1 * linspace_1
    latent_2_lin = latent_2 * linspace_2

    # Combine these two to complete the interpolation.
    full_interpolated_samps = latent_1_lin + latent_2_lin

    # Pass these through the decoder
    interp_imgs = model.decode(full_interpolated_samps)
    row_imgs = torch.cat([interp_imgs[col_idx, 0, :, :] for col_idx in range(num_samples)], dim=1)

    img_np_array = row_imgs.cpu().detach().numpy()

    # Multiply by 255 to get better fidelity (Potentially?).
    # Temporarily also save a numpy array
    pil_array = copy.deepcopy(row_imgs.cpu().detach())
    pil_array[pil_array < 0] = 0
    generate_image(pil_array, "{}_polar_interp.jpg".format(save_prefix))



def visualize_random(latent, likelihood_net):
    """Given a latent vector z, visualize a sample x' ~ q(x|z).

    Parameters:
    latent: torch.Tensor. The latent vector z.
    likelihood_net: Torch Model. Generative network, outputs 
        a tuple of (means, sigmas), both equally sized `torch.Tensor`s
        which define a distribution over the generated images.
    """
    pass

def visualize_random_latent(likelihood_net):
    """ Visualize a random vector from a random likelihood dist.
    Using a latent vector z sampled from the prior p(.), 
    visualize a sample x' ~ q(x|z).

    """
    pass

def visualize_mean_latent_space(X, posterior_net, dims, labels=None):
    """ Visualize the means of produced by the posterior net.

    Parameters:
    X: torch.Tensor. The images we'll be visualizing the means for.
    posterior_net: Torch Model. The model we're using to parameterize
        the posterior. Should output a tuple of means and variances,
        which are `torch.Tensors` of equal size, which are used
        to parameterize the posterior.
    dims: (int, int). The two dimensions of the mean vector
        we'll be visualizing.
    labels: torch.Tensor or None. Optional. Labels for X. If provided,
        datapoints will be coloured according to label.

    """
    pass