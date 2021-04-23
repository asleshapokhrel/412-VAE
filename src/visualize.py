"""Methods for visualizing the variational posterior."""
import numpy as np
from PIL import Image
import torch


def generate_image(img_tensor, save_dir):
    """ Helper function for generating images from tensors.

    Parameters:
    img_tensor: torch.Tensor. The tensor we'll be converting into
        an image.
    save_dir: str. The path to save the image to.
    """
    # Convert the tensor to a numpy array.
    img_np_array = img_tensor.cpu().detach().numpy()
    image = Image.fromarray(np.uint8(img_np_array)).convert('RGB')
    image.save(save_dir)


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