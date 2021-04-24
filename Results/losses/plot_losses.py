import ast
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from matplotlib import rc

# for formatting the graph
sns.set()
font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)


def load_data(f_name, total=False):
    with open(f_name, "r") as f:
        loaded_list = f.read()
        f.close()

    arr = ast.literal_eval(loaded_list)

    if total:
        arr = np.array(arr) / (60000 / 256)

    return arr


val = load_data("plot_val_loss_beta_4_latent4.txt")
train = load_data("plot_total_losses_per_epochbeta_4_latent4.txt", total=True)

if __name__ == "__main__":
    plt.figure()
    plt.plot(val, label="validation loss")
    plt.plot(train, label="training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Losses during training")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.show()


