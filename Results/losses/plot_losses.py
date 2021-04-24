import ast
import matplotlib.pyplot as plt

f_name = "total_losses_per_epochvanilla_latent_dim_20.txt"
with open(f_name, "r") as f:
    loaded_list = f.read()
    f.close()

x = ast.literal_eval(loaded_list)

plt.figure()
plt.plot(x)
plt.xlabel("epoch")
plt.ylabel("loss")
