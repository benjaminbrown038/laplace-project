import matplotlib.pyplot as plt
import os

def plot_slice(u, outdir):
    plt.imshow(u[:, :, u.shape[2] // 2], cmap="inferno")
    plt.colorbar()
    plt.savefig(os.path.join(outdir, "slice.png"))
    plt.close()
