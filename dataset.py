import numpy as np
import torch

def build_dataset(u):
    nx, ny, nz = u.shape
    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, nx),
        np.linspace(0, 1, ny),
        np.linspace(0, 1, nz),
        indexing="ij"
    )

    pts = torch.tensor(
        np.stack([X, Y, Z], axis=-1).reshape(-1, 3),
        dtype=torch.float32
    )

    vals = torch.tensor(
        u.reshape(-1, 1),
        dtype=torch.float32
    )

    return pts, vals
