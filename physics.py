import numpy as np

def solve_physics(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    tol, max_iters = cfg["tol"], cfg["max_iters"]

    u = np.zeros((nx, ny, nz))
    u[0, :, :] = 1.0  # boundary condition

    for _ in range(max_iters):
        u_old = u.copy()

        u[1:-1, 1:-1, 1:-1] = (
            u_old[:-2, 1:-1, 1:-1] +
            u_old[2:, 1:-1, 1:-1] +
            u_old[1:-1, :-2, 1:-1] +
            u_old[1:-1, 2:, 1:-1] +
            u_old[1:-1, 1:-1, :-2] +
            u_old[1:-1, 1:-1, 2:]
        ) / 6.0

        if np.linalg.norm(u - u_old) < tol:
            break

    return u
