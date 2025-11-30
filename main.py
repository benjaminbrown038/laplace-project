import os
from physics import solve_physics
from dataset import build_dataset
from model import MLP
from train import train_model
from plotting import plot_slice

def main():
    cfg = {
        "nx": 30, "ny": 30, "nz": 30,
        "tol": 1e-5, "max_iters": 2000,
        "hidden": 128, "depth": 3,
        "lr": 1e-3, "epochs": 600,
        "outdir": "experiments/results/laplace"
    }

    os.makedirs(cfg["outdir"], exist_ok=True)

    # Solve PDE
    u = solve_physics(cfg)

    # Build training data
    pts, vals = build_dataset(u)

    # Train model
    model = MLP(cfg["hidden"], cfg["depth"])
    train_model(model, pts, vals, cfg)

    # Plot ground truth slice
    plot_slice(u, cfg["outdir"])

if __name__ == "__main__":
    main()
