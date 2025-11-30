import torch.optim as optim
import torch.nn as nn

def train_model(model, pts, vals, cfg):
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    for _ in range(cfg["epochs"]):
        opt.zero_grad()
        loss = loss_fn(model(pts), vals)
        loss.backward()
        opt.step()
