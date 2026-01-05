import torch
from torch.optim import SGD

torch.manual_seed(0)

def run_experiment(lr: float, steps: int = 10):
    print(f"\n=== Learning rate = {lr} ===")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(steps):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(f"step {t:02d}: loss = {loss.item():.6f}")
        loss.backward()
        opt.step()

if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        run_experiment(lr, steps=10)