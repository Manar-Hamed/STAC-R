"""Run an ordinary PyTorch model with STAC-R; no downloads or GPU required."""

import torch
from stacr import apply_stacr, set_step


def main():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    layers = apply_stacr(model, target_names=["0", "2"], rank=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(60):
        set_step(layers, step)
        optimizer.zero_grad(set_to_none=True)
        x, target = torch.randn(16, 32), torch.randn(16, 8)
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    print(f"loss={loss.item():.4f}; compressed projections={len(layers)}")


if __name__ == "__main__":
    main()
