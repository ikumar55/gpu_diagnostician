"""Validation script — Rule 3: Host synchronisation stall.

Broken because: loss.item() is called every step, forcing a device→host copy
(cudaMemcpyDtoH) that stalls the CPU-GPU pipeline assembly line on every iteration.

Expected fingerprint on CUDA:
  sync_events_per_step >= 1
  gpu_idle_fraction     > 0.20  (elevated; idle gaps align with sync events)
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

N_STEPS = 30
WARMUP = 5
BATCH_SIZE = 64
NUM_WORKERS = 4
INPUT_DIM = 784
NUM_CLASSES = 10
DATASET_SIZE = 2000


def _build_model(device: torch.device) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES),
    ).to(device)


def run(steps: int = N_STEPS, profile: bool = False) -> dict:
    """Run the broken training loop and return timing metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    xs = torch.randn(DATASET_SIZE, INPUT_DIM)
    ys = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))
    loader = DataLoader(TensorDataset(xs, ys), batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, shuffle=True)
    loader_iter = iter(loader)

    def _step():
        nonlocal loader_iter
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        # ← the bug: .item() every step forces a D2H sync
        print(f"loss: {loss.item():.4f}", end="\r")

    for _ in range(WARMUP):
        _step()

    step_times_ms = []
    for _ in range(steps):
        t0 = time.perf_counter()
        _step()
        step_times_ms.append((time.perf_counter() - t0) * 1000)

    print()  # newline after carriage-return prints
    return {
        "step_times_ms": step_times_ms,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    result = run()
    mean_ms = sum(result["step_times_ms"]) / len(result["step_times_ms"])
    print(f"broken_sync        mean step: {mean_ms:.1f} ms  "
          f"device: {'cuda' if result['cuda_available'] else 'cpu'}")
