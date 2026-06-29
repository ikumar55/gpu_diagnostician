"""Validation script — Rule 1: Dataloader starvation.

Broken because: num_workers=0 and each dataset item burns ~5ms of CPU time,
so the GPU sits idle waiting for every batch to be prepared on a single thread.

Expected fingerprint on CUDA:
  gpu_util_pct      < 60
  cpu_util_pct      > 85
  gpu_idle_fraction > 0.20
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

N_STEPS = 30
WARMUP = 5
BATCH_SIZE = 64
NUM_WORKERS = 0          # ← the bug
INPUT_DIM = 784
NUM_CLASSES = 10
_CPU_BURN_ITERS = 30_000  # iterations of sqrt; tuned so one CPU core is saturated
# on a multi-core Colab VM. psutil now reports per-core max, so one hot core
# reads as ~100% regardless of how many other cores are idle.


class SlowFakeDataset(Dataset):
    """Fake MNIST-shaped dataset; __getitem__ burns CPU time to simulate heavy aug."""

    def __init__(self, size: int = 2000) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # Spin the CPU — this is the deliberate bottleneck.
        acc = 1.0
        for i in range(1, _CPU_BURN_ITERS + 1):
            acc = math.sqrt(acc + i)
        x = torch.randn(INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, ()).item()
        return x, y


def _build_model(device: torch.device) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES),
    ).to(device)


def run(steps: int = N_STEPS, profile: bool = False) -> dict:
    """Run the broken training loop and return timing metadata.

    Args:
        steps: Number of measured steps (warmup steps are additional).
        profile: If True and CUDA is available, wrap with torch.profiler.

    Returns:
        dict with keys: step_times_ms, batch_size, num_workers, cuda_available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = SlowFakeDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=True)
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

    # Warmup
    for _ in range(WARMUP):
        _step()

    # Measured steps
    step_times_ms = []
    for _ in range(steps):
        t0 = time.perf_counter()
        _step()
        step_times_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "step_times_ms": step_times_ms,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    result = run()
    mean_ms = sum(result["step_times_ms"]) / len(result["step_times_ms"])
    print(f"broken_dataloader  mean step: {mean_ms:.1f} ms  "
          f"device: {'cuda' if result['cuda_available'] else 'cpu'}")
