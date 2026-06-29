"""Validation script — Rule 5: Compute-bound / healthy baseline.

Well-tuned because: large batch size, good DataLoader config, matmul-dominant
model (wide linear layers), no per-step syncs. The GPU should be genuinely
saturated doing real matrix math.

Expected fingerprint on CUDA:
  gpu_util_pct       > 80
  mem_bound_fraction < 0.25
  gpu_idle_fraction  low
  tiny_kernel_fraction low
  sync_events_per_step = 0
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

N_STEPS = 30
WARMUP = 5
BATCH_SIZE = 256         # large batch — keeps GPU fed
NUM_WORKERS = 4          # parallel data loading
INPUT_DIM = 784
HIDDEN_DIM = 2048        # wide layers → matmul-dominant arithmetic intensity
NUM_CLASSES = 10
DATASET_SIZE = 4000


def _build_model(device: torch.device) -> nn.Module:
    # Wide layers maximise matmul work per batch, pushing toward compute ceiling.
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, NUM_CLASSES),
    ).to(device)


def run(steps: int = N_STEPS, profile: bool = False) -> dict:
    """Run the healthy training loop and return timing metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    xs = torch.randn(DATASET_SIZE, INPUT_DIM)
    ys = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))
    loader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        shuffle=True,
    )
    loader_iter = iter(loader)

    def _step():
        nonlocal loader_iter
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        # No .item() here — no sync in the hot loop.

    for _ in range(WARMUP):
        _step()

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
    print(f"healthy            mean step: {mean_ms:.1f} ms  "
          f"device: {'cuda' if result['cuda_available'] else 'cpu'}")
