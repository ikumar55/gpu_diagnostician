"""Validation script — Rule 4: Memory-bandwidth-bound.

Broken because: the forward pass is a long chain of elementwise operations
(add, mul, sigmoid, relu, etc.) on large tensors. Each op does negligible
arithmetic per byte it reads/writes, so the GPU spends almost all its time
waiting on memory, not computing. There is no matmul to saturate the FP units.

Expected fingerprint on CUDA:
  gpu_util_pct         > 70   (GPU IS busy — different from Rules 1-3)
  mem_bound_fraction   > 0.40 (most kernel time is in memory-bound ops)
"""

import time
import torch
import torch.nn as nn

N_STEPS = 30
WARMUP = 5
BATCH_SIZE = 256
# Large tensors to stress memory bandwidth; each is 256 × 4096 floats ≈ 4 MB.
TENSOR_DIM = 4096
CHAIN_LENGTH = 20       # number of elementwise ops in the chain


class ElementwiseChainModel(nn.Module):
    """A model whose forward pass is nothing but a long elementwise op chain."""

    def __init__(self, dim: int, length: int) -> None:
        super().__init__()
        self.dim = dim
        self.length = length
        # Learnable scale/bias so the model has gradients to propagate.
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for _ in range(length)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(length)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Alternate between mul/add/sigmoid/relu — all memory-bound ops.
        ops = [torch.sigmoid, torch.relu, torch.tanh, torch.relu]
        for i in range(self.length):
            x = x * self.scales[i] + self.biases[i]
            x = ops[i % len(ops)](x)
        # Reduce to a scalar loss-like value without a linear layer.
        return x.mean()


def run(steps: int = N_STEPS, profile: bool = False) -> dict:
    """Run the broken training loop and return timing metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElementwiseChainModel(TENSOR_DIM, CHAIN_LENGTH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _step():
        x = torch.randn(BATCH_SIZE, TENSOR_DIM, device=device)
        opt.zero_grad()
        loss = model(x)
        loss.backward()
        opt.step()

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
        "num_workers": 0,           # no DataLoader — tensors created on-device
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    result = run()
    mean_ms = sum(result["step_times_ms"]) / len(result["step_times_ms"])
    print(f"broken_memory      mean step: {mean_ms:.1f} ms  "
          f"device: {'cuda' if result['cuda_available'] else 'cpu'}")
