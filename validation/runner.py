"""Validation runner.

Executes each broken/healthy training script, captures every available feature,
and writes one JSON artifact per script to artifacts/<name>_features.json.

Always-captured (any device):
  cpu_util_pct, batch_size, num_workers, mean_step_ms

CUDA-only (set to null when CUDA unavailable):
  gpu_util_pct, gpu_idle_fraction, tiny_kernel_fraction,
  mem_bound_fraction, sync_events_per_step, top_kernels

Usage:
  python validation/runner.py              # run all scripts
  python validation/runner.py healthy      # run one script by name
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import torch

# ── CUDA-only imports ─────────────────────────────────────────────────────────
_CUDA = torch.cuda.is_available()

if _CUDA:
    try:
        import pynvml
        pynvml.nvmlInit()
        _PYNVML = True
    except Exception:
        _PYNVML = False
else:
    _PYNVML = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_ARTIFACTS = _REPO_ROOT / "artifacts"

# ── Kernel name categorisation heuristic (v0 approximation) ──────────────────
_COMPUTE_KEYWORDS = ("gemm", "matmul", "conv", "bmm", "wgrad", "dgrad")


def _categorise_kernel(name: str) -> str:
    """Return 'compute' or 'memory' based on kernel name heuristics."""
    low = name.lower()
    if any(kw in low for kw in _COMPUTE_KEYWORDS):
        return "compute"
    return "memory"


# ── CPU utilisation sampler ───────────────────────────────────────────────────

class _CpuSampler:
    """Samples cpu_percent in a background thread at ~5 Hz."""

    def __init__(self) -> None:
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def _loop(self) -> None:
        psutil.cpu_percent(interval=None)  # discard first call (initialises counter)
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=None))
            time.sleep(0.2)


# ── GPU utilisation sampler (CUDA only) ──────────────────────────────────────

class _GpuSampler:
    """Samples GPU utilisation via pynvml in a background thread at ~5 Hz."""

    def __init__(self) -> None:
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0) if _PYNVML else None

    def start(self) -> None:
        if _PYNVML:
            self._thread.start()

    def stop(self) -> Optional[float]:
        if not _PYNVML:
            return None
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(float(util.gpu))
            except Exception:
                pass
            time.sleep(0.2)


# ── Profiler trace parser ─────────────────────────────────────────────────────

def _parse_trace(trace_path: Path, n_steps: int) -> Dict:
    """Parse a torch.profiler JSON trace into GPU feature scalars.

    Returns a dict with keys:
      gpu_idle_fraction, tiny_kernel_fraction, mem_bound_fraction,
      sync_events_per_step, top_kernels
    All values are None if the trace file doesn't exist or has no CUDA events.
    """
    null_result = {
        "gpu_idle_fraction": None,
        "tiny_kernel_fraction": None,
        "mem_bound_fraction": None,
        "sync_events_per_step": None,
        "top_kernels": None,
    }

    if not trace_path.exists():
        return null_result

    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # ── Collect CUDA kernel durations and categories ──────────────────────────
    kernel_durations: List[float] = []
    kernel_categories: List[str] = []
    sync_count = 0
    compute_us = 0.0
    memory_us = 0.0

    TINY_US = 10.0

    for ev in events:
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur_us = ev.get("dur", 0)

        if cat == "kernel":
            kernel_durations.append(dur_us)
            c = _categorise_kernel(name)
            kernel_categories.append(c)
            if c == "compute":
                compute_us += dur_us
            else:
                memory_us += dur_us

        elif cat == "cuda_memcpy" and "DtoH" in name:
            sync_count += 1

    if not kernel_durations:
        return null_result

    n_kernels = len(kernel_durations)
    n_tiny = sum(1 for d in kernel_durations if d < TINY_US)
    tiny_kernel_fraction = n_tiny / n_kernels

    total_kernel_us = compute_us + memory_us
    mem_bound_fraction = memory_us / total_kernel_us if total_kernel_us > 0 else None

    # ── Idle fraction: gaps between consecutive GPU kernel end→start ──────────
    # Build a timeline of (start, end) for each kernel event on GPU.
    gpu_intervals: List[tuple] = []
    for ev in events:
        if ev.get("cat") == "kernel":
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            gpu_intervals.append((ts, ts + dur))

    gpu_idle_fraction = None
    if gpu_intervals:
        gpu_intervals.sort(key=lambda x: x[0])
        total_span = gpu_intervals[-1][1] - gpu_intervals[0][0]
        idle_us = 0.0
        prev_end = gpu_intervals[0][1]
        for start, end in gpu_intervals[1:]:
            if start > prev_end:
                idle_us += start - prev_end
            prev_end = max(prev_end, end)
        gpu_idle_fraction = idle_us / total_span if total_span > 0 else 0.0

    # ── Top kernels by total time ─────────────────────────────────────────────
    from collections import defaultdict
    kernel_totals: Dict[str, float] = defaultdict(float)
    for ev in events:
        if ev.get("cat") == "kernel":
            kernel_totals[ev["name"]] += ev.get("dur", 0)
    top_kernels = sorted(
        [{"name": k, "total_us": v, "category": _categorise_kernel(k)}
         for k, v in kernel_totals.items()],
        key=lambda x: x["total_us"],
        reverse=True,
    )[:10]

    return {
        "gpu_idle_fraction": gpu_idle_fraction,
        "tiny_kernel_fraction": tiny_kernel_fraction,
        "mem_bound_fraction": mem_bound_fraction,
        "sync_events_per_step": sync_count / n_steps if n_steps > 0 else None,
        "top_kernels": top_kernels,
    }


# ── Per-script capture ────────────────────────────────────────────────────────

def _capture(script_name: str, script_module) -> Dict:
    """Run one validation script and return its full feature dict."""
    _ARTIFACTS.mkdir(exist_ok=True)
    trace_path = _ARTIFACTS / f"{script_name}_trace.json"

    cpu_sampler = _CpuSampler()
    gpu_sampler = _GpuSampler()

    cpu_sampler.start()
    gpu_sampler.start()

    if _CUDA:
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(_ARTIFACTS), worker_name=script_name
            ),
            record_shapes=False,
            with_stack=False,
        )
        profiler.start()

    raw = script_module.run(profile=_CUDA)

    if _CUDA:
        profiler.step()
        profiler.stop()
        # torch tensorboard handler writes to a subdirectory; find the latest file.
        trace_files = sorted((_ARTIFACTS).glob(f"{script_name}*.json"))
        if trace_files:
            trace_path = trace_files[-1]

    cpu_util = cpu_sampler.stop()
    gpu_util = gpu_sampler.stop()

    step_times = raw["step_times_ms"]
    mean_step_ms = sum(step_times) / len(step_times) if step_times else None

    gpu_features = _parse_trace(trace_path, n_steps=len(step_times)) if _CUDA else {
        "gpu_idle_fraction": None,
        "tiny_kernel_fraction": None,
        "mem_bound_fraction": None,
        "sync_events_per_step": None,
        "top_kernels": None,
    }

    features = {
        "script": script_name,
        "cuda_available": _CUDA,
        # Always-available features
        "cpu_util_pct": cpu_util,
        "batch_size": raw.get("batch_size"),
        "num_workers": raw.get("num_workers"),
        "mean_step_ms": mean_step_ms,
        "step_times_ms": step_times,
        # CUDA-only features
        "gpu_util_pct": gpu_util,
        **gpu_features,
    }

    out_path = _ARTIFACTS / f"{script_name}_features.json"
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)

    print(f"  wrote {out_path.relative_to(_REPO_ROOT)}")
    return features


# ── Registry ──────────────────────────────────────────────────────────────────

def _load_scripts():
    """Import all validation scripts as modules, keyed by name."""
    import importlib
    scripts_dir = Path(__file__).parent / "scripts"
    sys.path.insert(0, str(scripts_dir.parent.parent))  # ensure repo root on path

    names = [
        "broken_dataloader",
        "broken_batchsize",
        "broken_sync",
        "broken_memory",
        "healthy",
    ]
    modules = {}
    for name in names:
        spec = importlib.util.spec_from_file_location(
            name, scripts_dir / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        modules[name] = mod
    return modules


# ── Entry point ───────────────────────────────────────────────────────────────

def main(targets: Optional[List[str]] = None) -> None:
    modules = _load_scripts()

    if targets:
        unknown = [t for t in targets if t not in modules]
        if unknown:
            print(f"Unknown scripts: {unknown}. Available: {list(modules)}")
            sys.exit(1)
        run_names = targets
    else:
        run_names = list(modules)

    print(f"Running {len(run_names)} script(s)  "
          f"[CUDA={'yes' if _CUDA else 'no (GPU features will be null)'}]")

    results = {}
    for name in run_names:
        print(f"\n→ {name}")
        results[name] = _capture(name, modules[name])
        mean = results[name].get("mean_step_ms")
        gpu = results[name].get("gpu_util_pct")
        cpu = results[name].get("cpu_util_pct")
        print(f"  mean_step={mean:.1f}ms  cpu={cpu:.1f}%  "
              f"gpu={'N/A' if gpu is None else f'{gpu:.1f}%'}")

    print(f"\nDone. Artifacts in {_ARTIFACTS.relative_to(_REPO_ROOT)}/")


if __name__ == "__main__":
    targets = sys.argv[1:] or None
    main(targets)
