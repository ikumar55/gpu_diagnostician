# v1 Bottleneck Taxonomy — Diagnosis Logic Spec

This is the core logic of the tool: how it turns measured numbers into a ranked, evidence-backed diagnosis with a concrete fix.

> ⚠️ **Read this first.** Every threshold below is a **starting value (v0)**, not gospel. Real values depend on hardware, model, and batch size. The correct workflow is: implement these rules, then **tune the thresholds against your validation suite** until each broken script trips the right rule and nothing else. Being able to *explain why a threshold sits where it does* is part of what makes this project credible in interviews — so own these numbers, don't just trust them.

---

## 1. Inputs the rules consume (the "features")

The feature-extraction stage boils the raw profiler trace down to these. Each rule reads from this set.

| Feature | What it is | How you get it | Difficulty |
|---|---|---|---|
| `gpu_util_pct` | avg GPU utilization over the active window | sample via `pynvml` / nvidia-ml-py during the run | easy |
| `cpu_util_pct` | avg CPU utilization | `psutil` sampling | easy |
| `gpu_idle_fraction` | fraction of step time the GPU is doing nothing | gaps between GPU kernels in the trace | medium |
| `idle_gaps` | list of idle gaps with timestamps | from the trace timeline | medium |
| `tiny_kernel_fraction` | share of kernels shorter than ~10µs | kernel durations from `torch.profiler` | easy |
| `kernel_time_split` | GPU time split into compute-heavy vs memory-heavy vs idle | categorize kernels (see note) | medium |
| `top_kernels` | slowest kernels by total time, each tagged compute- or memory-bound | trace + categorization | medium |
| `sync_events_per_step` | device→host syncs inside the loop (`.item()`, `.cpu()`, `cudaStreamSynchronize`) | sync/memcpy-D2H events in the trace | medium |
| `batch_size` | training batch size | read from user config if available | easy if exposed |
| `num_workers` | DataLoader worker count | introspect DataLoader if accessible | easy if exposed |

**Honest note on the hard part — arithmetic intensity.** A true roofline needs FLOPs-per-byte for each kernel, which is involved to compute exactly. For v1, **approximate** it by categorizing kernels by name: matmul / gemm / conv → compute-bound; elementwise / add / mul / copy / normalization / activation → memory-bound. This is a heuristic, not a real roofline. Say so in your README — *knowing* it's an approximation and being able to describe the exact version is itself a strong signal. (Exact roofline is great "future work.")

---

## 2. Output format (every diagnosis looks like this)

Each fired rule emits a structured result, not just a label:

```
Diagnosis: Dataloader starvation
Confidence: 94%   (matched 4/4 signals)
Evidence:
  ✓ GPU utilization 43%        (threshold: < 60%)
  ✓ CPU utilization 97%        (threshold: > 85%)
  ✓ GPU idle 38% of each step  (threshold: > 20%)
  ✓ num_workers = 0            (corroborating)
Estimated impact: recovering data-wait idle could cut step time ~30–38%
Recommended fix (concrete patch):
    - DataLoader(dataset, batch_size=64, num_workers=0)
    + DataLoader(dataset, batch_size=64, num_workers=8,
                 pin_memory=True, persistent_workers=True)
```

Two rules for this format:
- **Confidence = matched signal weight / total signal weight**, shown as a %, with the "X/Y signals" reason. Surface the evidence checklist every time — it's what makes the tool an instrument, not a black box.
- **Concrete patch** when the fix is a clean code change. When it isn't (e.g. fusing ops = restructuring the model), fall back to a precise prose instruction instead of a fake patch.

---

## 3. The five rules

### Rule 1 — Dataloader starvation
*The GPU is idle waiting for the CPU to prepare the next batch.*

**Signals (weights):**
- `gpu_util_pct < 60` (0.30)
- `cpu_util_pct > 85` (0.25)
- `gpu_idle_fraction > 0.20` AND idle gaps land at **step boundaries** (before the first compute kernel of each step) (0.30)
- `num_workers` is 0 or very low (0.15) — corroborating

**Fires when:** the first three are true. The `num_workers` signal raises confidence but isn't required (you may not always be able to read it).

**Fix / patch:** raise `num_workers` toward (CPU core count − 1), add `pin_memory=True`, `persistent_workers=True`, and consider `prefetch_factor`.

**Impact estimate:** if the GPU is idle `f` fraction of each step *and that idle is at step boundaries*, hiding the data wait can recover up to ~`f` of step time. Report it as an **upper bound**, not a promise.

---

### Rule 2 — Work too small (launch overhead)
*The GPU gets work in pieces too tiny to be worth the per-launch cost.*

**Signals (weights):**
- `gpu_util_pct < 60` (0.30)
- `tiny_kernel_fraction > 0.5` (more than half of kernels under ~10µs) (0.40)
- `batch_size` very small, e.g. ≤ 4 (0.30) — corroborating

**Fires when:** GPU underused **and** tiny-kernel fraction high, **and** the dataloader pattern is *not* the better match (CPU is not pegged — this is what distinguishes it from Rule 1).

**Fix / patch:** increase `batch_size`; apply `model = torch.compile(model)` to fuse ops and cut launch overhead.

**Impact estimate:** hard to predict precisely; note that gains are large at very small batch sizes and shrink as batch grows. Be explicit that this one is rougher than Rule 1.

---

### Rule 3 — Host synchronization stall
*Code forces the CPU and GPU to stop and wait for each other every step.*

**Signals (weights):**
- `sync_events_per_step ≥ 1` (0.45)
- idle gaps **coincide in time** with the sync events (0.35)
- `gpu_idle_fraction` elevated (0.20)

**Fires when:** there's at least one per-step sync **and** idle gaps line up with it.

**Fix / patch:** remove read-backs from the hot loop. Classic case:
```
- print(loss.item())            # every step → forces a sync
+ if step % 50 == 0:            # log occasionally instead
+     print(loss.item())
```
Better still, accumulate loss on-GPU and `.item()` once per logging interval.

**Impact estimate:** proportional to the idle time attributable to syncs (sum the coincident idle gaps).

---

### Rule 4 — Memory-bandwidth-bound
*The GPU is busy, but busy moving data, not doing math — wasting its compute power.*

**Signals (weights):**
- `gpu_util_pct > 70` (GPU **is** busy — this is the key difference from Rules 1–3) (0.30)
- share of GPU compute time in **memory-bound** kernels > 40% (0.45)
- top kernels are low-arithmetic-intensity (elementwise / norm / activation) (0.25)

**Fires when:** GPU is well-utilized but its time is dominated by memory-bound kernels.

**Fix / patch:** fuse pointwise ops (`torch.compile` does this well); use a fused optimizer (`fused=True`); check tensor layouts / `channels_last`.

**Impact estimate:** fusion can meaningfully cut memory-bound time, but the size depends on how fusible the ops are — give a qualitative "likely significant" rather than a fake precise number unless you can measure it.

---

### Rule 5 — Compute-bound (the healthy / at-the-ceiling case)
*The GPU is genuinely saturated doing real math. This is the good state.*

**Signals (weights):**
- `gpu_util_pct > 80` (0.35)
- memory-bound time fraction **low** (< 25%) (0.35)
- `gpu_idle_fraction` low, `tiny_kernel_fraction` low, no per-step syncs (0.30)

**Fires when:** all healthy signals hold.

**"Fix":** none needed — report that the workload is near the compute ceiling. *If* the user wants to go faster anyway, the levers are lower precision (fp16 / bf16 / fp8), a better algorithm, or faster hardware — not a bug fix. (This is the natural hook to your low-precision background.)

**Why include a "no problem" rule:** so the tool doesn't invent phantom bottlenecks on a healthy workload. Correctly saying "you're fine, stop optimizing here" is a real, trust-building output.

---

## 4. When several rules fire at once

Real workloads often trip 2–3 rules. Two principles:

1. **Show all that fired, each with its own confidence and evidence.** Don't hide secondary issues.
2. **Rank by estimated recoverable step-time (impact), not by confidence.** A medium-confidence/high-impact issue should outrank a high-confidence/trivial one. Display confidence alongside so the user can judge.

⚠️ Honest caveat to put in the code comments and README: impact estimates are themselves heuristics built on rough assumptions. Rank by them, but present them as estimates, and let the *measured* speedup from your validation suite be the real proof that the ranking was right.

---

## 5. How this ties to the validation suite

Each rule above maps to one deliberately-broken script:

| Rule | Broken script | Must diagnose | Predicted-fix check |
|---|---|---|---|
| 1 | `broken_dataloader.py` (`num_workers=0` + heavy CPU aug) | Dataloader starvation | applying the patch raises GPU util / cuts step time as estimated |
| 2 | `broken_batchsize.py` (`batch_size=1`) | Work too small | larger batch reduces step time |
| 3 | `broken_sync.py` (`print(loss.item())` each step) | Host sync stall | removing the readback cuts idle |
| 4 | `broken_memory.py` (long pointwise chain) | Memory-bandwidth-bound | `torch.compile` fusion cuts memory-bound time |
| 5 | `healthy.py` (well-tuned baseline) | Compute-bound / no major issue | no false bottleneck reported |

Build these **first**. They are both your credibility proof *and* the objective oracle that lets you (and Claude) tune the thresholds safely. A threshold is "right" when every script trips its intended rule and nothing else.
