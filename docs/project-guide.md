# The GPU Performance Diagnostician — Project Guide & v1 Spec

This document explains the project from the ground up (no GPU-performance background assumed), then gives you a concrete v1 build plan. Read Part 1 to *understand* it; use Part 2 to *build* it.

---

## Part 1 — Understanding the project

### 1.1 What it is, in one sentence

When people run AI models on a GPU, the model is often much slower than it should be — and figuring out *why* takes years of experience. This tool encodes that experience into automatic rules, so it watches your model run, figures out what's slowing it down, and tells you in plain English what to change.

> **Which GPUs? NVIDIA only — and that's on purpose.** Almost all serious AI training runs on NVIDIA GPUs, using NVIDIA's software layer called CUDA. The "sensors" this tool reads (utilization, the names of the math operations, the stop-and-wait signals) are NVIDIA-specific — they don't exist on an Apple Mac's graphics chip. So the tool targets NVIDIA/CUDA, because that's the hardware the jobs you're aiming for actually care about. A practical consequence: you can *write and test* the code on any computer, but to actually *measure a real model* you need an NVIDIA GPU. If you don't have one, a free Google Colab GPU or a cheap rented GPU box (RunPod, Lambda, Vast — a few dollars an hour) covers it. More on this in section 2.4.

### 1.2 The analogy that makes it click

Think of a modern car. It has hundreds of sensors constantly recording data — temperature, oxygen, RPM, pressure. That raw sensor data already exists, but staring at it tells a normal person nothing.

What's actually useful is the diagnostic reader a mechanic plugs in: it reads all those sensors and says *"cylinder 3 misfire — replace the spark plug."* It turned a flood of raw numbers into a diagnosis and a fix.

In the AI world:
- **The sensors already exist.** Tools like PyTorch Profiler, NVIDIA Nsight, and Meta's Holistic Trace Analysis collect enormous amounts of data about what your model did, microsecond by microsecond.
- **The diagnostic reader does not exist as a clean product.** Today, a senior engineer plays the role of the mechanic — they read the raw data and just *know* what's wrong. Your tool is the missing diagnostic reader. It's the spark-plug verdict, not another sensor.

That gap — "tools tell you *what happened*, not *why* or *what to do*" — is the entire reason this project is worth building.

### 1.3 The one idea you have to understand: why GPUs sit idle

A GPU is, very loosely, a giant kitchen with thousands of cooks. It's astonishingly fast at doing many small math operations at the same time, and the math inside a neural network (mostly multiplying big grids of numbers) is exactly that kind of work.

But thousands of cooks only help if they're kept busy. They need two things constantly delivered: **ingredients** (the data to compute on) and **recipes** (the instructions telling them what to do). If either supply chain stutters, the cooks stand around doing nothing — and you're paying for an idle kitchen.

Here's the key insight that the whole project rests on: **a "slow" model almost always means the GPU is sitting idle, waiting on something.** The expensive chip isn't the bottleneck — something *around* it is failing to keep it fed. So the diagnostic question is always the same:

> The GPU is waiting. **Waiting on what?**

Everything the tool does is answering that one question and recommending how to stop the waiting.

### 1.4 The kinds of "waiting" (this becomes the tool's brain)

There are only a handful of common reasons the GPU waits. Each one is a "bottleneck type," and each leaves a recognizable fingerprint. These five are your v1:

**1. Dataloader starvation — "the kitchen runs out of ingredients."**
Before the GPU can compute on a batch of data, the CPU has to prepare it: read it off the disk, decode images, apply augmentations. If the CPU can't prepare the next batch fast enough, the GPU finishes its current batch and then just waits.
*Fingerprint:* GPU mostly idle, CPU maxed out, with regular little gaps between batches.
*Typical fix:* use more CPU worker processes to prepare data in parallel; pin memory for faster transfer.

**2. Work too small — "giving the kitchen one tiny order at a time."**
Every chunk of work you hand the GPU has a fixed setup cost. If you hand it tiny chunks (a very small batch size, or thousands of tiny operations), that setup overhead dominates and the cooks are barely cooking.
*Fingerprint:* GPU underused, an enormous number of very short operations.
*Typical fix:* increase batch size; combine many small operations into fewer big ones.

**3. Host synchronization stalls — "the chef stops to phone the manager after every dish."**
Normally the CPU and GPU work ahead of each other like an assembly line. But certain code forces them to stop and wait for each other — the classic culprit is reading a value back from the GPU mid-loop (e.g. printing the loss every single step). Each one freezes the assembly line.
*Fingerprint:* regular stalls that line up exactly with these "read back" moments.
*Typical fix:* stop pulling values off the GPU inside the hot loop; log less often.

**4. Memory-bandwidth-bound — "the cooks spend all their time walking to the pantry, not cooking."**
Some operations do very little math but shovel huge amounts of data in and out of the GPU's memory. The cooks' hands are idle because they're constantly waiting for ingredients to be carried over. (This is where the *roofline* idea comes in — see 1.5.)
*Fingerprint:* the slowest operations are ones that move lots of data per unit of math.
*Typical fix:* "fuse" several of these operations so data is touched once instead of many times.

**5. Compute-bound — "the kitchen is genuinely slammed."**
The GPU is actually saturated doing real math. This is the *good* state — it means you're using the hardware well. It's not a bug, but the tool should still recognize and report it, so the user knows they've hit the real ceiling and shouldn't waste time hunting for a phantom problem.

(Two more — out-of-memory crashes, and multi-GPU communication overhead — are real but belong in "future work," not v1. More on scope later.)

### 1.5 The "roofline" — explained without math

Every GPU has exactly two speed limits:
- how fast it can **do math**, and
- how fast it can **move data** to and from its memory.

For any operation, you can ask: which limit is it hitting? An operation that does *lots of math per byte of data* (a big matrix multiply) is limited by the math ceiling — that's fine, that's the GPU working hard. An operation that does *very little math per byte* (adding two big tensors together) is limited by the data-movement ceiling — it's wasting the GPU's math muscle while waiting on memory.

The "roofline" is just a chart with those two ceilings drawn as lines (the shape looks like a roof). You place each of your operations on it. Where it lands tells you which wall you're hitting, and therefore which *kind* of fix will help. That's the whole concept you need — you're not computing anything exotic, you're sorting operations into "math-limited" vs "memory-limited."

### 1.6 Why this doesn't need AI (and is better for it)

It's tempting to think "AI tool = the AI does the diagnosing." It shouldn't. Each bottleneck above has a clean numerical fingerprint, so the diagnosis is just **rules** — plain if/then logic on a few measured numbers. For example:

> IF the GPU is busy less than 60% of the time
> AND the CPU is pegged above 85%
> AND there are regular idle gaps between batches
> THEN → dataloader starvation.

That's engineering, not machine learning, and it's *more* impressive precisely because it proves you understand the systems well enough to write the rule. The only place an AI model might appear is at the very end, to turn the verdict ("dataloader starvation, 38% idle, sync each batch") into a friendly paragraph. The brains are the deterministic rules; the AI is optional gift-wrap.

---

## Part 2 — Building it (the v1 spec)

### 2.1 Architecture — five simple stages

The tool is a pipeline. Each stage has one job:

```
  Your training script
          │
          ▼
  [1] COLLECTOR        ← wraps your training step, records what happens
          │              (built on top of PyTorch Profiler — don't reinvent this)
          ▼
  [2] FEATURE EXTRACTOR← boils the huge raw trace down to ~10 key numbers
          │
          ▼
  [3] DIAGNOSIS ENGINE ← the deterministic rules; outputs a ranked list of bottlenecks
          │
          ▼
  [4] RECOMMENDER      ← maps each bottleneck to a concrete fix + estimated impact
          │
          ▼
  [5] REPORTER         ← prints a clean, ranked, plain-English report
```

You are *standing on* the existing sensor tools (stage 1) and building the valuable layer (stages 2–5) that nobody packages well.

### 2.2 What each stage actually does

**[1] Collector.** A small wrapper the user puts around their training loop. Under the hood it runs `torch.profiler` for a handful of steps (skip the first few — the first iterations include one-time warm-up costs you don't want to measure) and also samples GPU and CPU utilization. Output: a raw trace file plus utilization stats.

**[2] Feature extractor.** The raw trace is huge and unreadable. This stage reduces it to a small set of numbers the rules can reason about:
- GPU utilization % and CPU utilization %
- distribution of idle gaps (how often, how long the GPU waits)
- where GPU time goes: compute vs memory-movement vs idle
- the top few slowest operations, and roughly how math-heavy vs memory-heavy each is
- count of "read-back / sync" events inside the loop
- peak memory used

**[3] Diagnosis engine.** The deterministic rules from 1.6, one per bottleneck type, each producing a confidence score. Output: a ranked list, e.g. *"1. Dataloader starvation (high confidence). 2. Small batch size (medium)."* Ranking matters — a real workload often has several issues and the user needs to know which to fix first.

**[4] Recommender.** Each diagnosis maps to a specific, actionable fix and a rough expected payoff: *"Increase DataLoader workers from 0 to 8 and enable pin_memory — expected to roughly halve step time, since the GPU is idle ~45% of each step waiting on data."*

**[5] Reporter.** A clean terminal report (and a saved HTML/markdown version). Ranked problems, the evidence for each, the recommended fix, the estimated impact. Optionally pass the structured verdict through an LLM for a friendlier write-up — but the report must be fully usable without it.

### 2.3 The validation suite — this is the spine of the whole project

This is the single most important part, and the part most people would skip. It's what converts *"I wrapped the profiler"* into *"I built a diagnostic system and proved it works."*

You write a small set of tiny training scripts, each **deliberately broken in exactly one known way**:

| Script | What's wrong with it | What the tool must diagnose |
|---|---|---|
| `broken_dataloader.py` | 0 data workers + heavy CPU augmentation | Dataloader starvation |
| `broken_batchsize.py` | batch size = 1 | Work too small |
| `broken_sync.py` | `print(loss.item())` every step | Host synchronization stall |
| `broken_memory.py` | long chain of elementwise ops | Memory-bandwidth-bound |
| `healthy.py` | well-tuned baseline | No major bottleneck (near compute ceiling) |

Then you demonstrate two things, and you put the results table front and center in your README:

1. **Correct diagnosis:** the tool identifies the right bottleneck in each broken script.
2. **Predicted fix works:** when you apply the fix the tool recommended, the *measured* speedup matches roughly what it predicted.

That second point is the killer. It turns the project from "a description tool" into "a tool that makes a falsifiable prediction and is right." That's the thing an interviewer cannot dismiss.

### 2.4 Scope discipline — v1 vs. "future work"

**Build in v1 (a few focused weeks):**
- Single GPU only.
- The 5 bottleneck types above.
- The 5-script validation suite.
- A command-line tool that prints a ranked report.

**Mention in the README as "future work" — but do NOT build now:**
- Out-of-memory diagnosis module.
- Multi-GPU / communication bottlenecks.
- A mode that *automatically applies* fixes and re-benchmarks ("auto-optimizer").
- Learning from many runs to discover new bottleneck patterns.

Listing these shows vision and that you know where it goes; building them now guarantees you don't finish. **Narrow-and-validated beats broad-and-shallow every time for getting hired.** A tool that nails 5 bottlenecks with proof is far more impressive than a half-built "platform" that does 12 things weakly.

⚠️ One specific trap to avoid: the "automatically apply the fix and re-benchmark" idea sounds amazing but is a research project in disguise. Many real fixes can't be safely auto-applied (changing batch size alters training behavior; fusing operations means rewriting the user's model code). An *advisor* that says "do X" is clean and finishable. An *agent* that does X to arbitrary code is full of correctness landmines. Keep it as a north star in the README, not a v1 feature.

### 2.4b Where you build it vs. where you run it (important if you're on a Mac)

Because the tool's sensors are NVIDIA-only (see the note in 1.1), the work splits into two places:

**On your own computer (even a Mac, no GPU needed):**
- Write all the code: the project structure, the diagnosis rules, the report formatting, the command-line interface.
- Run all the non-measurement tests — anything that checks the *logic* rather than reading a real GPU.
- This is most of the day-to-day building, so you're not blocked by hardware.

**On a real NVIDIA GPU (free Colab or a cheap rented box):**
- Actually run the five deliberately-broken scripts and capture their real "fingerprints."
- Confirm each script trips the right diagnosis, and that the recommended fix produces the real speedup.
- This is the measurement-and-proof step — it *has* to happen on NVIDIA hardware, because that's the only place the real numbers exist.

Two rules that keep this honest:
1. **Tests that need a GPU should skip themselves when there's no GPU**, and run for real on the GPU box — so your laptop tests stay green without pretending. (In practice: mark them to skip when CUDA isn't available.)
2. **Never hand-write fake GPU numbers to make tests pass locally.** The entire credibility of the project is that the fingerprints are *really measured*. Faking them, even as a convenience, quietly destroys the one thing that makes the project impressive. Measure on real hardware, always.

### 2.5 README framing (recruiters skim this in ~30 seconds)

Structure it in this order:
1. **The hook** — one sentence: *"Point it at a slow PyTorch training loop and it tells you why it's slow and what to change."*
2. **A demo** — a short terminal recording (GIF or asciinema) of it diagnosing a broken script.
3. **The validation table** — the table from 2.3. This is your proof; put it high.
4. **Architecture diagram** — the 5-stage pipeline.
5. **How it works** — a short, plain explanation of the roofline + rules idea (you can lift from Part 1; explaining it well is itself a signal of depth).
6. **How to run it** — clean install + one example command.
7. **Future work** — the parked ideas from 2.4.

Write the whole thing as if explaining to a skeptical senior engineer who has 30 seconds. Clarity is a graded part of the project.

### 2.6 Rough timeline

- **Week 1:** Collector + feature extractor. Get clean numbers out of `torch.profiler` for one simple training loop.
- **Week 2:** Diagnosis engine + recommender for the first 2–3 bottleneck types. Write `broken_dataloader.py` and `broken_sync.py` and confirm correct diagnosis.
- **Week 3:** Remaining bottleneck types + the full validation suite. Verify each predicted fix produces the predicted speedup.
- **Week 4:** Reporter polish, README, demo recording, optional LLM explanation layer.

Difficulty is a dial: if a stage is taking too long, cut a bottleneck type rather than ship something unvalidated. A working tool covering 4 bottlenecks with proof beats a broken one aiming at 7.

---

## Why this is the right project for you

- It produces a **running artifact** people use, not a research write-up.
- It's **useful to real users** — and those users are ML engineers, i.e. the people interviewing you.
- It **doesn't already exist** as a packaged product (the pieces exist; the opinionated diagnosis layer doesn't).
- It sits on **rare skill** — performance engineering — that's hard to build and hard to fake, and is exactly what systems/silicon-leaning teams screen for.
- In interviews it hands you the best possible setup: when they ask *"how do you detect a memory-bound kernel?"*, you get to explain the roofline — you demonstrating depth, instead of describing a project.
