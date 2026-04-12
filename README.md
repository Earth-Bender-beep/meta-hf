---
title: IRis Compiler Optimization Environment
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# IRis Compiler Optimization Environment

An OpenEnv environment where an **LLM agent selects LLVM optimization passes** to minimize execution time of C programs compiled for **RISC-V architecture**. Each pass is auto-compiled and measured — the agent gets immediate feedback and must manage its pass sequence to beat standard optimization baselines (O0–O3).

## How It Works

1. **reset()** picks a C program and compiles it at O0–O3 to establish **baselines**
2. The agent adds passes one at a time — each is **auto-compiled and measured**
3. Passes that improve execution time are **kept**; passes that don't stay until the agent removes them
4. The agent can **reorder** its sequence to find better orderings
5. **finalize()** ends the episode and awards a bonus based on baseline comparison

### Compilation Pipeline

```
clang (-O0) → opt (agent's passes) → llc → gcc (riscv64) → qemu-riscv64
```

## Action Space (4 Actions)

| Action | Format | Description | Reward |
|--------|--------|-------------|--------|
| `add_pass` | `add_pass:<name>` | Add a pass, auto-compile | +0.1 (improves), -0.05 (doesn't), -0.1 (duplicate/invalid) |
| `remove_pass` | `remove_pass:<name>` | Remove a kept pass | -0.05 |
| `reorder_sequence` | `reorder_sequence:<p1,p2,...>` | Reorder passes, auto-compile | +0.1 (improves), -0.05 (reverts) |
| `finalize` | `finalize` | End episode, compute final bonus | +1.0 to -0.5 (tier-based) |

## Available Passes (19)

| Category | Passes |
|----------|--------|
| **Memory** | `mem2reg`, `sroa` |
| **CFG** | `simplifycfg` |
| **Scalar** | `instcombine`, `gvn`, `sccp`, `dce`, `dse`, `early-cse`, `reassociate` |
| **Loop** | `loop-simplify`, `loop-rotate`, `loop-unroll`, `licm`, `indvars`, `loop-reduce` |
| **Inline** | `inline` |
| **RISC-V** | `consthoist`, `div-rem-pairs` |

## Reward Structure

### Per-Step Rewards (add_pass)
| Outcome | Reward |
|---------|--------|
| Pass improves execution time (kept) | +0.1 |
| Pass does not improve (stays in sequence) | -0.05 |
| Duplicate or invalid pass | -0.1 |

### Per-Step Rewards (remove_pass)
| Outcome | Reward |
|---------|--------|
| Pass removed from sequence | -0.05 |

### Per-Step Rewards (reorder_sequence)
| Outcome | Reward |
|---------|--------|
| New order improves time (kept) | +0.1 |
| New order doesn't improve (reverted) | -0.05 |

### Final Bonus (finalize)
| Tier | Reward |
|------|--------|
| Beat O3 | +1.0 |
| Beat O2 | +0.5 |
| Beat O1 | +0.3 |
| Beat O0 | +0.1 |
| Slower than O0 | -0.5 |

## Observation

**CompilerObservation** fields:
- **data** (dict) — pass info, execution times, current sequence, baselines
- **status** (str) — `pass_kept`, `pass_not_improving`, `pass_removed_by_agent`, `reorder_improved`, `reorder_reverted`, `finalized`, `error`, `duplicate`
- **message** (str) — human-readable feedback
- **reward** (float) — cumulative episode reward
- **done** (bool) — whether episode has ended

## Evaluation Tasks

| Task ID | Program | Difficulty |
|---------|---------|------------|
| `easy` | `01_insertion_sort.c` | Simple sorting |
| `medium` | `08_strassen_matrix.c` | Matrix multiplication |
| `hard` | `120_karatsuba_multiply.c` | Karatsuba multiplication |

## Agent Memory

The agent maintains memory across episodes via `agent_memory.json`:
- **good_sequences** — pass sequences that improved execution time
- **bad_sequences** — pass sequences that didn't help
- **insights** — auto-generated learnings (e.g., "mem2reg first works well on sorting")

Memory is loaded at the start of each inference run and injected into the LLM's initial prompt. It is updated and saved after each task episode, so later tasks benefit from earlier learnings.

## Configuration

| Parameter | Value |
|-----------|-------|
| Max steps per episode | 30 |
| Max total reward | 2.9 |
| Temperature | 0.7 |
| Default model | Qwen/Qwen2.5-72B-Instruct |

## System Requirements

The Docker image includes:
- **clang** + **llvm** (opt, llc) — LLVM compilation toolchain
- **gcc-riscv64-linux-gnu** — RISC-V cross-compiler
- **qemu-user** (qemu-riscv64) — RISC-V binary emulation

## Building & Deploying

```bash
# Build Docker image
openenv build

# Deploy to HF Spaces
openenv push -r HideIron/compiler-env
```

## Project Structure

```
compiler_env/
|-- inference.py                     # LLM agent script (OpenAI tool-calling)
|-- agent_memory.json                # Persistent agent memory
|-- openenv.yaml                     # OpenEnv manifest (3 tasks)
|-- pyproject.toml                   # Dependencies
|-- Dockerfile                       # Docker build (clang, llvm, qemu, gcc-riscv64)
|-- client.py                        # CompilerEnv WebSocket client
|-- models.py                        # CompilerAction & CompilerObservation
|-- ENVIRONMENT_LOGIC.md             # Detailed environment design doc
|-- training_programs/               # 200 C programs
|   |-- 01_insertion_sort.c
|   |-- 08_strassen_matrix.c
|   +-- ... (198 more)
+-- server/
    |-- compiler_env_environment.py  # Core environment logic
    +-- app.py                       # FastAPI app (HTTP + WebSocket)
```
