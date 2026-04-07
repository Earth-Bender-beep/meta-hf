---
title: IRis Compiler Optimization Environment
emoji: zap
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

An OpenEnv environment where an **LLM agent selects LLVM optimization passes** to minimize execution time of C programs compiled for **RISC-V architecture**. The agent builds a sequence of compiler passes, then compiles and measures performance against standard optimization level baselines (O0-O3).

## Quick Start

```python
from compiler_env import CompilerAction, CompilerEnv

with CompilerEnv.from_docker_image("compiler_env-env:latest") as env:
    # Start episode - picks a C program and computes baselines
    result = env.reset()
    print(f"Program: {result.observation.data['program']}")
    print(f"O3 baseline: {result.observation.data['baselines']['O3']}s")

    # Build a pass sequence
    for pass_name in ["mem2reg", "simplifycfg", "instcombine", "licm", "gvn"]:
        result = env.step(CompilerAction(action=f"add_pass:{pass_name}"))
        print(f"Added {pass_name}, reward so far: {result.reward}")

    # Compile and measure
    result = env.step(CompilerAction(action="compile_and_measure"))
    print(f"Execution time: {result.observation.data['execution_time']}s")
    print(f"Final reward: {result.reward}")
```

## How It Works

1. **reset()** picks a C program (random from 200 training programs, or a fixed eval task)
2. The environment compiles it with `-O0`, `-O1`, `-O2`, `-O3` and measures each on QEMU -> **baselines**
3. The agent calls tools to build an optimization pass sequence
4. **compile_and_measure** runs the full pipeline: `clang -> opt (agent's passes) -> llc -> gcc -> qemu`
5. Reward is based on how the agent's execution time compares to baselines

## Action Space

| Action | Format | Description |
|--------|--------|-------------|
| `add_pass` | `add_pass:<pass_name>` | Add an LLVM pass to the sequence |
| `compile_and_measure` | `compile_and_measure` | Compile & run, ends episode |
| `get_program_info` | `get_program_info` | Get program name + baselines |
| `list_passes` | `list_passes` | List all 44 valid LLVM passes |
| `get_current_sequence` | `get_current_sequence` | Get current pass sequence |

## Observation

**CompilerObservation** fields:
- **data** (dict) - Action-specific data (baselines, sequences, execution times, etc.)
- **status** (str) - `"success"`, `"error"`, `"invalid_pass"`, `"failed"`
- **message** (str) - Human-readable description
- **reward** (float) - Cumulative episode reward
- **done** (bool) - Whether the episode has ended

## Reward Structure (Partial Progress)

| Event | Reward |
|-------|--------|
| Valid pass added | +0.05 |
| Invalid pass attempted | -0.2 |
| Missing pass argument | -0.1 |
| Beat O3 baseline | +1.0 |
| Beat O2 baseline | +0.5 |
| Beat O1 baseline | +0.3 |
| Beat O0 baseline | +0.1 |
| Compilation failure | -0.5 |

## Evaluation Tasks

| Task ID | Program | Difficulty |
|---------|---------|------------|
| `easy` | `01_insertion_sort.c` | Simple sorting algorithm |
| `medium` | `08_strassen_matrix.c` | Matrix multiplication |
| `hard` | `114_polynomial_multiply_fft.c` | FFT-based polynomial multiply |

## System Requirements

The Docker image includes all needed tools:
- **clang** + **llvm** (opt, llc) - LLVM compilation toolchain
- **gcc-riscv64-linux-gnu** - RISC-V cross-compiler
- **qemu-user** (qemu-riscv64) - RISC-V binary emulation

## Building & Deploying

```bash
# Build Docker image
docker build -t compiler_env-env:latest -f server/Dockerfile .

# Deploy to HF Spaces
openenv push
```

## Running Locally

```bash
# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
compiler_env/
|-- __init__.py                      # Module exports
|-- README.md                        # This file
|-- openenv.yaml                     # OpenEnv manifest
|-- pyproject.toml                   # Dependencies
|-- client.py                        # CompilerEnv WebSocket client
|-- models.py                        # CompilerAction & CompilerObservation
|-- inference.py                     # Baseline script (OpenAI API)
|-- training_programs/               # 200 C programs
|   |-- 01_insertion_sort.c
|   |-- 08_strassen_matrix.c
|   |-- 114_polynomial_multiply_fft.c
|   +-- ... (197 more)
+-- server/
    |-- __init__.py
    |-- compiler_env_environment.py  # Core environment logic
    |-- app.py                       # FastAPI app (HTTP + WebSocket)
    |-- requirements.txt
    +-- Dockerfile
```
