# IRis Compiler Optimization Environment — Logic Overview

## High-Level Goal

An LLM agent selects a sequence of **LLVM optimization passes** to minimize the execution time of C programs compiled for **RISC-V** architecture. The agent's chosen pass sequence is compared against standard compiler optimization levels (O0–O3).

---

## Architecture

```
┌─────────────┐     WebSocket      ┌─────────────────────────┐
│  inference.py│◄──────────────────►│  CompilerEnvironment    │
│  (LLM Agent) │                    │  (FastAPI + QEMU)       │
└─────────────┘                    └─────────────────────────┘
       │                                      │
       │ OpenAI API                           │ subprocess
       ▼                                      ▼
┌─────────────┐                    ┌─────────────────────────┐
│  Qwen 72B   │                    │  clang → opt → llc →    │
│  (LLM)      │                    │  gcc → qemu-riscv64     │
└─────────────┘                    └─────────────────────────┘
```

---

## Episode Flow

```
    ┌──────────┐
    │  reset() │  Picks a C program, compiles it at O0/O1/O2/O3,
    └────┬─────┘  measures each baseline execution time via QEMU
         │
         ▼
    ┌──────────────────────────┐
    │  Agent Loop (up to 50    │
    │  steps)                  │
    │                          │
    │  LLM picks a tool call:  │
    │  ┌────────────────────┐  │
    │  │ add_pass(name)     │──┼──► Appends pass to sequence (+0.05 reward)
    │  │ list_passes()      │──┼──► Returns all 44 valid passes (info only)
    │  │ get_program_info() │──┼──► Returns program name + baselines (info only)
    │  │ get_current_seq()  │──┼──► Returns current pass sequence (info only)
    │  │ compile_and_measure│──┼──► Compiles & runs → ends episode
    │  └────────────────────┘  │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Episode Ends            │
    │  Score = f(reward)       │
    └──────────────────────────┘
```

---

## Programs

- **200 C programs** in `training_programs/` (sorting, graph algorithms, crypto, compression, etc.)
- **3 eval tasks** mapped by difficulty:

| Task   | Program                     | Description                              |
|--------|-----------------------------|------------------------------------------|
| easy   | `01_insertion_sort.c`       | Simple sorting algorithm                 |
| medium | `08_strassen_matrix.c`      | Strassen matrix multiplication           |
| hard   | `120_karatsuba_multiply.c`  | Karatsuba large-integer multiplication   |

- Training episodes pick a **random** program from the remaining ~197
- Eval tasks always use the **fixed** program for reproducibility

---

## 44 Valid LLVM Passes

Organized by category:

| Category           | Passes                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------|
| **Memory**         | `mem2reg`, `memcpyopt`, `sroa`                                                                 |
| **CFG**            | `simplifycfg`, `mergereturn`, `lowerswitch`, `break-crit-edges`                                |
| **Loop**           | `loop-simplify`, `loop-rotate`, `loop-unroll`, `loop-unroll-and-jam`, `licm`, `loop-deletion`, `loop-reduce`, `loop-vectorize`, `indvars` |
| **Scalar**         | `gvn`, `sccp`, `ipsccp`, `dce`, `adce`, `dse`, `early-cse`, `reassociate`, `instcombine`, `instsimplify`, `jump-threading`, `correlated-propagation`, `tailcallelim` |
| **Inline**         | `inline`, `always-inline`, `partial-inliner`                                                   |
| **Interprocedural**| `globalopt`, `globaldce`, `argpromotion`, `deadargelim`, `function-attrs`                      |
| **Vectorize**      | `slp-vectorizer`                                                                               |
| **RISC-V Aware**   | `consthoist`, `lower-constant-intrinsics`, `div-rem-pairs`                                     |
| **Target**         | `lower-expect`, `strip-dead-prototypes`, `elim-avail-extern`, `lower-matrix-intrinsics`, `annotation-remarks` |

---

## Compilation Pipeline

When the agent calls `compile_and_measure`, the full pipeline runs:

```
C source ──► clang -O0 -emit-llvm ──► LLVM bitcode (.bc)
                                           │
                                           ▼
                                   opt -passes=<agent's passes>
                                           │
                                           ▼
                                   Optimized bitcode (.opt.bc)
                                           │
                                           ▼
                                   llc -march=riscv64 ──► Assembly (.s)
                                           │
                                           ▼
                                   riscv64-linux-gnu-gcc -static ──► Executable
                                           │
                                           ▼
                                   qemu-riscv64 ──► Execution time measured
```

- Source is compiled with **-O0** first (no built-in optimizations)
- Agent's passes are applied via `opt`
- Final binary is **statically linked** for RISC-V and run under **QEMU**

---

## Reward Structure

### Per-Step Rewards
| Event                          | Reward   |
|--------------------------------|----------|
| Valid pass added (`add_pass`)  | **+0.05** |
| Invalid pass name              | **-0.20** |
| Missing pass name              | **-0.10** |
| Compilation/execution failure  | **-0.50** |

### Final Reward (on `compile_and_measure`)
| Condition                      | Reward   |
|--------------------------------|----------|
| Beat O3 baseline               | **+1.0** |
| Beat O2 baseline               | **+0.5** |
| Beat O1 baseline               | **+0.3** |
| Beat O0 baseline               | **+0.1** |
| Slower than O0                 | **+0.0** |

### Episode Reward
`episode_reward` = sum of all per-step rewards + final reward

**Max possible**: 0.05 × 50 + 1.0 = **3.5** (all valid passes + beat O3)
**Min possible**: -0.2 × 50 - 0.5 = **-10.5** (all invalid passes + compilation failure)

---

## Grading / Scoring

The `grade()` method normalizes the episode reward to **(0, 1)**:

```
score = (episode_reward - min_reward) / (max_reward - min_reward)
score = clamp(score, 0.001, 0.999)
```

Where:
- `max_reward = 0.05 × 50 + 1.0 = 3.5`
- `min_reward = -0.2 × 50 - 0.5 = -10.5`

This ensures scores are **strictly between 0 and 1** (never exactly 0.0 or 1.0).

---

## Inference Script (inference.py)

The inference script runs **all 3 eval tasks** in sequence:

```
for task in ["easy", "medium", "hard"]:
    [START] task=easy env=compiler_env model=Qwen/Qwen2.5-72B-Instruct
    
    reset(task_id=task)
    loop:
        LLM → tool call → env.step() → observation
        [STEP] step=1 action=add_pass:mem2reg reward=0.05 done=false error=null
        ...
    
    [END] success=true steps=8 score=0.722 rewards=0.05,0.10,...
```

### LLM Interaction
- Uses **Qwen 72B** via OpenAI-compatible API
- **Tool calling**: LLM returns structured function calls (`add_pass`, `compile_and_measure`, etc.)
- Tool results are fed back into the conversation for multi-turn reasoning
- If LLM returns no tool call, `compile_and_measure` is forced

---

## Key State Variables

| Variable            | Type        | Description                                    |
|---------------------|-------------|------------------------------------------------|
| `current_program`   | `str`       | Path to the current C source file              |
| `current_sequence`  | `list[str]` | Agent's accumulated LLVM pass sequence         |
| `episode_reward`    | `float`     | Cumulative reward for the episode              |
| `done`              | `bool`      | Whether the episode has ended                  |
| `baseline_O0..O3`   | `float`     | Execution times (seconds) at each opt level    |
| `max_steps`         | `int`       | 50 — max steps before forced episode end       |
| `step_count`        | `int`       | Current step number in the episode             |

---

## API Endpoints

| Endpoint      | Method | Description                                    |
|---------------|--------|------------------------------------------------|
| `/reset`      | POST   | Start a new episode (optional `task_id`)       |
| `/step`       | POST   | Execute an action (WebSocket maintains state)  |
| `/state`      | GET    | Get current environment state                  |
| `/tasks`      | GET    | List all 3 eval tasks with descriptions        |
| `/grade`      | POST   | Grade a task (resets, runs minimal episode)     |
| `/schema`     | GET    | Action/observation schemas                     |

---

## Docker Environment

The container includes:
- **clang/LLVM** — C compiler + optimizer + code generator
- **riscv64-linux-gnu-gcc** — RISC-V cross-compiler for linking
- **qemu-riscv64** — RISC-V user-mode emulator for execution
- **Python + FastAPI + uvicorn** — Web server

All compilation and execution happens **inside the container**, ensuring reproducibility regardless of host OS.
