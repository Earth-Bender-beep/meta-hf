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
    └────┬─────┘  measures baselines. best_time = O0 time.
         │
         ▼
    ┌──────────────────────────────────────────────────┐
    │  Agent Loop (up to 50 steps)                     │
    │                                                  │
    │  add_pass(name) ──► auto-compile ──► measure     │
    │    ├─ improved? ──► KEEP pass (+0.1 reward)      │
    │    ├─ no gain?  ──► AUTO-REMOVE pass (-0.05)     │
    │    ├─ duplicate? ─► REJECT (-0.1)                │
    │    └─ compile fail ► REMOVE (-0.1)               │
    │                                                  │
    │  list_passes()      ──► info only (no reward)    │
    │  get_program_info() ──► info only (no reward)    │
    │  get_current_seq()  ──► info only (no reward)    │
    │  finalize()         ──► end episode + final bonus│
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────┐
    │  Episode Ends            │
    │  Final bonus by tier     │
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

Each `add_pass` triggers the full compilation pipeline automatically:

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

### Per-Step Rewards (on `add_pass`)
| Event                                        | Reward    |
|----------------------------------------------|-----------|
| Pass improves execution time → **KEPT**      | **+0.10** |
| Pass doesn't improve → **AUTO-REMOVED**      | **-0.05** |
| Duplicate pass → **REJECTED**                | **-0.10** |
| Invalid pass name                            | **-0.10** |
| Compilation failure → **REMOVED**            | **-0.10** |

### Final Bonus (on `finalize()`)
| Condition                      | Bonus    |
|--------------------------------|----------|
| Beat O3 baseline               | **+1.0** |
| Beat O2 baseline               | **+0.5** |
| Beat O1 baseline               | **+0.3** |
| Beat O0 baseline               | **+0.1** |
| Failed to beat O0              | **-0.5** |

### Episode Reward
`episode_reward` = sum of all per-step rewards + final bonus

**Max possible**: 0.1 × 44 (all unique passes improve) + 1.0 (beat O3) = **5.4**
**Min possible**: -0.1 × 50 (all steps invalid/duplicate) - 0.5 (fail finalize) = **-5.5**

---

## Grading / Scoring

The `grade()` method normalizes the episode reward to **(0, 1)**:

```
score = (episode_reward - min_reward) / (max_reward - min_reward)
score = clamp(score, 0.001, 0.999)
```

Where:
- `max_reward = 0.1 × 44 + 1.0 = 5.4`
- `min_reward = -0.1 × 50 - 0.5 = -5.5`

This ensures scores are **strictly between 0 and 1** (never exactly 0.0 or 1.0).

---

## Inference Script (inference.py)

The inference script runs **all 3 eval tasks** in sequence:

```
for task in ["easy", "medium", "hard"]:
    [START] task=easy env=compiler_env model=Qwen/Qwen2.5-72B-Instruct
    
    reset(task_id=task)
    loop:
        LLM → add_pass(name) → auto-compile → feedback (kept/removed)
        [STEP] step=1 action=add_pass:mem2reg reward=0.10 done=false error=null
        [STEP] step=2 action=add_pass:licm reward=-0.05 done=false error=null  (removed)
        ...
        LLM → finalize() → final bonus
        [STEP] step=8 action=finalize reward=0.85 done=true error=null
    
    [END] success=true steps=8 score=0.150 rewards=0.10,0.05,...
```

### LLM Interaction
- Uses **Qwen 72B** via OpenAI-compatible API
- **Tool calling**: LLM returns structured function calls (`add_pass`, `finalize`, etc.)
- After each `add_pass`, the agent sees whether the pass was kept or removed + measured time
- Tool results are fed back into the conversation for multi-turn reasoning
- If LLM returns no tool call, `finalize` is forced

---

## Key State Variables

| Variable            | Type        | Description                                    |
|---------------------|-------------|------------------------------------------------|
| `current_program`   | `str`       | Path to the current C source file              |
| `current_sequence`  | `list[str]` | Only passes that improved — the "winners"      |
| `episode_reward`    | `float`     | Cumulative reward for the episode              |
| `done`              | `bool`      | Whether the episode has ended                  |
| `baseline_O0..O3`   | `float`     | Execution times (seconds) at each opt level    |
| `best_time`         | `float`     | Best execution time seen (starts at O0)        |
| `current_time`      | `float`     | Latest measured execution time                 |
| `max_steps`         | `int`       | 50 — max steps before auto-finalize            |
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
