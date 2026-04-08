# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
IRis Compiler Optimization Environment Implementation.

An LLM agent selects LLVM optimization passes to minimize execution time
of C programs compiled for RISC-V architecture. The agent builds a sequence
of compiler passes, then compiles and measures performance against standard
optimization level baselines (O0-O3).
"""

import glob
import os
import random
import subprocess
import tempfile
import time
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CompilerAction, CompilerObservation
except ImportError:
    from models import CompilerAction, CompilerObservation


# === PROGRAM CONFIGURATION ===

PROGRAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "training_programs")

EVAL_PROGRAMS = {
    "easy": os.path.join(PROGRAMS_DIR, "01_insertion_sort.c"),
    "medium": os.path.join(PROGRAMS_DIR, "08_strassen_matrix.c"),
    "hard": os.path.join(PROGRAMS_DIR, "114_polynomial_multiply_fft.c"),
}

ALL_PROGRAMS = sorted(glob.glob(os.path.join(PROGRAMS_DIR, "*.c")))
TRAIN_PROGRAMS = [p for p in ALL_PROGRAMS if p not in EVAL_PROGRAMS.values()]


# === VALID LLVM PASSES (44 total) ===

ALL_PASSES = [
    # mem
    "mem2reg", "memcpyopt", "sroa",
    # cfg
    "simplifycfg", "mergereturn", "lowerswitch", "break-crit-edges",
    # loop
    "loop-simplify", "loop-rotate", "loop-unroll", "loop-unroll-and-jam",
    "licm", "loop-deletion", "loop-reduce", "loop-vectorize", "indvars",
    # scalar
    "gvn", "sccp", "ipsccp", "dce", "adce", "dse", "early-cse",
    "reassociate", "instcombine", "instsimplify", "jump-threading",
    "correlated-propagation", "tailcallelim",
    # inline
    "inline", "always-inline", "partial-inliner",
    # interprocedural
    "globalopt", "globaldce", "argpromotion", "deadargelim", "function-attrs",
    # vectorize
    "slp-vectorizer",
    # riscv_optimization
    "consthoist", "lower-constant-intrinsics", "div-rem-pairs",
    # target_aware
    "lower-expect", "strip-dead-prototypes", "elim-avail-extern",
    "lower-matrix-intrinsics", "annotation-remarks",
]


class CompilerEnvironment(Environment):
    """
    IRis Compiler Optimization Environment.

    The agent selects LLVM optimization passes to minimize execution time
    of C programs compiled for RISC-V. Programs are compiled using the full
    LLVM pipeline (clang -> opt -> llc -> gcc) and executed via QEMU.

    Episode flow:
        1. reset() picks a C program and computes O0-O3 baselines
        2. Agent calls tools: add_pass, list_passes, get_program_info, etc.
        3. Agent calls compile_and_measure to test its pass sequence
        4. Reward based on how the sequence performs vs baselines

    Example:
        >>> env = CompilerEnvironment()
        >>> obs = env.reset()
        >>> obs = env.step(CompilerAction(action="list_passes"))
        >>> obs = env.step(CompilerAction(action="add_pass:mem2reg"))
        >>> obs = env.step(CompilerAction(action="compile_and_measure"))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the compiler optimization environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.max_steps = 30
        self.action_space = [
            "add_pass", "compile_and_measure", "get_program_info",
            "list_passes", "get_current_sequence",
        ]
        self.current_sequence = []
        self.valid_passes = ALL_PASSES
        self.episode_reward = 0.0
        self.done = False

        self.eval_programs = EVAL_PROGRAMS
        self.train_programs = TRAIN_PROGRAMS
        self.current_program = None
        self.baseline_O0 = None
        self.baseline_O1 = None
        self.baseline_O2 = None
        self.baseline_O3 = None

    def reset(self, task_id=None, seed=None) -> CompilerObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: Optional task ID ('easy', 'medium', 'hard') for eval.
                     None selects a random training program.
            seed: Optional random seed for reproducible program selection.

        Returns:
            CompilerObservation with program info and baselines
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Choose program
        if task_id is not None:
            if task_id in self.eval_programs:
                self.current_program = self.eval_programs[task_id]
            else:
                raise ValueError(
                    f"Task ID '{task_id}' not found. Valid: {list(self.eval_programs.keys())}"
                )
            programs_to_try = [self.current_program]
        else:
            programs_to_try = [random.choice(self.train_programs)]
            # Build a retry list of up to 5 different programs
            tried = {programs_to_try[0]}
            while len(programs_to_try) < 5:
                candidate = random.choice(self.train_programs)
                if candidate not in tried:
                    programs_to_try.append(candidate)
                    tried.add(candidate)

        # Try programs until one compiles successfully
        last_error = None
        for program in programs_to_try:
            self.current_program = program
            self.current_sequence = []
            self.episode_reward = 0.0
            self.done = False
            try:
                self.baseline_O0 = self._compute_baseline(self.current_program, "O0")
                self.baseline_O1 = self._compute_baseline(self.current_program, "O1")
                self.baseline_O2 = self._compute_baseline(self.current_program, "O2")
                self.baseline_O3 = self._compute_baseline(self.current_program, "O3")
                last_error = None
                break
            except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                last_error = e
                continue

        if last_error is not None:
            raise RuntimeError(f"All programs failed to compile: {last_error}")

        return CompilerObservation(
            data={
                "program": os.path.basename(self.current_program),
                "baselines": {
                    "O0": self.baseline_O0, "O1": self.baseline_O1,
                    "O2": self.baseline_O2, "O3": self.baseline_O3,
                },
                "current_sequence": [],
                "steps_remaining": self.max_steps,
            },
            status="ok",
            message=f"Episode started. Program: {os.path.basename(self.current_program)}",
            done=False,
            reward=0.0,
        )

    def step(self, action: CompilerAction) -> CompilerObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: CompilerAction with action string (e.g. "add_pass:mem2reg")

        Returns:
            CompilerObservation with action-specific data, reward, and done flag
        """
        # Check if episode is done
        if self.done:
            return CompilerObservation(
                data={"error": "Episode is done. Call reset() to start a new episode."},
                status="error",
                message="Episode already finished.",
                done=True,
                reward=self.episode_reward,
            )

        # Parse action string
        action_str = action.action
        action_name = action_str.split(":")[0]
        action_arg = action_str.split(":", 1)[1] if ":" in action_str else None

        # Validate action
        if action_name not in self.action_space:
            return CompilerObservation(
                data={"error": f"Unknown action '{action_name}'. Valid: {self.action_space}"},
                status="error",
                message=f"Invalid action: {action_name}",
                done=False,
                reward=self.episode_reward,
            )

        # Increment step count
        self._state.step_count += 1

        # Dispatch
        if action_name == "add_pass":
            obs_data, status, message = self._add_pass(action_arg)
        elif action_name == "compile_and_measure":
            obs_data, status, message = self._compile_and_measure()
        elif action_name == "get_program_info":
            obs_data, status, message = self._get_program_info()
        elif action_name == "list_passes":
            obs_data, status, message = self._list_passes()
        elif action_name == "get_current_sequence":
            obs_data, status, message = self._get_current_sequence()
        else:
            raise ValueError(f"Unknown action: {action_name}")

        # Check max steps
        if self._state.step_count >= self.max_steps and not self.done:
            self.done = True

        return CompilerObservation(
            data=obs_data,
            status=status,
            message=message,
            done=self.done,
            reward=self.episode_reward,
            metadata={"step_count": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state


    def _add_pass(self, pass_name):
        """Add a pass to the sequence. Returns (data, status, message)."""
        if pass_name is None:
            self.episode_reward -= 0.1
            return (
                {"valid_passes": self.valid_passes},
                "error",
                "No pass name provided. Use format: add_pass:pass_name",
            )

        if pass_name not in self.valid_passes:
            self.episode_reward -= 0.2
            return (
                {"valid_passes": self.valid_passes},
                "invalid_pass",
                f"Pass '{pass_name}' is not valid.",
            )

        self.current_sequence.append(pass_name)
        self.episode_reward += 0.05
        return (
            {
                "current_sequence": list(self.current_sequence),
                "sequence_length": len(self.current_sequence),
                "steps_remaining": self.max_steps - self._state.step_count,
            },
            "success",
            f"Added '{pass_name}' to sequence.",
        )

    def _get_program_info(self):
        """Return program name and baselines."""
        return (
            {
                "program": os.path.basename(self.current_program),
                "baselines": {
                    "O0": self.baseline_O0, "O1": self.baseline_O1,
                    "O2": self.baseline_O2, "O3": self.baseline_O3,
                },
                "steps_remaining": self.max_steps - self._state.step_count,
            },
            "success",
            f"Program: {os.path.basename(self.current_program)}",
        )

    def _list_passes(self):
        """Return all valid LLVM passes."""
        return (
            {"valid_passes": self.valid_passes, "total_count": len(self.valid_passes)},
            "success",
            f"{len(self.valid_passes)} passes available.",
        )

    def _get_current_sequence(self):
        """Return current pass sequence."""
        return (
            {
                "current_sequence": list(self.current_sequence),
                "sequence_length": len(self.current_sequence),
                "steps_remaining": self.max_steps - self._state.step_count,
            },
            "success",
            f"Sequence has {len(self.current_sequence)} passes.",
        )

    # ----------------------------------------------
    # Compilation & Measurement
    # ----------------------------------------------

    def _compute_baseline(self, program_path, flag: str) -> float:
        """
        Compile a C program with a standard flag (-O0 to -O3),
        run on QEMU, and return execution time in seconds.
        """
        exe_file = os.path.join(tempfile.mkdtemp(), "baseline.exe")
        clang_cmd = [
            "clang", "--target=riscv64-unknown-linux-gnu",
            "--sysroot=/usr/riscv64-linux-gnu",
            "--gcc-toolchain=/usr",
            f"-{flag}", program_path, "-o", exe_file, "-static",
        ]
        result = subprocess.run(clang_cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(
                f"Baseline compilation failed for {os.path.basename(program_path)} "
                f"with -{flag}: {stderr}"
            )

        start = time.perf_counter()
        subprocess.run(
            ["qemu-riscv64", exe_file], check=True, capture_output=True, timeout=10,
        )
        end = time.perf_counter()
        return end - start

    def _compile_and_measure(self):
        """
        Full compilation pipeline with the agent's pass sequence.
        Pipeline: clang -> opt (agent's passes) -> llc -> gcc -> qemu
        Ends the episode and returns reward based on baseline comparison.
        Returns (data, status, message).
        """
        tmp_dir = tempfile.mkdtemp()
        bc_file = os.path.join(tmp_dir, "program.bc")
        opt_bc_file = os.path.join(tmp_dir, "program.opt.bc")
        asm_file = os.path.join(tmp_dir, "program.s")
        exe_file = os.path.join(tmp_dir, "program.exe")

        try:
            # Step 1: C -> LLVM bitcode (with -O0, no built-in opts)
            subprocess.run(
                ["clang", "--target=riscv64-unknown-linux-gnu",
                 "--sysroot=/usr/riscv64-linux-gnu",
                 "--gcc-toolchain=/usr",
                 "-O0", "-emit-llvm", "-c", self.current_program, "-o", bc_file],
                check=True, capture_output=True, timeout=30,
            )

            # Step 2: Apply agent's passes via opt
            if self.current_sequence:
                pass_arg = f"-passes={','.join(self.current_sequence)}"
            else:
                pass_arg = "-passes=default<O0>"

            subprocess.run(
                ["opt", pass_arg, bc_file, "-o", opt_bc_file],
                check=True, capture_output=True, timeout=60,
            )

            # Step 3: LLVM bitcode -> RISC-V assembly
            subprocess.run(
                ["llc", "-march=riscv64", opt_bc_file, "-o", asm_file],
                check=True, capture_output=True, timeout=30,
            )

            # Step 4: Assembly -> executable (static linking)
            subprocess.run(
                ["riscv64-linux-gnu-gcc", asm_file, "-o", exe_file, "-static"],
                check=True, capture_output=True, timeout=30,
            )

            # Step 5: Run on QEMU and measure
            start = time.perf_counter()
            subprocess.run(
                ["qemu-riscv64", exe_file],
                check=True, capture_output=True, timeout=10,
            )
            end = time.perf_counter()
            execution_time = end - start

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.episode_reward -= 0.5
            self.done = True
            return (
                {"error": str(e), "reward": -0.5, "episode_reward": self.episode_reward},
                "failed",
                "Compilation or execution failed.",
            )

        # Compare against baselines
        if execution_time < self.baseline_O3:
            reward = 1.0
        elif execution_time < self.baseline_O2:
            reward = 0.5
        elif execution_time < self.baseline_O1:
            reward = 0.3
        elif execution_time < self.baseline_O0:
            reward = 0.1
        else:
            reward = 0.0

        self.episode_reward += reward
        self.done = True

        return (
            {
                "execution_time": execution_time,
                "baselines": {
                    "O0": self.baseline_O0, "O1": self.baseline_O1,
                    "O2": self.baseline_O2, "O3": self.baseline_O3,
                },
                "pass_sequence": list(self.current_sequence),
                "reward": reward,
                "episode_reward": self.episode_reward,
            },
            "success",
            f"Execution time: {execution_time:.4f}s, reward: {reward}",
        )

    def grade(self, task_id: str) -> float:
        """
        Grade the agent's performance on an eval task.
        Returns a score between 0.0 and 1.0.
        """
        if not self.done:
            return 0.001
        if self.baseline_O3 is None or self.baseline_O0 is None:
            return 0.001

        max_reward = 0.05 * self.max_steps + 1.0
        min_reward = -0.2 * self.max_steps - 0.5
        score = (self.episode_reward - min_reward) / (max_reward - min_reward)
        return max(0.001, min(0.999, score))
