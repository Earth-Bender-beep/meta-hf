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
    "hard": os.path.join(PROGRAMS_DIR, "120_karatsuba_multiply.c"),
}

ALL_PROGRAMS = sorted(glob.glob(os.path.join(PROGRAMS_DIR, "*.c")))
TRAIN_PROGRAMS = [p for p in ALL_PROGRAMS if p not in EVAL_PROGRAMS.values()]


# === VALID LLVM PASSES (19 high-impact passes) ===

ALL_PASSES = [
    # mem
    "mem2reg", "sroa",
    # cfg
    "simplifycfg",
    # loop
    "loop-simplify", "loop-rotate", "loop-unroll",
    "licm", "loop-reduce", "indvars",
    # scalar
    "gvn", "sccp", "dce", "dse", "early-cse",
    "reassociate", "instcombine",
    # inline
    "inline",
    # riscv
    "consthoist", "div-rem-pairs",
]


class CompilerEnvironment(Environment):
    """
    IRis Compiler Optimization Environment.

    The agent selects LLVM optimization passes to minimize execution time
    of C programs compiled for RISC-V. Programs are compiled using the full
    LLVM pipeline (clang -> opt -> llc -> gcc) and executed via QEMU.

    Each add_pass automatically compiles and measures. If the pass improves
    execution time it is kept (+0.1 reward). If not, it is auto-removed
    (-0.05 reward). Duplicate passes are rejected (-0.1). The agent calls
    finalize() when satisfied to receive a final bonus based on baselines.

    Episode flow:
        1. reset() picks a C program and computes O0-O3 baselines
        2. Agent calls add_pass — each pass is auto-compiled and evaluated
        3. Only passes that improve execution time are kept in the sequence
        4. Agent calls finalize() to end the episode and receive final bonus

    Example:
        >>> env = CompilerEnvironment()
        >>> obs = env.reset()
        >>> obs = env.step(CompilerAction(action="list_passes"))
        >>> obs = env.step(CompilerAction(action="add_pass:mem2reg"))
        >>> obs = env.step(CompilerAction(action="finalize"))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the compiler optimization environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.max_steps = 30
        self.action_space = [
            "add_pass", "remove_pass", "finalize", "get_program_info",
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
        self.best_time = None
        self.current_time = None

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
            self.best_time = None
            self.current_time = None
            try:
                self.baseline_O0 = self._compute_baseline(self.current_program, "O0")
                self.baseline_O1 = self._compute_baseline(self.current_program, "O1")
                self.baseline_O2 = self._compute_baseline(self.current_program, "O2")
                self.baseline_O3 = self._compute_baseline(self.current_program, "O3")
                self.best_time = self.baseline_O0
                self.current_time = self.baseline_O0
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
                "best_time": self.best_time,
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
        elif action_name == "remove_pass":
            obs_data, status, message = self._remove_pass(action_arg)
        elif action_name == "finalize":
            obs_data, status, message = self._finalize()
        elif action_name == "get_program_info":
            obs_data, status, message = self._get_program_info()
        elif action_name == "list_passes":
            obs_data, status, message = self._list_passes()
        elif action_name == "get_current_sequence":
            obs_data, status, message = self._get_current_sequence()
        else:
            raise ValueError(f"Unknown action: {action_name}")

        # Check max steps — auto-finalize if out of steps
        if self._state.step_count >= self.max_steps and not self.done:
            fin_data, fin_status, fin_message = self._finalize()
            obs_data.update(fin_data)
            message += f" | Auto-finalized: {fin_message}"

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
        """
        Add a pass to the sequence with auto-compile feedback.

        1. Reject if missing, invalid, or duplicate
        2. Tentatively add and compile the full sequence
        3. If execution time improves -> keep (+0.1)
        4. If not -> auto-remove (-0.05)
        """
        base_data = {
            "current_sequence": list(self.current_sequence),
            "best_time": self.best_time,
            "steps_remaining": self.max_steps - self._state.step_count,
        }

        if pass_name is None:
            self.episode_reward -= 0.1
            return (base_data, "error", "No pass name provided. Use format: add_pass:pass_name")

        if pass_name not in self.valid_passes:
            self.episode_reward -= 0.1
            return (base_data, "invalid_pass", f"Pass '{pass_name}' is not valid.")

        if pass_name in self.current_sequence:
            self.episode_reward -= 0.1
            return (
                {**base_data, "reason": "duplicate"},
                "duplicate",
                f"Pass '{pass_name}' is already in sequence. Duplicates not allowed.",
            )

        # Tentatively add the pass
        self.current_sequence.append(pass_name)

        # Auto-compile and measure
        try:
            new_time = self._auto_compile_and_measure()
        except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.current_sequence.pop()
            self.episode_reward -= 0.1
            return (
                {**base_data, "error": str(e)},
                "compile_failed",
                f"Pass '{pass_name}' caused compilation failure. Removed.",
            )

        prev_best = self.best_time
        self.current_time = new_time

        if new_time < prev_best:
            # Improvement — keep the pass
            self.best_time = new_time
            self.episode_reward += 0.1
            return (
                {
                    "kept": True,
                    "pass": pass_name,
                    "execution_time": new_time,
                    "best_time": self.best_time,
                    "improvement": True,
                    "current_sequence": list(self.current_sequence),
                    "sequence_length": len(self.current_sequence),
                    "steps_remaining": self.max_steps - self._state.step_count,
                },
                "pass_kept",
                f"Pass '{pass_name}' KEPT. Time: {new_time:.4f}s (improved from {prev_best:.4f}s).",
            )
        else:
            # No improvement — pass stays, agent must call remove_pass to remove it
            self.episode_reward -= 0.05
            return (
                {
                    "kept": False,
                    "pass": pass_name,
                    "execution_time": new_time,
                    "best_time": self.best_time,
                    "improvement": False,
                    "current_sequence": list(self.current_sequence),
                    "sequence_length": len(self.current_sequence),
                    "steps_remaining": self.max_steps - self._state.step_count,
                },
                "pass_not_improving",
                f"Pass '{pass_name}' did NOT improve time. Time: {new_time:.4f}s vs best: {self.best_time:.4f}s. Use remove_pass('{pass_name}') to remove it.",
            )

    def _remove_pass(self, pass_name):
        """
        Remove a previously kept pass from the sequence.
        The agent calls this to undo a pass it no longer wants.
        Always gives a negative reward (-0.05) since it means a bad choice was made.
        """
        base_data = {
            "current_sequence": list(self.current_sequence),
            "sequence_length": len(self.current_sequence),
            "best_time": self.best_time,
            "steps_remaining": self.max_steps - self._state.step_count,
        }

        if not pass_name:
            self.episode_reward -= 0.1
            return (
                {**base_data, "reason": "missing_name"},
                "error",
                "No pass name provided to remove_pass.",
            )

        if pass_name not in self.current_sequence:
            self.episode_reward -= 0.1
            return (
                {**base_data, "reason": "not_in_sequence"},
                "error",
                f"Pass '{pass_name}' is not in the current sequence. Current: {self.current_sequence}",
            )

        # Remove the pass
        self.current_sequence.remove(pass_name)
        self.episode_reward -= 0.05

        # Re-compile to update current_time
        if len(self.current_sequence) == 0:
            self.current_time = self.baseline_O0
        else:
            try:
                self.current_time = self._auto_compile_and_measure()
            except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                self.current_time = self.baseline_O0

        return (
            {
                "removed": True,
                "pass": pass_name,
                "current_time": self.current_time,
                "best_time": self.best_time,
                "current_sequence": list(self.current_sequence),
                "sequence_length": len(self.current_sequence),
                "steps_remaining": self.max_steps - self._state.step_count,
            },
            "pass_removed_by_agent",
            f"Pass '{pass_name}' removed from sequence. Reward: -0.05. Current sequence: {self.current_sequence}",
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
        """Return current pass sequence and best time."""
        return (
            {
                "current_sequence": list(self.current_sequence),
                "sequence_length": len(self.current_sequence),
                "best_time": self.best_time,
                "steps_remaining": self.max_steps - self._state.step_count,
            },
            "success",
            f"Sequence has {len(self.current_sequence)} passes. Best time: {self.best_time:.4f}s",
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

    def _auto_compile_and_measure(self) -> float:
        """
        Internal: compile the current pass sequence and return execution time.
        Raises RuntimeError / CalledProcessError / TimeoutExpired on failure.
        Pipeline: clang -O0 -> opt (agent passes) -> llc -> gcc -> qemu
        """
        tmp_dir = tempfile.mkdtemp()
        bc_file = os.path.join(tmp_dir, "program.bc")
        opt_bc_file = os.path.join(tmp_dir, "program.opt.bc")
        asm_file = os.path.join(tmp_dir, "program.s")
        exe_file = os.path.join(tmp_dir, "program.exe")

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
        return end - start

    def _finalize(self):
        """
        End the episode and compute final bonus based on best_time vs baselines.
        Returns (data, status, message).
        """
        self.done = True

        # Final bonus based on how best_time compares to baselines
        if self.best_time < self.baseline_O3:
            bonus = 1.0
            tier = "O3"
        elif self.best_time < self.baseline_O2:
            bonus = 0.5
            tier = "O2"
        elif self.best_time < self.baseline_O1:
            bonus = 0.3
            tier = "O1"
        elif self.best_time < self.baseline_O0:
            bonus = 0.1
            tier = "O0"
        else:
            bonus = -0.5
            tier = "none"

        self.episode_reward += bonus

        return (
            {
                "final_time": self.best_time,
                "baselines": {
                    "O0": self.baseline_O0, "O1": self.baseline_O1,
                    "O2": self.baseline_O2, "O3": self.baseline_O3,
                },
                "pass_sequence": list(self.current_sequence),
                "sequence_length": len(self.current_sequence),
                "bonus": bonus,
                "tier_beaten": tier,
                "episode_reward": self.episode_reward,
            },
            "finalized",
            f"Episode complete. Best time: {self.best_time:.4f}s. Beat: {tier}. Bonus: {bonus:+.1f}. Total reward: {self.episode_reward:.2f}",
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

        # Max: all 19 unique passes improve (+0.1 each) + beat O3 (+1.0) = 2.9
        max_reward = 0.1 * len(ALL_PASSES) + 1.0
        # Min: all 30 steps are invalid/duplicate (-0.1 each) + fail finalize (-0.5) = -3.5
        min_reward = -0.1 * self.max_steps - 0.5
        score = (self.episode_reward - min_reward) / (max_reward - min_reward)
        return max(0.001, min(0.999, score))
