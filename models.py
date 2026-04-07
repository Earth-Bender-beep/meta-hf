# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the IRis Compiler Optimization Environment.

An LLM agent selects LLVM optimization passes to minimize execution time
of C programs compiled for RISC-V architecture.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CompilerAction(Action):
    """
    Action for the compiler optimization environment.

    Format: "action_name" or "action_name:argument"

    Available actions:
      - add_pass:<pass_name>       : Add an LLVM optimization pass to the sequence
      - compile_and_measure        : Compile with current passes and measure performance
      - get_program_info           : Get info about the current program and baselines
      - list_passes                : List all valid LLVM optimization passes
      - get_current_sequence       : Get the current pass sequence
    """

    action: str = Field(
        ...,
        description="Action string in format 'action_name' or 'action_name:argument'",
        examples=["add_pass:mem2reg", "compile_and_measure", "list_passes"],
    )


class CompilerObservation(Observation):
    """
    Observation from the compiler optimization environment.

    Contains action-specific data in the 'data' field, plus standard
    reward, done, and metadata fields.
    """

    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific observation data",
    )
    status: str = Field(
        default="ok",
        description="Status of the action: 'success', 'error', 'invalid_pass', 'failed'",
    )
    message: str = Field(
        default="",
        description="Human-readable message about the action result",
    )
