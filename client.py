# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""IRis Compiler Optimization Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CompilerAction, CompilerObservation


class CompilerEnv(
    EnvClient[CompilerAction, CompilerObservation, State]
):
    """
    Client for the IRis Compiler Optimization Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with CompilerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.data["program"])
        ...
        ...     result = client.step(CompilerAction(action="add_pass:mem2reg"))
        ...     print(result.observation.status)

    Example with Docker:
        >>> client = CompilerEnv.from_docker_image("compiler_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CompilerAction(action="list_passes"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CompilerAction) -> Dict:
        """
        Convert CompilerAction to JSON payload for step message.

        Args:
            action: CompilerAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CompilerObservation]:
        """
        Parse server response into StepResult[CompilerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CompilerObservation
        """
        obs_data = payload.get("observation", {})
        observation = CompilerObservation(
            data=obs_data.get("data", {}),
            status=obs_data.get("status", "ok"),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
