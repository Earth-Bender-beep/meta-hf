# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Compiler Env Environment.

This module creates an HTTP server that exposes the CompilerEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
import sys

import yaml

# Ensure parent directory is on sys.path for imports to resolve in all contexts
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from openenv.core.env_server.http_server import create_app

try:
    from ..models import CompilerAction, CompilerObservation
    from .compiler_env_environment import CompilerEnvironment
except ImportError:
    from models import CompilerAction, CompilerObservation
    from server.compiler_env_environment import CompilerEnvironment

# Create the app with web interface and README integration
app = create_app(
    CompilerEnvironment,
    CompilerAction,
    CompilerObservation,
    env_name="compiler_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# --- Custom endpoints for hackathon task/grader discovery ---

_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openenv.yaml")
_tasks_config = {}
try:
    with open(_yaml_path) as _f:
        _cfg = yaml.safe_load(_f)
        _tasks_config = _cfg.get("tasks", {})
except Exception:
    _tasks_config = {
        "easy": {"description": "Optimize insertion sort", "grader": "builtin"},
        "medium": {"description": "Optimize Strassen matrix multiplication", "grader": "builtin"},
        "hard": {"description": "Optimize polynomial multiplication via FFT", "grader": "builtin"},
    }


@app.get("/tasks", tags=["Tasks"])
async def list_tasks():
    """Return the list of available tasks with graders."""
    return {
        "tasks": {
            name: {"description": info.get("description", ""), "grader": info.get("grader", "builtin")}
            for name, info in _tasks_config.items()
        }
    }


@app.post("/grade", tags=["Tasks"])
async def grade_task(request: dict = None):
    """
    Grade the current episode. Call after an episode is done (compile_and_measure called).
    Optionally pass {"task_id": "easy"} to specify which task is being graded.
    """
    if request is None:
        request = {}
    task_id = request.get("task_id", "easy")
    env = CompilerEnvironment()
    env.reset(task_id=task_id)
    # Run a minimal episode: just compile_and_measure with no passes
    env.step(CompilerAction(action="compile_and_measure"))
    score = env.grade(task_id)
    return {"task_id": task_id, "score": score, "episode_reward": env.episode_reward}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m compiler_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn compiler_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
