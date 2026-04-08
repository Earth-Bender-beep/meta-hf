"""
Inference Script for IRis Compiler Optimization Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named inference.py and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from compiler_env import CompilerAction, CompilerEnv
except ImportError:
    # Running directly from repo root (not installed as package)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import CompilerEnv
    from models import CompilerAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("COMPILER_ENV_BENCHMARK", "compiler_env")
MAX_STEPS = 50
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.1

# Max possible reward: 0.05 * 50 (all valid passes) + 1.0 (beat O3) = 3.5
MAX_TOTAL_REWARD = 3.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert LLVM compiler optimization agent. Your goal is to select
    a sequence of LLVM optimization passes that minimizes the execution time of
    a C program compiled for RISC-V architecture.

    Each turn you must reply with exactly ONE tool call. Available tools:
    - add_pass(pass_name): Add an LLVM optimization pass to the sequence
    - compile_and_measure(): Compile with current passes and measure performance (ends episode)
    - get_program_info(): Get program name and baseline execution times
    - list_passes(): List all 44 valid LLVM optimization passes
    - get_current_sequence(): Get the current pass sequence built so far

    Strategy:
    1. Start with foundational passes: mem2reg, simplifycfg, instcombine
    2. Add loop optimizations for loop-heavy programs: licm, loop-unroll, indvars
    3. Use gvn for redundancy elimination
    4. When satisfied, call compile_and_measure to test your sequence
    5. Order matters: mem2reg should come before most other passes
    """
).strip()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_pass",
            "description": "Add an LLVM optimization pass to the current sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pass_name": {
                        "type": "string",
                        "description": "Name of the LLVM pass (e.g. 'mem2reg', 'licm', 'gvn')",
                    }
                },
                "required": ["pass_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_and_measure",
            "description": "Compile the program with the current pass sequence and measure execution time. Ends the episode.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_program_info",
            "description": "Get information about the current program and its baseline execution times.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_passes",
            "description": "List all valid LLVM optimization passes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_sequence",
            "description": "Get the current pass sequence built so far.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def tool_call_to_action(tool_name: str, tool_args: dict) -> str:
    """Convert an OpenAI tool call into an environment action string."""
    if tool_name == "add_pass":
        return f"add_pass:{tool_args.get('pass_name', '')}"
    return tool_name


def get_model_tool_call(client: OpenAI, messages: list) -> tuple:
    """Ask the LLM for its next tool call. Returns (tool_name, tool_args, raw_tool_call)."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        msg = completion.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return None, None, msg

        tc = msg.tool_calls[0]
        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        return fn_name, fn_args, tc

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None, None, None


TASK_IDS = ["easy", "medium", "hard"]


async def create_env() -> CompilerEnv:
    """Create environment client, trying multiple connection methods."""
    # 1. If an explicit URL is provided, connect directly
    env_url = os.getenv("OPENENV_URL") or os.getenv("ENV_URL")
    if env_url:
        print(f"[DEBUG] Connecting to env URL: {env_url}", flush=True)
        client = CompilerEnv(base_url=env_url)
        await client.connect()
        return client

    # 2. If LOCAL_IMAGE_NAME is set, start from Docker image
    if LOCAL_IMAGE_NAME:
        print(f"[DEBUG] Starting from Docker image: {LOCAL_IMAGE_NAME}", flush=True)
        return await CompilerEnv.from_docker_image(LOCAL_IMAGE_NAME)

    # 3. Fallback: try localhost:8000 (container already running)
    print("[DEBUG] No image/URL set, trying localhost:8000", flush=True)
    client = CompilerEnv(base_url="http://localhost:8000")
    await client.connect()
    return client


async def run_task(
    task_id: str, env: CompilerEnv, llm_client: OpenAI
) -> None:
    """Run a single task (one [START]...[END] block)."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        obs_data = obs.data

        # Build initial conversation with program info
        history_messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"You are optimizing: {obs_data.get('program', 'unknown')}\n"
                    f"Baseline execution times:\n"
                    f"  O0: {obs_data.get('baselines', {}).get('O0', '?')}s\n"
                    f"  O1: {obs_data.get('baselines', {}).get('O1', '?')}s\n"
                    f"  O2: {obs_data.get('baselines', {}).get('O2', '?')}s\n"
                    f"  O3: {obs_data.get('baselines', {}).get('O3', '?')}s\n"
                    f"Steps remaining: {obs_data.get('steps_remaining', MAX_STEPS)}\n\n"
                    f"Select LLVM optimization passes to minimize execution time. "
                    f"When ready, call compile_and_measure."
                ),
            },
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Ask LLM for next tool call
            fn_name, fn_args, raw_tc = get_model_tool_call(llm_client, history_messages)

            if fn_name is None:
                # No tool call -- agent stopped; force compile_and_measure
                action_str = "compile_and_measure"
            else:
                action_str = tool_call_to_action(fn_name, fn_args)

            # Step the environment
            result = await env.step(CompilerAction(action=action_str))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = obs.message if obs.status in ("error", "invalid_pass", "failed") else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Feed tool result back to LLM conversation
            if raw_tc and hasattr(raw_tc, "id"):
                tool_result_str = json.dumps({
                    "status": obs.status,
                    "message": obs.message,
                    "data": obs.data,
                    "reward": reward,
                    "done": done,
                })
                history_messages.append({
                    "role": "tool",
                    "tool_call_id": raw_tc.id,
                    "content": tool_result_str,
                })

            if done:
                break

        # Compute score: normalize cumulative reward to [0, 1]
        total_reward = rewards[-1] if rewards else 0.0  # last reward is cumulative
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await create_env()

    try:
        for task_id in TASK_IDS:
            await run_task(task_id, env, llm_client)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[END] success=false steps=0 score=0.001 rewards=", flush=True)
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
        sys.exit(1)
