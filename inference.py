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

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_memory.json")
MAX_GOOD_SEQUENCES = 10
MAX_BAD_SEQUENCES = 10
MAX_INSIGHTS = 15

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("COMPILER_ENV_BENCHMARK", "compiler_env")
MAX_STEPS = 30
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.1

# Max possible reward: 0.1 * 19 (all unique passes improve) + 1.0 (beat O3) = 2.9
MAX_TOTAL_REWARD = 2.9

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert LLVM compiler optimization agent. Your goal is to select
    LLVM optimization passes that minimize the execution time of a C program
    compiled for RISC-V architecture.

    IMPORTANT: Each add_pass is automatically compiled and measured.
    - If the pass IMPROVES execution time -> it is KEPT in the sequence (+0.1 reward)
    - If it does NOT improve -> it is AUTO-REMOVED (-0.05 reward)
    - Duplicate passes are REJECTED (-0.1 reward)
    You get immediate feedback after each add_pass showing the measured time.

    Each turn you must reply with exactly ONE tool call. Available tools:
    - add_pass(pass_name): Try adding a pass. Auto-compiled and evaluated.
    - remove_pass(pass_name): Remove a previously kept pass (-0.05 penalty).
    - finalize(): End the episode. You receive a final bonus based on baselines.
    - get_program_info(): Get program name and baseline execution times
    - list_passes(): List all 19 valid LLVM optimization passes
    - get_current_sequence(): Get kept passes and current best time

    Available 19 passes (use exact names):
    MEMORY:
      mem2reg     — Promotes stack allocations to SSA registers (foundational, apply first)
      sroa        — Scalar Replacement of Aggregates: breaks structs into scalars
    CFG:
      simplifycfg — Simplifies control flow: merges blocks, removes dead branches
    SCALAR:
      instcombine — Peephole optimizations: combines/simplifies instructions
      gvn         — Global Value Numbering: eliminates redundant computations
      sccp        — Sparse Conditional Constant Propagation: folds constants
      dce         — Dead Code Elimination: removes unused instructions
      dse         — Dead Store Elimination: removes writes never read
      early-cse   — Common Subexpression Elimination: removes duplicate expressions
      reassociate — Reassociates arithmetic for better constant folding
    LOOP:
      loop-simplify — Canonicalizes loops into standard form (prerequisite for other loop opts)
      loop-rotate   — Rotates loops to do-while form (enables better unrolling)
      loop-unroll   — Unrolls loops to reduce branch overhead (big perf impact)
      licm          — Loop Invariant Code Motion: hoists invariant code out of loops
      indvars       — Simplifies induction variables in loops
      loop-reduce   — Loop Strength Reduction: replaces expensive ops (multiply→add)
    INLINE:
      inline        — Inlines function calls to eliminate call overhead
    RISC-V:
      consthoist    — Hoists expensive constants to reduce register pressure
      div-rem-pairs — Combines division+remainder into single operation

    Strategy:
    1. Start with foundational passes: mem2reg, sroa, simplifycfg, instcombine
    2. Add scalar optimizations: gvn, dce, dse, early-cse, sccp, reassociate
    3. Add loop optimizations: loop-simplify, loop-rotate, licm, indvars, loop-unroll, loop-reduce
    4. Try inlining: inline
    5. Try RISC-V specific: consthoist, div-rem-pairs
    6. Read the feedback after each add_pass to understand what helps
    7. Call finalize() when you've tried enough passes or are satisfied
    8. Order matters: mem2reg should come before most other passes
    9. Do NOT add the same pass twice — duplicates are penalized
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
            "name": "remove_pass",
            "description": "Remove a previously kept pass from the sequence. Use this to undo a bad pass choice. Costs -0.05 reward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pass_name": {
                        "type": "string",
                        "description": "Name of the LLVM pass to remove from the kept sequence",
                    }
                },
                "required": ["pass_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "End the episode. Computes final bonus based on how your best time compares to O0/O1/O2/O3 baselines. Call this when you are satisfied with your pass sequence.",
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
            "description": "Get the current kept pass sequence and best execution time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ------------------------------------------------------------------
# Agent Memory
# ------------------------------------------------------------------

def load_memory() -> dict:
    """Load agent memory from JSON file."""
    default = {"good_sequences": [], "bad_sequences": [], "insights": []}
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_memory(memory: dict) -> None:
    """Save agent memory to JSON file."""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"[DEBUG] Failed to save memory: {e}", flush=True)


def format_memory_for_prompt(memory: dict) -> str:
    """Format memory as a readable string for the LLM system prompt."""
    parts = []

    if memory.get("good_sequences"):
        parts.append("GOOD SEQUENCES (these worked well before):")
        for entry in memory["good_sequences"][-5:]:
            seq = ", ".join(entry["sequence"])
            parts.append(f"  - [{seq}] improvement={entry.get('improvement', '?')} program={entry.get('program_type', '?')}")

    if memory.get("bad_sequences"):
        parts.append("BAD SEQUENCES (avoid these):")
        for entry in memory["bad_sequences"][-5:]:
            seq = ", ".join(entry["sequence"])
            parts.append(f"  - [{seq}] reason={entry.get('reason', '?')}")

    if memory.get("insights"):
        parts.append("INSIGHTS FROM PREVIOUS EPISODES:")
        for insight in memory["insights"][-5:]:
            parts.append(f"  - {insight}")

    return "\n".join(parts) if parts else "No previous experience yet."


def update_memory(
    memory: dict,
    program_name: str,
    kept_passes: List[str],
    removed_passes: List[str],
    best_time: float,
    baseline_O0: float,
    tier_beaten: str,
) -> dict:
    """Update agent memory after an episode."""
    # Compute improvement percentage
    if baseline_O0 and baseline_O0 > 0 and best_time < baseline_O0:
        improvement_pct = ((baseline_O0 - best_time) / baseline_O0) * 100
        improvement_str = f"+{improvement_pct:.1f}%"
    else:
        improvement_str = "0%"

    # Add good sequence if any passes were kept
    if kept_passes:
        memory["good_sequences"].append({
            "sequence": kept_passes,
            "improvement": improvement_str,
            "program_type": program_name,
        })
        # Trim to max size
        memory["good_sequences"] = memory["good_sequences"][-MAX_GOOD_SEQUENCES:]

    # Add bad sequences (passes that were removed)
    if removed_passes:
        memory["bad_sequences"].append({
            "sequence": removed_passes,
            "reason": f"No improvement on {program_name}",
        })
        memory["bad_sequences"] = memory["bad_sequences"][-MAX_BAD_SEQUENCES:]

    # Auto-generate insights
    if kept_passes and tier_beaten != "none":
        first_pass = kept_passes[0]
        memory["insights"].append(
            f"Starting with {first_pass} worked well on {program_name} (beat {tier_beaten})"
        )
    if len(kept_passes) == 0:
        memory["insights"].append(
            f"No passes helped on {program_name} — may need different approach"
        )
    memory["insights"] = memory["insights"][-MAX_INSIGHTS:]

    return memory


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
    if tool_name == "remove_pass":
        return f"remove_pass:{tool_args.get('pass_name', '')}"
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
    task_id: str, env: CompilerEnv, llm_client: OpenAI, memory: dict
) -> dict:
    """Run a single task (one [START]...[END] block). Returns updated memory."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False
    kept_passes: List[str] = []
    removed_passes: List[str] = []
    program_name = "unknown"
    baseline_O0 = 0.0
    best_time = 0.0
    tier_beaten = "none"

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        obs_data = obs.data
        program_name = obs_data.get("program", "unknown")
        baseline_O0 = obs_data.get("baselines", {}).get("O0", 0.0)
        best_time = obs_data.get("best_time", baseline_O0)

        # Format memory context for the LLM
        memory_context = format_memory_for_prompt(memory)

        # Build initial conversation with program info + memory
        history_messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"You are optimizing: {program_name}\n"
                    f"Baseline execution times:\n"
                    f"  O0: {obs_data.get('baselines', {}).get('O0', '?')}s\n"
                    f"  O1: {obs_data.get('baselines', {}).get('O1', '?')}s\n"
                    f"  O2: {obs_data.get('baselines', {}).get('O2', '?')}s\n"
                    f"  O3: {obs_data.get('baselines', {}).get('O3', '?')}s\n"
                    f"Current best time: {obs_data.get('best_time', '?')}s (starts at O0)\n"
                    f"Steps remaining: {obs_data.get('steps_remaining', MAX_STEPS)}\n\n"
                    f"--- MEMORY FROM PREVIOUS EPISODES ---\n"
                    f"{memory_context}\n"
                    f"--- END MEMORY ---\n\n"
                    f"Add passes one at a time. Each is auto-compiled and measured.\n"
                    f"Only passes that improve execution time are kept.\n"
                    f"Use insights from memory to guide your choices.\n"
                    f"Call finalize() when satisfied."
                ),
            },
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Ask LLM for next tool call
            fn_name, fn_args, raw_tc = get_model_tool_call(llm_client, history_messages)

            if fn_name is None:
                # No tool call -- agent stopped; force finalize
                action_str = "finalize"
            else:
                action_str = tool_call_to_action(fn_name, fn_args)

            # Step the environment
            result = await env.step(CompilerAction(action=action_str))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = obs.message if obs.status in ("error", "invalid_pass", "compile_failed", "duplicate") else None

            # Track kept/removed passes for memory
            if obs.status == "pass_kept":
                kept_passes.append(obs.data.get("pass", ""))
                best_time = obs.data.get("best_time", best_time)
            elif obs.status == "pass_removed":
                removed_passes.append(obs.data.get("pass", ""))
            elif obs.status == "finalized":
                tier_beaten = obs.data.get("tier_beaten", "none")
                best_time = obs.data.get("final_time", best_time)

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

    # Update memory with episode results
    memory = update_memory(
        memory, program_name, kept_passes, removed_passes,
        best_time, baseline_O0, tier_beaten,
    )
    return memory


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await create_env()
    memory = load_memory()

    try:
        for task_id in TASK_IDS:
            memory = await run_task(task_id, env, llm_client, memory)
            save_memory(memory)
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
