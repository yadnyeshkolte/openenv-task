"""
Inference Script for API Integration Debugging Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from models import ApiDebugAction, ApiDebugObservation
from server.api_debug_env_environment import ApiDebugEnvironment
from scenarios import get_all_task_ids

# ─── Environment Variables ─────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "api_debug_env"
MAX_STEPS = 40  # max across all tasks (hard has 40)
TEMPERATURE = 0.3
MAX_TOKENS = 800
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert API debugging agent. You are tasked with diagnosing and fixing
broken API integrations. You interact with a simulated multi-service environment.

Available actions (respond with JSON):
{
  "action_type": "inspect_logs" | "inspect_config" | "inspect_endpoint" | "submit_fix",
  "target": "<service_name>",
  "fix_payload": { ... }  // required only for submit_fix
}

Strategy:
1. First inspect_logs on each service to identify error patterns
2. Then inspect_config to understand current (broken) settings
3. Use inspect_endpoint to see actual error responses
4. Submit fixes with corrected configuration values

IMPORTANT: When submitting a fix, include ALL the corrected key-value pairs in fix_payload.
For nested keys like "headers.Authorization", use the nested format:
{"headers.Authorization": "Bearer <token>"}

Respond with ONLY valid JSON. No explanation text.
""").strip()


# ─── Logging Functions ──────────────────────────────────────────────────────────

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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM Interaction ────────────────────────────────────────────────────────────

def build_user_prompt(obs: ApiDebugObservation, step: int) -> str:
    """Build a prompt from the current observation."""
    parts = [
        f"Step: {step}",
        f"Task: {obs.task_description}",
        f"Remaining steps: {obs.remaining_steps}",
        f"Issues found: {obs.issues_found}/{obs.issues_total}",
        f"Issues fixed: {obs.issues_fixed}/{obs.issues_total}",
        f"Last action result: {obs.action_result}",
        f"Available targets: {obs.available_targets}",
    ]

    if obs.logs:
        parts.append("Logs:\n" + "\n".join(obs.logs))
    if obs.config_snapshot:
        parts.append(f"Config: {json.dumps(obs.config_snapshot, indent=2)}")
    if obs.api_response:
        parts.append(f"API Response: {json.dumps(obs.api_response, indent=2)}")
    if obs.hints:
        parts.append(f"Hints: {'; '.join(obs.hints)}")

    return "\n".join(parts)


def get_model_action(
    client: OpenAI,
    obs: ApiDebugObservation,
    step: int,
    messages: List[Dict],
) -> ApiDebugAction:
    """Get next action from the LLM."""
    user_prompt = build_user_prompt(obs, step)
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Try to extract JSON from the response
        # Handle cases where model wraps JSON in markdown code blocks
        if "```" in text:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                text = text[json_start:json_end]

        action_json = json.loads(text)
        messages.append({"role": "assistant", "content": json.dumps(action_json)})

        return ApiDebugAction(
            action_type=action_json.get("action_type", "inspect_logs"),
            target=action_json.get("target", obs.available_targets[0] if obs.available_targets else ""),
            fix_payload=action_json.get("fix_payload"),
        )
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback: inspect logs of first available target
        fallback_target = obs.available_targets[0] if obs.available_targets else ""
        return ApiDebugAction(
            action_type="inspect_logs",
            target=fallback_target,
        )


# ─── Main Execution ─────────────────────────────────────────────────────────────

async def run_task(task_id: str, client: OpenAI) -> tuple:
    """Run a single task and return (score, rewards, steps)."""
    env = ApiDebugEnvironment(task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action = get_model_action(client, obs, step, messages)
            action_str = f"{action.action_type}(target={action.target})"
            if action.fix_payload:
                action_str = f"{action.action_type}(target={action.target}, fix={json.dumps(action.fix_payload)})"

            obs = env.step(action)

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = env.grade()  # already clamped to (0.001, 0.999)
        score = max(0.001, min(0.999, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, rewards, steps_taken


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = get_all_task_ids()  # ["easy", "medium", "hard"]

    for task_id in task_ids:
        await run_task(task_id, client)


if __name__ == "__main__":
    asyncio.run(main())
