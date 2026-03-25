"""
Baseline inference script for the API Integration Debugging Environment.

This script demonstrates an LLM-powered agent interacting with the environment
using the OpenAI API. It runs all 3 tasks (easy, medium, hard) and reports
baseline scores.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY=your-key-here

    # Run baseline
    python scripts/baseline_inference.py

    # Or specify a server URL
    python scripts/baseline_inference.py --server-url http://localhost:8000
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ApiDebugAction, ApiDebugObservation
from scenarios import get_all_task_ids, get_scenario
from server.api_debug_env_environment import ApiDebugEnvironment


def run_rule_based_baseline(task_id: str) -> Dict[str, Any]:
    """
    Run a simple rule-based baseline agent (no LLM needed).

    Strategy:
    1. Inspect all logs
    2. Inspect all configs
    3. Test all endpoints
    (Does not attempt fixes — tests reward signal for exploration-only behavior)
    """
    env = ApiDebugEnvironment(task_id=task_id)
    obs = env.reset()
    total_reward = 0.0

    # Phase 1: Inspect all logs
    for service in obs.available_targets:
        if obs.done:
            break
        obs = env.step(ApiDebugAction(action_type="inspect_logs", target=service))
        total_reward += obs.reward

    # Phase 2: Inspect all configs
    for service in obs.available_targets:
        if obs.done:
            break
        obs = env.step(ApiDebugAction(action_type="inspect_config", target=service))
        total_reward += obs.reward

    # Phase 3: Test all endpoints
    for service in obs.available_targets:
        if obs.done:
            break
        obs = env.step(ApiDebugAction(action_type="inspect_endpoint", target=service))
        total_reward += obs.reward

    score = env.grade()
    return {
        "task_id": task_id,
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps_used": env._state.step_count,
        "issues_found": len(env._issues_found),
        "issues_fixed": len(env._issues_fixed),
        "issues_total": len(env._scenario.issues) if env._scenario else 0,
    }


def run_llm_baseline(task_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Run an LLM-powered baseline agent using OpenAI API.

    The LLM reads observations and decides what to do next.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI package not installed. Running rule-based baseline instead.")
        return run_rule_based_baseline(task_id)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("No OPENAI_API_KEY set. Running rule-based baseline instead.")
        return run_rule_based_baseline(task_id)

    client = OpenAI(api_key=key)
    env = ApiDebugEnvironment(task_id=task_id)
    obs = env.reset()
    total_reward = 0.0

    system_prompt = f"""You are an API debugging agent. Your task: {obs.task_description}

Available actions:
- inspect_logs: Read error logs for a service
- inspect_config: See the configuration of a service
- inspect_endpoint: Test-call an endpoint
- submit_fix: Submit a config fix (requires fix_payload dict)

Available targets: {obs.available_targets}
Total issues to fix: {obs.issues_total}

Respond with JSON: {{"action_type": "...", "target": "...", "fix_payload": {{...}} }}
Only include fix_payload when action_type is "submit_fix"."""

    messages = [{"role": "system", "content": system_prompt}]

    while not obs.done:
        # Build observation message
        obs_text = f"""Step {env._state.step_count}/{env._scenario.max_steps if env._scenario else '?'}
Remaining steps: {obs.remaining_steps}
Issues found: {obs.issues_found}/{obs.issues_total}
Issues fixed: {obs.issues_fixed}/{obs.issues_total}
Last action result: {obs.action_result}"""

        if obs.logs:
            obs_text += f"\nLogs:\n" + "\n".join(obs.logs)
        if obs.config_snapshot:
            obs_text += f"\nConfig: {json.dumps(obs.config_snapshot, indent=2)}"
        if obs.api_response:
            obs_text += f"\nAPI Response: {json.dumps(obs.api_response, indent=2)}"
        if obs.hints:
            obs_text += f"\nHints: {'; '.join(obs.hints)}"

        messages.append({"role": "user", "content": obs_text})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            action_json = json.loads(response.choices[0].message.content)
            messages.append({"role": "assistant", "content": json.dumps(action_json)})

            action = ApiDebugAction(
                action_type=action_json.get("action_type", "inspect_logs"),
                target=action_json.get("target", obs.available_targets[0] if obs.available_targets else ""),
                fix_payload=action_json.get("fix_payload"),
            )
        except Exception as e:
            print(f"  LLM error: {e}. Falling back to inspect_logs.")
            action = ApiDebugAction(
                action_type="inspect_logs",
                target=obs.available_targets[0] if obs.available_targets else "",
            )

        obs = env.step(action)
        total_reward += obs.reward

    score = env.grade()
    return {
        "task_id": task_id,
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps_used": env._state.step_count,
        "issues_found": len(env._issues_found),
        "issues_fixed": len(env._issues_fixed),
        "issues_total": len(env._scenario.issues) if env._scenario else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for API Debug Env")
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule",
                        help="Baseline mode: 'rule' for rule-based, 'llm' for LLM-powered")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run specific task only (easy/medium/hard)")
    args = parser.parse_args()

    print("=" * 60)
    print("API Integration Debugging — Baseline Inference")
    print("=" * 60)

    task_ids = [args.task] if args.task else get_all_task_ids()
    all_results = {}

    for task_id in task_ids:
        print(f"\n{'─' * 40}")
        print(f"Task: {task_id}")
        print(f"{'─' * 40}")

        if args.mode == "llm":
            result = run_llm_baseline(task_id, args.api_key)
        else:
            result = run_rule_based_baseline(task_id)

        all_results[task_id] = result
        print(f"  Score:        {result['score']}")
        print(f"  Reward:       {result['total_reward']}")
        print(f"  Steps:        {result['steps_used']}")
        print(f"  Issues found: {result['issues_found']}/{result['issues_total']}")
        print(f"  Issues fixed: {result['issues_fixed']}/{result['issues_total']}")

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for tid, res in all_results.items():
        print(f"  {tid:8s}  score={res['score']:.4f}  fixed={res['issues_fixed']}/{res['issues_total']}")

    avg_score = sum(r["score"] for r in all_results.values()) / len(all_results)
    print(f"\n  Average score: {avg_score:.4f}")

    return all_results


if __name__ == "__main__":
    main()
