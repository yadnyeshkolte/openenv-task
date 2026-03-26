# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the API Integration Debugging Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /tasks: List all tasks with action schema
    - POST /grader: Get grader score for current episode
    - POST /baseline: Run baseline inference on all tasks

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os
from typing import Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import ApiDebugAction, ApiDebugObservation
    from .api_debug_env_environment import ApiDebugEnvironment
except ImportError:
    from models import ApiDebugAction, ApiDebugObservation
    from server.api_debug_env_environment import ApiDebugEnvironment

try:
    from ..scenarios import get_all_task_ids, get_scenario
except ImportError:
    from scenarios import get_all_task_ids, get_scenario


# ─── Create the core OpenEnv app ─────────────────────────────────────────────

app = create_app(
    ApiDebugEnvironment,
    ApiDebugAction,
    ApiDebugObservation,
    env_name="api_debug_env",
    max_concurrent_envs=3,
)

# ─── Root endpoint (required: hackathon validator pings / and expects 200) ────

@app.get("/")
async def root():
    """Root endpoint — returns environment info and available endpoints."""
    return {
        "name": "api_debug_env",
        "description": "API Integration Debugging Environment",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health", "/schema", "/docs"],
    }


# ─── Hackathon-required endpoints ─────────────────────────────────────────────

# Store environment instances per task for grading
_grading_envs: Dict[str, ApiDebugEnvironment] = {}


class GraderRequest(BaseModel):
    task_id: str = "easy"


class BaselineRequest(BaseModel):
    api_key: Optional[str] = None


@app.get("/tasks")
async def list_tasks():
    """Return list of all tasks with action schema."""
    tasks = []
    for task_id in get_all_task_ids():
        scenario = get_scenario(task_id)
        tasks.append({
            "task_id": task_id,
            "difficulty": scenario.difficulty,
            "description": scenario.description,
            "max_steps": scenario.max_steps,
            "issues_count": len(scenario.issues),
            "services": scenario.services,
            "action_schema": {
                "action_type": {
                    "type": "string",
                    "enum": ["inspect_logs", "inspect_config", "inspect_endpoint", "submit_fix"],
                },
                "target": {
                    "type": "string",
                    "enum": scenario.services,
                },
                "fix_payload": {
                    "type": "object",
                    "required": False,
                },
            },
        })
    return {"tasks": tasks}


@app.post("/grader")
async def run_grader(request: GraderRequest):
    """Return grader score for a completed episode."""
    task_id = request.task_id

    if task_id in _grading_envs:
        env = _grading_envs[task_id]
        score = env.grade()
        return {
            "task_id": task_id,
            "score": score,
            "issues_fixed": len(env._issues_fixed),
            "issues_total": len(env._scenario.issues) if env._scenario else 0,
            "steps_used": env._state.step_count,
        }

    return {
        "task_id": task_id,
        "score": 0.0,
        "message": "No completed episode found. Run the environment first.",
    }


@app.post("/baseline")
async def run_baseline(request: BaselineRequest):
    """
    Run a simple rule-based baseline agent on all tasks.
    Returns baseline scores for each task.
    """
    results = {}

    for task_id in get_all_task_ids():
        env = ApiDebugEnvironment(task_id=task_id)
        obs = env.reset()

        # Simple baseline strategy: inspect all logs, then all configs, then submit fixes
        for service in obs.available_targets:
            if env._done:
                break
            obs = env.step(ApiDebugAction(
                action_type="inspect_logs",
                target=service,
            ))

        for service in obs.available_targets:
            if env._done:
                break
            obs = env.step(ApiDebugAction(
                action_type="inspect_config",
                target=service,
            ))

        for service in obs.available_targets:
            if env._done:
                break
            obs = env.step(ApiDebugAction(
                action_type="inspect_endpoint",
                target=service,
            ))

        # Store for grading
        _grading_envs[task_id] = env
        score = env.grade()

        results[task_id] = {
            "score": score,
            "steps_used": env._state.step_count,
            "issues_found": len(env._issues_found),
            "issues_fixed": len(env._issues_fixed),
            "issues_total": len(env._scenario.issues) if env._scenario else 0,
        }

    return {"baseline_scores": results}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the server directly."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
