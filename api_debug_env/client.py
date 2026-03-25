# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""API Integration Debugging Environment Client."""

from typing import Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ApiDebugAction, ApiDebugObservation


class ApiDebugEnv(
    EnvClient[ApiDebugAction, ApiDebugObservation, State]
):
    """
    Client for the API Integration Debugging Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with ApiDebugEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...
        ...     result = client.step(ApiDebugAction(
        ...         action_type="inspect_logs",
        ...         target="payment_client"
        ...     ))
        ...     print(result.observation.logs)
    """

    def _step_payload(self, action: ApiDebugAction) -> Dict:
        """Convert ApiDebugAction to JSON payload."""
        payload = {
            "action_type": action.action_type,
            "target": action.target,
        }
        if action.fix_payload is not None:
            payload["fix_payload"] = action.fix_payload
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[ApiDebugObservation]:
        """Parse server response into StepResult[ApiDebugObservation]."""
        obs_data = payload.get("observation", {})
        observation = ApiDebugObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            logs=obs_data.get("logs", []),
            config_snapshot=obs_data.get("config_snapshot", {}),
            api_response=obs_data.get("api_response"),
            hints=obs_data.get("hints", []),
            remaining_steps=obs_data.get("remaining_steps", 0),
            issues_found=obs_data.get("issues_found", 0),
            issues_fixed=obs_data.get("issues_fixed", 0),
            issues_total=obs_data.get("issues_total", 0),
            action_result=obs_data.get("action_result", ""),
            available_targets=obs_data.get("available_targets", []),
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
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
