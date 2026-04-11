# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the API Integration Debugging Environment.

An agent must diagnose and fix broken API integrations by reading error logs,
inspecting configurations, and writing corrected API calls.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ApiDebugAction(Action):
    """
    Agent action — what the agent does each step.

    Supported action_type values:
      - "inspect_logs"     : Read error logs for a specific service
      - "inspect_config"   : Inspect the config of a specific service/endpoint
      - "inspect_endpoint" : Test-call an endpoint to see current response
      - "submit_fix"       : Submit a fix (requires fix_payload)
    """

    action_type: str = Field(
        ...,
        description="One of: 'inspect_logs', 'inspect_config', 'inspect_endpoint', 'submit_fix'",
    )
    target: str = Field(
        ...,
        description="The service or component to act on (e.g. 'auth_service', 'webhook_handler', 'service_a')",
    )
    fix_payload: Optional[Dict] = Field(
        default=None,
        description="Required when action_type='submit_fix'. Dict with the corrected configuration.",
    )


class ApiDebugObservation(Observation):
    """
    What the agent sees after each action.

    Provides error logs, configuration snapshots, API responses,
    and progress tracking for the debugging task.
    """

    # Environment context
    task_id: str = Field(default="", description="Current task identifier (easy/medium/hard)")
    task_description: str = Field(default="", description="Human-readable description of what needs debugging")

    # Inspection results
    logs: List[str] = Field(default_factory=list, description="Error log lines visible to the agent")
    config_snapshot: Dict = Field(default_factory=dict, description="Current configuration of the inspected component")
    api_response: Optional[Dict] = Field(default=None, description="Response from testing the current endpoint config")
    hints: List[str] = Field(default_factory=list, description="Progressive hints based on step count")

    # Progress tracking
    remaining_steps: int = Field(default=0, description="Steps remaining before episode timeout")
    issues_found: int = Field(default=0, description="Issues the agent has correctly identified so far")
    issues_fixed: int = Field(default=0, description="Issues the agent has correctly fixed so far")
    issues_total: int = Field(default=0, description="Total issues in the current scenario")

    # Feedback
    action_result: str = Field(default="", description="Feedback on the last action taken (e.g. 'Fix accepted', 'Wrong fix')")
    available_targets: List[str] = Field(default_factory=list, description="List of valid targets the agent can inspect/fix")
