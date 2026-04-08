# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
API Integration Debugging Environment Implementation.

A real-world environment where an AI agent diagnoses and fixes broken
API integrations by reading error logs, inspecting configurations,
and submitting corrected configurations.
"""

import copy
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ApiDebugAction, ApiDebugObservation
    from ..scenarios import Issue, Scenario, get_all_task_ids, get_scenario
except ImportError:
    from models import ApiDebugAction, ApiDebugObservation
    from scenarios import Issue, Scenario, get_all_task_ids, get_scenario


class ApiDebugEnvironment(Environment):
    """
    API Integration Debugging Environment.

    An agent must diagnose and fix broken API integrations by:
    1. Inspecting error logs to identify issues
    2. Inspecting service configurations
    3. Testing endpoints to observe failures
    4. Submitting configuration fixes

    Supports 3 difficulty levels (easy, medium, hard) with different
    numbers of issues and complexity.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "easy"):
        """
        Initialize the environment.

        Args:
            task_id: One of 'easy', 'medium', 'hard'
        """
        self._task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Scenario] = None
        self._current_configs: Dict[str, Dict[str, Any]] = {}
        self._issues_found: Set[str] = set()
        self._issues_fixed: Set[str] = set()
        self._inspected_targets: Set[str] = set()
        self._done = False
        self._last_action_result = ""
        self._cumulative_reward = 0.0

    def reset(self, task_id: Optional[str] = None) -> ApiDebugObservation:
        """
        Reset the environment, optionally with a new task.

        Args:
            task_id: Override the task difficulty. One of 'easy', 'medium', 'hard'.

        Returns:
            Initial observation with task description and available targets.
        """
        if task_id is not None:
            self._task_id = task_id

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = get_scenario(self._task_id)
        self._current_configs = copy.deepcopy(self._scenario.configs)
        self._issues_found = set()
        self._issues_fixed = set()
        self._inspected_targets = set()
        self._done = False
        self._last_action_result = ""
        self._cumulative_reward = 0.0

        return ApiDebugObservation(
            task_id=self._task_id,
            task_description=self._scenario.description,
            logs=[],
            config_snapshot={},
            api_response=None,
            hints=self._get_hints(),
            remaining_steps=self._scenario.max_steps,
            issues_found=0,
            issues_fixed=0,
            issues_total=len(self._scenario.issues),
            action_result="Environment reset. Use 'inspect_logs' or 'inspect_config' to start debugging.",
            available_targets=self._scenario.services,
            done=False,
            reward=0.0,
        )

    def step(self, action: ApiDebugAction) -> ApiDebugObservation:  # type: ignore[override]
        """
        Execute one debugging step.

        Args:
            action: ApiDebugAction with action_type, target, and optional fix_payload

        Returns:
            ApiDebugObservation with results of the action
        """
        if self._scenario is None:
            # Auto-reset if not initialized
            self.reset()

        assert self._scenario is not None  # for type checker

        self._state.step_count += 1
        reward = 0.0
        logs: List[str] = []
        config_snapshot: Dict[str, Any] = {}
        api_response: Optional[Dict[str, Any]] = None

        # Validate target
        if action.target not in self._scenario.services:
            self._last_action_result = (
                f"Invalid target '{action.target}'. "
                f"Valid targets: {self._scenario.services}"
            )
            reward = -0.05
        elif action.action_type == "inspect_logs":
            logs, reward = self._handle_inspect_logs(action.target)
        elif action.action_type == "inspect_config":
            config_snapshot, reward = self._handle_inspect_config(action.target)
        elif action.action_type == "inspect_endpoint":
            api_response, reward = self._handle_inspect_endpoint(action.target)
        elif action.action_type == "submit_fix":
            reward = self._handle_submit_fix(action.target, action.fix_payload or {})
        else:
            self._last_action_result = (
                f"Invalid action_type '{action.action_type}'. "
                "Valid types: inspect_logs, inspect_config, inspect_endpoint, submit_fix"
            )
            reward = -0.05

        self._cumulative_reward += reward

        # Check episode termination
        remaining = self._scenario.max_steps - self._state.step_count
        all_fixed = len(self._issues_fixed) == len(self._scenario.issues)

        if all_fixed:
            self._done = True
            reward += 0.2  # completion bonus
            self._cumulative_reward += 0.2
            self._last_action_result += " 🎉 All issues fixed! Episode complete."

        if remaining <= 0 and not self._done:
            self._done = True
            self._last_action_result += " ⏰ Out of steps. Episode ended."

        return ApiDebugObservation(
            task_id=self._task_id,
            task_description=self._scenario.description,
            logs=logs,
            config_snapshot=config_snapshot,
            api_response=api_response,
            hints=self._get_hints(),
            remaining_steps=max(0, remaining),
            issues_found=len(self._issues_found),
            issues_fixed=len(self._issues_fixed),
            issues_total=len(self._scenario.issues),
            action_result=self._last_action_result,
            available_targets=self._scenario.services,
            done=self._done,
            reward=reward,
            metadata={
                "cumulative_reward": self._cumulative_reward,
                "step": self._state.step_count,
                "issues_found_ids": list(self._issues_found),
                "issues_fixed_ids": list(self._issues_fixed),
            },
        )

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state

    # ─── Action Handlers ──────────────────────────────────────────────────

    def _handle_inspect_logs(self, target: str) -> tuple:
        """Return logs for a service and reward for relevant inspection."""
        assert self._scenario is not None
        logs = self._scenario.logs.get(target, [])
        self._inspected_targets.add(f"logs:{target}")

        # Check if any unfound issues have log hints in these logs
        found_new = False
        for issue in self._scenario.issues:
            if issue.issue_id not in self._issues_found:
                for log_line in logs:
                    if issue.log_hint in log_line:
                        self._issues_found.add(issue.issue_id)
                        found_new = True

        if found_new:
            reward = 0.15
            self._last_action_result = f"Inspected logs for '{target}'. Found relevant error patterns!"
        elif logs:
            reward = 0.05
            self._last_action_result = f"Inspected logs for '{target}'. {len(logs)} log entries found."
        else:
            reward = 0.0
            self._last_action_result = f"No logs available for '{target}'."

        return logs, reward

    def _handle_inspect_config(self, target: str) -> tuple:
        """Return current config for a service."""
        assert self._scenario is not None
        config = self._current_configs.get(target, {})
        self._inspected_targets.add(f"config:{target}")

        # Small reward for inspecting a service that has issues
        has_issues = any(i.service == target for i in self._scenario.issues if i.issue_id not in self._issues_fixed)
        reward = 0.05 if has_issues else 0.02

        self._last_action_result = f"Inspected config for '{target}'. Configuration retrieved."
        return config, reward

    def _handle_inspect_endpoint(self, target: str) -> tuple:
        """Simulate testing an endpoint and return the response."""
        assert self._scenario is not None

        # Find unfixed issues for this service
        unfixed = [
            i for i in self._scenario.issues
            if i.service == target and i.issue_id not in self._issues_fixed
        ]

        if unfixed:
            # Simulate a failure based on the first unfixed issue
            issue = unfixed[0]
            api_response = {
                "status": "error",
                "status_code": 401 if "auth" in issue.issue_id else 500,
                "error": issue.description,
                "hint": f"Check the {issue.fix_key} configuration",
            }
            reward = 0.05
            self._last_action_result = f"Tested endpoint on '{target}'. Got error response."
        else:
            api_response = {
                "status": "success",
                "status_code": 200,
                "message": f"{target} is working correctly.",
            }
            reward = 0.02
            self._last_action_result = f"Tested endpoint on '{target}'. Service responding OK."

        return api_response, reward

    def _handle_submit_fix(self, target: str, fix_payload: Dict[str, Any]) -> float:
        """Process a fix submission and score it."""
        assert self._scenario is not None

        if not fix_payload:
            self._last_action_result = "Fix rejected: fix_payload cannot be empty."
            return -0.1

        # Find issues for this target service
        target_issues = [
            i for i in self._scenario.issues
            if i.service == target and i.issue_id not in self._issues_fixed
        ]

        if not target_issues:
            self._last_action_result = f"No unfixed issues found for '{target}'."
            return -0.05

        reward = 0.0
        fixed_any = False

        for issue in target_issues:
            if self._check_fix(issue, fix_payload):
                self._issues_fixed.add(issue.issue_id)
                self._issues_found.add(issue.issue_id)  # finding + fixing counts
                self._apply_fix(target, fix_payload)
                reward += 0.25
                fixed_any = True

        if fixed_any:
            fixed_count = sum(1 for i in target_issues if i.issue_id in self._issues_fixed)
            self._last_action_result = (
                f"Fix accepted for '{target}'! "
                f"Fixed {fixed_count} issue(s). "
                f"Total fixed: {len(self._issues_fixed)}/{len(self._scenario.issues)}"
            )
        else:
            self._last_action_result = (
                f"Fix rejected for '{target}'. The payload doesn't address any known issues. "
                "Try inspecting logs and config to identify the correct fix."
            )
            reward = -0.1

        return reward

    # ─── Helper Methods ───────────────────────────────────────────────────

    def _check_fix(self, issue: Issue, fix_payload: Dict[str, Any]) -> bool:
        """
        Check if a fix payload correctly addresses an issue.

        Uses fuzzy matching — the fix is accepted if:
        1. The fix_key is present in the payload, OR
        2. Any expected_fix key is present in the payload with a reasonable value
        """
        # Direct key match
        if issue.fix_key in fix_payload:
            return True

        # Check nested key (e.g., "headers.Authorization" -> check payload for "Authorization")
        if "." in issue.fix_key:
            parts = issue.fix_key.split(".")
            leaf_key = parts[-1]
            if leaf_key in fix_payload:
                return True

        # Check expected fix keys
        for key in issue.expected_fix:
            if key in fix_payload:
                return True
            if "." in key:
                leaf = key.split(".")[-1]
                if leaf in fix_payload:
                    return True

        return False

    def _apply_fix(self, target: str, fix_payload: Dict[str, Any]) -> None:
        """Apply a fix to the current configuration."""
        if target not in self._current_configs:
            return

        config = self._current_configs[target]
        for key, value in fix_payload.items():
            if "." in key:
                # Nested key: e.g., "headers.Authorization"
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    if part not in obj:
                        obj[part] = {}
                    obj = obj[part]
                obj[parts[-1]] = value
            else:
                config[key] = value

    def _get_hints(self) -> List[str]:
        """Return progressive hints based on step count."""
        if self._scenario is None:
            return []

        hints = []
        step = self._state.step_count
        total_issues = len(self._scenario.issues)
        unfixed = total_issues - len(self._issues_fixed)

        if step == 0:
            hints.append("Start by inspecting error logs for each service to find clues.")
            hints.append(f"There are {total_issues} issues to find and fix.")
        elif step > 0 and len(self._issues_found) == 0:
            hints.append("Try 'inspect_logs' on different services to find error patterns.")
        elif len(self._issues_found) > 0 and len(self._issues_fixed) == 0:
            hints.append("You've found issues! Use 'inspect_config' to see current settings, then 'submit_fix'.")
        elif unfixed > 0:
            hints.append(f"{unfixed} issue(s) remaining. Check services you haven't inspected yet.")

        # Late-game hints
        if self._scenario.max_steps - step <= 5 and unfixed > 0:
            # Give more specific hints when running low on steps
            for issue in self._scenario.issues:
                if issue.issue_id not in self._issues_fixed:
                    hints.append(f"Hint: Check '{issue.service}' — look for '{issue.fix_key}' in the config.")

        return hints

    # ─── Grading ──────────────────────────────────────────────────────────

    def grade(self) -> float:
        """
        Grade the agent's performance on the current episode.

        Score = (issues_fixed / issues_total) * efficiency_bonus + exploration_bonus
        Efficiency bonus = 1.0 + (remaining_steps / max_steps * 0.3)
        Exploration bonus = small credit for inspecting services (max 0.05)

        Returns:
            Score strictly between 0 and 1 (exclusive): in range (0.001, 0.999)
        """
        if self._scenario is None:
            return 0.001

        total = len(self._scenario.issues)
        if total == 0:
            return 0.999

        fix_ratio = len(self._issues_fixed) / total
        remaining = max(0, self._scenario.max_steps - self._state.step_count)
        efficiency_bonus = 1.0 + (remaining / self._scenario.max_steps * 0.3)

        # Small partial credit for exploration even if no fixes submitted
        exploration_bonus = min(0.05, len(self._inspected_targets) * 0.005)

        score = fix_ratio * efficiency_bonus + exploration_bonus

        # Clamp strictly to (0.001, 0.999) — NEVER exactly 0.0 or 1.0
        return max(0.001, min(0.999, round(score, 4)))

    def get_task_info(self) -> Dict[str, Any]:
        """Return information about the current task."""
        if self._scenario is None:
            return {"error": "Environment not initialized. Call reset() first."}

        return {
            "task_id": self._task_id,
            "difficulty": self._scenario.difficulty,
            "description": self._scenario.description,
            "max_steps": self._scenario.max_steps,
            "issues_total": len(self._scenario.issues),
            "services": self._scenario.services,
            "action_schema": {
                "action_type": {
                    "type": "string",
                    "enum": ["inspect_logs", "inspect_config", "inspect_endpoint", "submit_fix"],
                    "description": "The type of debugging action to take",
                },
                "target": {
                    "type": "string",
                    "enum": self._scenario.services,
                    "description": "The service to act on",
                },
                "fix_payload": {
                    "type": "object",
                    "description": "Configuration fix (required for submit_fix action)",
                    "required": False,
                },
            },
        }
