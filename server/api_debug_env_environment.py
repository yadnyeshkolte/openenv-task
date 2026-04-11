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

Key design features:
- Dynamic state: fixing issues changes service health and produces new logs
- Cascading failures: upstream fixes reveal downstream issues
- Multi-dimensional rubric grading (diagnosis, fix, efficiency, strategy)
- Rich reward signal with partial credit and diminishing returns
"""

import copy
from typing import Any, Dict, List, Optional, Set, Tuple
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

    Supports 3 difficulty levels (easy, medium, hard) with cascading
    failure dynamics and multi-dimensional grading.
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
        # Dynamic state tracking
        self._service_health: Dict[str, str] = {}
        self._dynamic_log_buffer: Dict[str, List[str]] = {}
        # Strategy tracking for grading
        self._action_history: List[Dict[str, Any]] = []
        self._diagnosed_before_fix: Set[str] = set()
        # Track which services were inspected before a fix was submitted

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> ApiDebugObservation:
        """
        Reset the environment, optionally with a new task.

        Args:
            task_id: Override the task difficulty. One of 'easy', 'medium', 'hard'.
            seed: Optional seed for reproducible randomized scenarios.

        Returns:
            Initial observation with task description and available targets.
        """
        if task_id is not None:
            self._task_id = task_id

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = get_scenario(self._task_id, seed=seed)
        self._current_configs = copy.deepcopy(self._scenario.configs)
        self._issues_found = set()
        self._issues_fixed = set()
        self._inspected_targets = set()
        self._done = False
        self._last_action_result = ""
        self._cumulative_reward = 0.0
        self._action_history = []
        self._diagnosed_before_fix = set()

        # Initialize service health from scenario graph
        self._service_health = {}
        for svc_name, node in self._scenario.service_graph.items():
            self._service_health[svc_name] = node.health_status
        # Fill in any services not in graph
        for svc in self._scenario.services:
            if svc not in self._service_health:
                self._service_health[svc] = "unknown"

        # Initialize dynamic log buffer
        self._dynamic_log_buffer = {svc: [] for svc in self._scenario.services}

        # Build dependency graph for observation
        dep_graph = {}
        for svc_name, node in self._scenario.service_graph.items():
            dep_graph[svc_name] = node.depends_on

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
            service_status=dict(self._service_health),
            dependency_graph=dep_graph,
            error_trace=self._build_error_trace(),
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
        reward = -0.01  # Small step cost to encourage efficiency
        logs: List[str] = []
        config_snapshot: Dict[str, Any] = {}
        api_response: Optional[Dict[str, Any]] = None

        # Record action for strategy scoring
        self._action_history.append({
            "step": self._state.step_count,
            "action_type": action.action_type,
            "target": action.target,
        })

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

        # Build dependency graph
        dep_graph = {}
        for svc_name, node in self._scenario.service_graph.items():
            dep_graph[svc_name] = node.depends_on

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
            service_status=dict(self._service_health),
            dependency_graph=dep_graph,
            error_trace=self._build_error_trace(),
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
        # Combine static logs with dynamic logs from fixes
        static_logs = self._scenario.logs.get(target, [])
        dynamic_logs = self._dynamic_log_buffer.get(target, [])
        logs = static_logs + dynamic_logs

        inspect_key = f"logs:{target}"
        is_repeat = inspect_key in self._inspected_targets
        self._inspected_targets.add(inspect_key)

        # Track that this service was inspected (for strategy scoring)
        self._diagnosed_before_fix.add(target)

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
        elif is_repeat and not dynamic_logs:
            reward = 0.0  # No reward for re-inspecting same logs with no changes
            self._last_action_result = f"Re-inspected logs for '{target}'. No new information."
        elif is_repeat and dynamic_logs:
            reward = 0.05  # Some reward for checking updated logs
            self._last_action_result = f"Re-inspected logs for '{target}'. New entries found after recent fixes."
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
        inspect_key = f"config:{target}"
        is_repeat = inspect_key in self._inspected_targets
        self._inspected_targets.add(inspect_key)

        # Track that this service was inspected (for strategy scoring)
        self._diagnosed_before_fix.add(target)

        # Reward based on relevance and novelty
        has_issues = any(
            i.service == target
            for i in self._scenario.issues
            if i.issue_id not in self._issues_fixed
        )
        if is_repeat:
            reward = 0.0  # No reward for re-inspecting same config
            self._last_action_result = f"Re-inspected config for '{target}'. No changes since last check."
        elif has_issues:
            reward = 0.05
            self._last_action_result = f"Inspected config for '{target}'. Configuration retrieved."
        else:
            reward = 0.01
            self._last_action_result = f"Inspected config for '{target}'. No issues detected in this service."

        return config, reward

    def _handle_inspect_endpoint(self, target: str) -> tuple:
        """Simulate testing an endpoint. Response changes based on current fix state."""
        assert self._scenario is not None

        # Track that this service was inspected
        self._diagnosed_before_fix.add(target)

        # Find unfixed issues for this service
        unfixed = [
            i for i in self._scenario.issues
            if i.service == target and i.issue_id not in self._issues_fixed
        ]

        # Also check if any DEPENDENCY issues are unfixed (cascade simulation)
        upstream_broken = False
        if target in self._scenario.service_graph:
            node = self._scenario.service_graph[target]
            for dep_svc in node.depends_on:
                dep_unfixed = [
                    i for i in self._scenario.issues
                    if i.service == dep_svc and i.issue_id not in self._issues_fixed
                ]
                if dep_unfixed:
                    upstream_broken = True

        if unfixed:
            issue = unfixed[0]
            # Determine status code based on issue category
            status_codes = {
                "authentication": 401,
                "protocol": 415,
                "networking": 504,
                "configuration": 500,
            }
            status_code = status_codes.get(issue.category, 500)
            api_response = {
                "status": "error",
                "status_code": status_code,
                "error": issue.description,
                "hint": f"Check the {issue.fix_key} configuration for '{target}'",
                "service_health": self._service_health.get(target, "unknown"),
            }
            reward = 0.05
            self._last_action_result = f"Tested endpoint on '{target}'. Got {status_code} error response."
        elif upstream_broken:
            api_response = {
                "status": "degraded",
                "status_code": 503,
                "error": f"{target} configuration is correct but upstream dependencies are failing.",
                "hint": "Fix upstream services first — check the dependency graph.",
                "service_health": "degraded",
            }
            reward = 0.03
            self._last_action_result = f"Tested '{target}'. Service config OK but upstream is broken."
        else:
            api_response = {
                "status": "success",
                "status_code": 200,
                "message": f"{target} is working correctly.",
                "service_health": "healthy",
            }
            reward = 0.02
            self._last_action_result = f"Tested endpoint on '{target}'. Service responding OK."

        return api_response, reward

    def _handle_submit_fix(self, target: str, fix_payload: Dict[str, Any]) -> float:
        """Process a fix submission with strict validation and cascade effects."""
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
        partial_credit = False

        # Check if the agent inspected this service before submitting
        inspected_first = target in self._diagnosed_before_fix

        for issue in target_issues:
            match_result = self._check_fix(issue, fix_payload)
            if match_result == "exact":
                self._issues_fixed.add(issue.issue_id)
                self._issues_found.add(issue.issue_id)
                self._apply_fix(target, fix_payload)
                self._update_service_health(issue)
                self._inject_dynamic_logs(issue)
                reward += 0.25
                fixed_any = True
                # Bonus for inspecting before fixing (strategy reward)
                if inspected_first:
                    reward += 0.05
            elif match_result == "partial":
                # Right key, close value — give partial credit
                partial_credit = True
                reward += 0.03

        if fixed_any:
            fixed_count = sum(1 for i in target_issues if i.issue_id in self._issues_fixed)
            self._last_action_result = (
                f"Fix accepted for '{target}'! "
                f"Fixed {fixed_count} issue(s). "
                f"Total fixed: {len(self._issues_fixed)}/{len(self._scenario.issues)}"
            )
        elif partial_credit:
            self._last_action_result = (
                f"Fix partially correct for '{target}'. "
                "The key is right but the value isn't quite right. Check the logs for exact values."
            )
        else:
            self._last_action_result = (
                f"Fix rejected for '{target}'. The payload doesn't address any known issues. "
                "Try inspecting logs and config to identify the correct fix."
            )
            reward = -0.1

        return reward

    # ─── Dynamic State Methods ────────────────────────────────────────────

    def _update_service_health(self, fixed_issue: Issue) -> None:
        """Update service health status after an issue is fixed."""
        assert self._scenario is not None

        # Check if the fixed service has any remaining issues
        remaining = [
            i for i in self._scenario.issues
            if i.service == fixed_issue.service and i.issue_id not in self._issues_fixed
        ]
        if not remaining:
            self._service_health[fixed_issue.service] = "healthy"
        else:
            self._service_health[fixed_issue.service] = "degraded"

        # Update downstream services affected by cascade
        for affected_svc, _effect in fixed_issue.cascade_effects.items():
            if affected_svc in self._service_health:
                # Check if the affected service still has its own issues
                svc_issues = [
                    i for i in self._scenario.issues
                    if i.service == affected_svc and i.issue_id not in self._issues_fixed
                ]
                if not svc_issues:
                    # Check if all upstream deps are healthy
                    if affected_svc in self._scenario.service_graph:
                        upstream_healthy = all(
                            self._service_health.get(dep, "error") == "healthy"
                            for dep in self._scenario.service_graph[affected_svc].depends_on
                        )
                        if upstream_healthy:
                            self._service_health[affected_svc] = "healthy"
                        else:
                            self._service_health[affected_svc] = "degraded"
                    else:
                        self._service_health[affected_svc] = "healthy"

    def _inject_dynamic_logs(self, fixed_issue: Issue) -> None:
        """Inject new log entries after an issue is fixed."""
        assert self._scenario is not None
        if fixed_issue.issue_id in self._scenario.dynamic_logs:
            for svc, new_logs in self._scenario.dynamic_logs[fixed_issue.issue_id].items():
                if svc in self._dynamic_log_buffer:
                    self._dynamic_log_buffer[svc].extend(new_logs)

    def _build_error_trace(self) -> List[str]:
        """Build an error propagation trace showing cascade chain."""
        if self._scenario is None:
            return []

        trace = []
        for issue in self._scenario.issues:
            if issue.issue_id not in self._issues_fixed:
                trace.append(
                    f"[{issue.severity.upper()}] {issue.service}: {issue.description}"
                )
                for affected_svc, effect in issue.cascade_effects.items():
                    trace.append(f"  └─> {affected_svc}: {effect}")

        if not trace:
            trace.append("All issues resolved. No error cascades active.")

        return trace

    # ─── Helper Methods ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Normalize a value for comparison (lowercase strings, sort lists, etc.)."""
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, list):
            return sorted([ApiDebugEnvironment._normalize_value(v) for v in value], key=str)
        if isinstance(value, dict):
            return {k: ApiDebugEnvironment._normalize_value(v) for k, v in value.items()}
        return value

    def _values_match(self, expected: Any, submitted: Any) -> bool:
        """
        Check if a submitted value matches the expected value.

        Supports:
        - Exact match
        - Case-insensitive string match
        - Numeric tolerance (10%)
        - Boolean coercion (e.g., "true" -> True)
        - List containment (submitted must contain all expected elements)
        - Pattern match for token-like values (Bearer <anything> matches Bearer <token>)
        """
        # Normalize both
        norm_expected = self._normalize_value(expected)
        norm_submitted = self._normalize_value(submitted)

        # Exact match after normalization
        if norm_expected == norm_submitted:
            return True

        # Numeric comparison with tolerance (10% — tighter than before)
        if isinstance(expected, (int, float)) and isinstance(submitted, (int, float)):
            if expected == 0:
                return submitted == 0
            return abs(expected - submitted) / max(abs(expected), 1) < 0.10

        # Boolean coercion
        if isinstance(expected, bool):
            if isinstance(submitted, str):
                return submitted.lower() in ("true", "1", "yes") if expected else submitted.lower() in ("false", "0", "no")
            return bool(submitted) == expected

        # String pattern match for tokens: "Bearer <token>" matches "Bearer <anything>"
        if isinstance(expected, str) and isinstance(submitted, str):
            exp_lower = expected.strip().lower()
            sub_lower = submitted.strip().lower()
            # If expected has a placeholder like <token>, accept any non-empty value
            if "<" in exp_lower and ">" in exp_lower:
                prefix = exp_lower.split("<")[0].strip()
                if prefix and sub_lower.startswith(prefix) and len(sub_lower) > len(prefix):
                    return True
            # If submitted has same prefix structure
            if exp_lower.startswith("bearer ") and sub_lower.startswith("bearer "):
                return len(sub_lower) > len("bearer ")

        # List: submitted must contain all expected elements
        if isinstance(expected, list) and isinstance(submitted, list):
            return all(any(self._values_match(e, s) for s in submitted) for e in expected)

        return False

    def _values_close(self, expected: Any, submitted: Any) -> bool:
        """Check if values are 'close' for partial credit (same type, right ballpark)."""
        if isinstance(expected, (int, float)) and isinstance(submitted, (int, float)):
            if expected == 0:
                return abs(submitted) < 5
            return abs(expected - submitted) / max(abs(expected), 1) < 0.50
        if isinstance(expected, str) and isinstance(submitted, str):
            # Same prefix / similar structure
            return expected.split("/")[0].lower() == submitted.split("/")[0].lower()
        if isinstance(expected, bool) and isinstance(submitted, bool):
            return True  # Right type at least
        return False

    def _check_fix(self, issue: Issue, fix_payload: Dict[str, Any]) -> str:
        """
        Check if a fix payload correctly addresses an issue.

        Returns:
            'exact' if fix is correct
            'partial' if fix has right key but wrong value
            'none' if fix doesn't match at all
        """
        found_key = False

        # Direct key match with value validation
        if issue.fix_key in fix_payload:
            found_key = True
            expected_val = issue.expected_fix.get(issue.fix_key)
            if expected_val is not None:
                if self._values_match(expected_val, fix_payload[issue.fix_key]):
                    return "exact"
                elif self._values_close(expected_val, fix_payload[issue.fix_key]):
                    return "partial"
                return "none"  # Right key, wrong value

            # If the submitted value is a dict and expected_fix has nested keys
            submitted_val = fix_payload[issue.fix_key]
            if isinstance(submitted_val, dict):
                nested_prefix = issue.fix_key + "."
                nested_expected = {
                    k[len(nested_prefix):]: v
                    for k, v in issue.expected_fix.items()
                    if k.startswith(nested_prefix)
                }
                if nested_expected:
                    all_match = all(
                        k in submitted_val and self._values_match(v, submitted_val[k])
                        for k, v in nested_expected.items()
                    )
                    if all_match:
                        return "exact"
                    # Check partial
                    any_match = any(
                        k in submitted_val and self._values_match(v, submitted_val[k])
                        for k, v in nested_expected.items()
                    )
                    if any_match:
                        return "partial"
                    return "none"

            # No expected value found — this shouldn't happen with well-defined issues
            # Do NOT accept blindly — require value validation
            return "none"

        # Check nested key (e.g., "headers.Authorization" -> check payload for "Authorization")
        if "." in issue.fix_key:
            parts = issue.fix_key.split(".")
            leaf_key = parts[-1]
            if leaf_key in fix_payload:
                found_key = True
                expected_val = issue.expected_fix.get(issue.fix_key)
                if expected_val is not None:
                    if self._values_match(expected_val, fix_payload[leaf_key]):
                        return "exact"
                    elif self._values_close(expected_val, fix_payload[leaf_key]):
                        return "partial"
                    return "none"
                return "none"

        # Check expected fix keys with value validation
        for key, expected_val in issue.expected_fix.items():
            # Direct key in payload
            if key in fix_payload:
                found_key = True
                if self._values_match(expected_val, fix_payload[key]):
                    return "exact"
            # Nested key leaf match
            if "." in key:
                leaf = key.split(".")[-1]
                if leaf in fix_payload:
                    found_key = True
                    if self._values_match(expected_val, fix_payload[leaf]):
                        return "exact"

        if found_key:
            return "partial"  # Found the key but value didn't match
        return "none"

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
        """Return progressive hints based on step count and progress."""
        if self._scenario is None:
            return []

        hints = []
        step = self._state.step_count
        total_issues = len(self._scenario.issues)
        unfixed = total_issues - len(self._issues_fixed)

        if step == 0:
            hints.append("Start by inspecting error logs for each service to find clues.")
            hints.append(f"There are {total_issues} issues to find and fix.")
            if self._scenario.context:
                hints.append(f"Context: {self._scenario.context}")
        elif step > 0 and len(self._issues_found) == 0:
            hints.append("Try 'inspect_logs' on different services to find error patterns.")
        elif len(self._issues_found) > 0 and len(self._issues_fixed) == 0:
            hints.append("You've found issues! Use 'inspect_config' to see current settings, then 'submit_fix'.")
        elif unfixed > 0:
            hints.append(f"{unfixed} issue(s) remaining. Check services you haven't inspected yet.")

        # Dependency hints
        for issue in self._scenario.issues:
            if issue.issue_id not in self._issues_fixed and issue.depends_on:
                deps_met = all(d in self._issues_fixed for d in issue.depends_on)
                if not deps_met:
                    dep_names = [
                        next((i.service for i in self._scenario.issues if i.issue_id == d), d)
                        for d in issue.depends_on
                    ]
                    if len(self._issues_fixed) > 0:
                        hints.append(
                            f"Some issues may be masked by upstream failures. "
                            f"Check services: {', '.join(set(dep_names))}"
                        )
                        break

        # Late-game hints
        if self._scenario.max_steps - step <= 5 and unfixed > 0:
            for issue in self._scenario.issues:
                if issue.issue_id not in self._issues_fixed:
                    hints.append(
                        f"Hint: Check '{issue.service}' — look for '{issue.fix_key}' in the config."
                    )

        return hints

    # ─── Multi-Dimensional Grading ────────────────────────────────────────

    def grade(self) -> float:
        """
        Grade the agent's performance using a multi-dimensional rubric.

        Score = weighted_average(
            diagnosis_score × 0.20,    # Did the agent inspect before fixing?
            fix_score × 0.40,          # Issues fixed / total
            efficiency_score × 0.15,   # Steps used vs available
            strategy_score × 0.25,     # Logical debugging approach
        )

        Returns:
            Score strictly between 0 and 1 (exclusive): in range (0.001, 0.999)
        """
        if self._scenario is None:
            return 0.001

        total = len(self._scenario.issues)
        if total == 0:
            return 0.999

        # 1. Fix Score (40% weight) — most important
        fix_ratio = len(self._issues_fixed) / total
        fix_score = fix_ratio

        # 2. Diagnosis Score (20% weight) — did you inspect before fixing?
        if self._issues_fixed:
            diagnosed_count = sum(
                1 for issue_id in self._issues_fixed
                if any(
                    i.service in self._diagnosed_before_fix
                    for i in self._scenario.issues
                    if i.issue_id == issue_id
                )
            )
            diagnosis_score = diagnosed_count / len(self._issues_fixed)
        else:
            # Give partial credit for exploration even without fixes
            diagnosis_score = min(1.0, len(self._inspected_targets) / max(1, len(self._scenario.services)))

        # 3. Efficiency Score (15% weight) — faster is better
        remaining = max(0, self._scenario.max_steps - self._state.step_count)
        efficiency_score = remaining / self._scenario.max_steps

        # 4. Strategy Score (25% weight) — logical debugging approach
        strategy_score = self._compute_strategy_score()

        # Weighted combination
        score = (
            fix_score * 0.40 +
            diagnosis_score * 0.20 +
            efficiency_score * 0.15 +
            strategy_score * 0.25
        )

        # Clamp strictly to (0.001, 0.999) — NEVER exactly 0.0 or 1.0
        return max(0.001, min(0.999, round(score, 4)))

    def _compute_strategy_score(self) -> float:
        """
        Score the agent's debugging strategy.

        Good strategy:
        - Inspect logs before configs (logs have more diagnostic info)
        - Don't repeat the same inspection
        - Fix issues in dependency order
        - Don't submit fixes without inspecting first
        """
        if not self._action_history:
            return 0.0

        score = 0.0
        total_checks = 0

        # Check 1: Did the agent inspect logs before submitting any fix?
        first_fix_step = None
        first_inspect_step = None
        for action in self._action_history:
            if action["action_type"] == "submit_fix" and first_fix_step is None:
                first_fix_step = action["step"]
            if action["action_type"] in ("inspect_logs", "inspect_config") and first_inspect_step is None:
                first_inspect_step = action["step"]

        total_checks += 1
        if first_inspect_step is not None and (first_fix_step is None or first_inspect_step < first_fix_step):
            score += 1.0  # Inspected before fixing

        # Check 2: Ratio of unique inspections to total inspections
        total_inspections = sum(
            1 for a in self._action_history
            if a["action_type"] in ("inspect_logs", "inspect_config", "inspect_endpoint")
        )
        unique_inspections = len(self._inspected_targets)
        total_checks += 1
        if total_inspections > 0:
            score += min(1.0, unique_inspections / total_inspections)

        # Check 3: Did fixes follow dependency order?
        if self._scenario and self._scenario.optimal_fix_order and len(self._issues_fixed) > 1:
            total_checks += 1
            fix_order = []
            for action in self._action_history:
                if action["action_type"] == "submit_fix":
                    # Find which issue was fixed in this step
                    for issue_id in self._issues_fixed:
                        issue = next((i for i in self._scenario.issues if i.issue_id == issue_id), None)
                        if issue and issue_id not in fix_order:
                            fix_order.append(issue_id)

            # Compare fix order with optimal order
            optimal = [o for o in self._scenario.optimal_fix_order if o in fix_order]
            if len(optimal) > 1:
                in_order = sum(
                    1 for i in range(len(fix_order) - 1)
                    if fix_order[i] in optimal and fix_order[i+1] in optimal
                    and optimal.index(fix_order[i]) < optimal.index(fix_order[i+1])
                )
                score += in_order / max(1, len(fix_order) - 1)

        # Check 4: Did the agent use a variety of action types?
        total_checks += 1
        action_types_used = set(a["action_type"] for a in self._action_history)
        score += len(action_types_used) / 4.0  # 4 possible action types

        return score / total_checks if total_checks > 0 else 0.0

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
            "service_dependencies": {
                svc: node.depends_on
                for svc, node in self._scenario.service_graph.items()
            },
            "context": self._scenario.context,
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
