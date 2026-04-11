# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Comprehensive tests for the API Integration Debugging Environment.

Tests cover:
- Environment reset and initialization
- Action handling (inspect_logs, inspect_config, inspect_endpoint, submit_fix)
- Grading formula correctness
- Fix validation (strict value matching)
- Episode termination conditions
- Repeated inspection penalty
- Seed-based reproducibility
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ApiDebugAction, ApiDebugObservation
from server.api_debug_env_environment import ApiDebugEnvironment
from scenarios import get_scenario, get_all_task_ids, Issue


# ─── Scenario Tests ──────────────────────────────────────────────────────────


class TestScenarios:
    """Test scenario loading and configuration."""

    def test_all_task_ids_returns_three(self):
        task_ids = get_all_task_ids()
        assert task_ids == ["easy", "medium", "hard"]

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_scenario_loads(self, task_id):
        scenario = get_scenario(task_id)
        assert scenario.task_id == task_id
        assert len(scenario.issues) > 0
        assert len(scenario.services) > 0
        assert scenario.max_steps > 0

    def test_invalid_task_id_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            get_scenario("nonexistent")

    def test_easy_has_two_issues(self):
        s = get_scenario("easy")
        assert len(s.issues) == 2

    def test_medium_has_three_issues(self):
        s = get_scenario("medium")
        assert len(s.issues) == 3

    def test_hard_has_five_issues(self):
        s = get_scenario("hard")
        assert len(s.issues) == 5

    def test_seed_randomization_shuffles_logs(self):
        """Same seed should produce same order, different seed different order."""
        s1 = get_scenario("easy", seed=42)
        s2 = get_scenario("easy", seed=42)
        s3 = get_scenario("easy", seed=99)

        # Same seed = same log order
        for service in s1.services:
            assert s1.logs.get(service) == s2.logs.get(service)

        # Different seed = potentially different order (may be same by chance,
        # but with enough log entries, it's unlikely)
        # We just verify it doesn't crash
        assert s3 is not None

    def test_each_issue_has_log_hint(self):
        """Every issue should have a corresponding log hint findable in the logs."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            for issue in s.issues:
                found = False
                for service_logs in s.logs.values():
                    for log_line in service_logs:
                        if issue.log_hint in log_line:
                            found = True
                            break
                    if found:
                        break
                assert found, f"Issue {issue.issue_id} log_hint '{issue.log_hint}' not found in any logs"


# ─── Environment Reset Tests ─────────────────────────────────────────────────


class TestEnvironmentReset:
    """Test environment initialization and reset."""

    def test_reset_returns_observation(self):
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert isinstance(obs, ApiDebugObservation)

    def test_reset_clears_state(self):
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert obs.issues_found == 0
        assert obs.issues_fixed == 0
        assert obs.done is False
        assert obs.remaining_steps == 15  # easy max_steps

    def test_reset_provides_available_targets(self):
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert len(obs.available_targets) > 0
        assert "payment_client" in obs.available_targets

    def test_reset_with_different_task(self):
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset(task_id="hard")
        assert obs.issues_total == 5

    def test_initial_reward_is_zero(self):
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert obs.reward == 0.0


# ─── Action Handler Tests ────────────────────────────────────────────────────


class TestInspectLogs:
    """Test inspect_logs action."""

    def test_inspect_logs_returns_logs(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        assert len(obs.logs) > 0

    def test_inspect_logs_finds_issues(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        assert obs.issues_found > 0
        assert obs.reward > 0  # Should get positive reward for finding issues

    def test_repeated_inspect_logs_no_reward(self):
        """Second inspection of same target should give 0 reward."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        # First inspection
        obs1 = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        # Second inspection (repeat)
        obs2 = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        # The step cost is -0.01, repeat inspect gives 0 + (-0.01) base
        assert obs2.reward < obs1.reward


class TestInspectConfig:
    """Test inspect_config action."""

    def test_inspect_config_returns_config(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="inspect_config",
            target="payment_client",
        ))
        assert len(obs.config_snapshot) > 0
        assert "headers" in obs.config_snapshot


class TestInspectEndpoint:
    """Test inspect_endpoint action."""

    def test_inspect_endpoint_shows_error(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="inspect_endpoint",
            target="payment_client",
        ))
        assert obs.api_response is not None
        assert obs.api_response["status"] == "error"


class TestSubmitFix:
    """Test submit_fix action with value validation."""

    def test_correct_fix_accepted(self):
        """Submitting the right key AND value should be accepted."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        assert obs.issues_fixed > 0
        assert "accepted" in obs.action_result.lower() or "fixed" in obs.action_result.lower()

    def test_wrong_value_rejected(self):
        """Right key but wrong value should be rejected."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "text/xml"},  # Wrong value!
        ))
        assert obs.issues_fixed == 0
        assert obs.reward < 0  # Should get negative reward

    def test_correct_auth_fix(self):
        """Bearer token fix should work with any valid token."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer my_actual_api_key_123"},
        ))
        assert obs.issues_fixed > 0

    def test_empty_payload_rejected(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={},
        ))
        assert obs.reward < 0

    def test_invalid_target_penalized(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="nonexistent_service",
            fix_payload={"key": "value"},
        ))
        assert obs.reward < 0

    def test_fix_all_issues_completes_episode(self):
        """Fixing all issues should mark episode as done with completion bonus."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        # Fix auth
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer valid_token_123"},
        ))
        # Fix content-type
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        assert obs.done is True
        assert obs.issues_fixed == 2


# ─── Grading Tests ────────────────────────────────────────────────────────────


class TestGrading:
    """Test the grading formula."""

    def test_grade_no_fixes_is_low(self):
        """Grade with no fixes should be very low (just exploration bonus)."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        env.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        score = env.grade()
        assert 0.0 < score < 0.1  # Exploration bonus only

    def test_grade_all_fixes_is_high(self):
        """Grade with all fixes should be high."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer valid_token_123"},
        ))
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score = env.grade()
        assert score > 0.8  # Should be high with efficiency bonus

    def test_grade_strictly_between_0_and_1(self):
        """Grade must be strictly in (0, 1), never exactly 0.0 or 1.0."""
        for task_id in get_all_task_ids():
            env = ApiDebugEnvironment(task_id=task_id)
            env.reset()
            score = env.grade()
            assert 0.0 < score < 1.0, f"Score for {task_id} was {score}"

    def test_efficiency_bonus(self):
        """Faster solutions should score higher."""
        # Quick partial solve (1 step, fix 1 of 2 issues)
        env1 = ApiDebugEnvironment(task_id="easy")
        env1.reset()
        env1.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score_fast = env1.grade()

        # Slow partial solve (many inspection steps, then fix same 1 issue)
        env2 = ApiDebugEnvironment(task_id="easy")
        env2.reset()
        for _ in range(10):
            env2.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        env2.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score_slow = env2.grade()

        assert score_fast > score_slow, f"Fast={score_fast} should beat Slow={score_slow}"


# ─── Episode Termination Tests ────────────────────────────────────────────────


class TestEpisodeTermination:
    """Test episode ending conditions."""

    def test_out_of_steps_ends_episode(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        # Take max_steps actions
        for _ in range(15):
            obs = env.step(ApiDebugAction(
                action_type="inspect_logs",
                target="payment_client",
            ))
        assert obs.done is True
        assert obs.remaining_steps == 0

    def test_invalid_action_type_penalized(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="nonexistent_action",
            target="payment_client",
        ))
        assert obs.reward < 0


# ─── Value Matching Tests ─────────────────────────────────────────────────────


class TestValueMatching:
    """Test the _values_match method directly."""

    def setup_method(self):
        self.env = ApiDebugEnvironment(task_id="easy")

    def test_exact_string_match(self):
        assert self.env._values_match("application/json", "application/json")

    def test_case_insensitive_match(self):
        assert self.env._values_match("Application/JSON", "application/json")

    def test_numeric_exact(self):
        assert self.env._values_match(10, 10)

    def test_numeric_tolerance(self):
        assert self.env._values_match(10, 9)  # Within 25%
        assert not self.env._values_match(10, 5)  # Outside 25%

    def test_boolean_match(self):
        assert self.env._values_match(True, True)
        assert not self.env._values_match(True, False)

    def test_boolean_from_string(self):
        assert self.env._values_match(True, "true")
        assert self.env._values_match(False, "false")

    def test_list_containment(self):
        assert self.env._values_match([429, 500], [429, 500])
        assert self.env._values_match([429, 500], [500, 429, 502])

    def test_bearer_token_pattern(self):
        assert self.env._values_match("Bearer <token>", "Bearer my_secret_key")
        assert not self.env._values_match("Bearer <token>", "Bearer ")  # Empty token

    def test_wrong_value_rejected(self):
        assert not self.env._values_match("application/json", "text/xml")
        assert not self.env._values_match(10, 100)


# ─── Integration Test ─────────────────────────────────────────────────────────


class TestFullEpisode:
    """Test a complete episode flow."""

    def test_easy_full_solve(self):
        """Run a complete easy episode from start to finish."""
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()

        # Step 1: Inspect logs
        obs = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        assert obs.issues_found >= 1

        # Step 2: Inspect config
        obs = env.step(ApiDebugAction(
            action_type="inspect_config",
            target="payment_client",
        ))
        assert "headers" in obs.config_snapshot

        # Step 3: Fix auth
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer my_token_123"},
        ))
        assert obs.issues_fixed >= 1

        # Step 4: Fix content-type
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        assert obs.issues_fixed == 2
        assert obs.done is True

        # Grade
        score = env.grade()
        assert score > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
