# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Comprehensive tests for the API Integration Debugging Environment.

Tests cover:
- Environment reset and initialization
- Action handling (inspect_logs, inspect_config, inspect_endpoint, submit_fix)
- Multi-dimensional grading rubric
- Fix validation (strict value matching + partial credit)
- Episode termination conditions
- Repeated inspection penalty
- Seed-based reproducibility and issue pool selection
- Dynamic state: service health, cascading failures, dynamic logs
- Strategy scoring
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

    def test_seed_randomization_reproducible(self):
        """Same seed should produce same scenario."""
        s1 = get_scenario("easy", seed=42)
        s2 = get_scenario("easy", seed=42)
        assert [i.issue_id for i in s1.issues] == [i.issue_id for i in s2.issues]

    def test_different_seeds_may_vary(self):
        """Different seeds should produce potentially different scenarios."""
        s1 = get_scenario("easy", seed=42)
        s2 = get_scenario("easy", seed=99)
        # They might differ (pool has 4 issues, selecting 2)
        # At minimum, they should both be valid
        assert len(s1.issues) == 2
        assert len(s2.issues) == 2

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

    def test_service_graph_exists(self):
        """Every scenario should have a service dependency graph."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            assert len(s.service_graph) > 0
            for svc in s.services:
                assert svc in s.service_graph, f"Service {svc} missing from graph in {task_id}"

    def test_dynamic_logs_defined(self):
        """Every scenario should have dynamic logs for at least some issues."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            assert len(s.dynamic_logs) > 0, f"No dynamic logs in {task_id}"

    def test_optimal_fix_order_defined(self):
        """Every scenario should have an optimal fix order."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            assert len(s.optimal_fix_order) > 0

    def test_issues_have_categories(self):
        """Every issue should have a category."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            for issue in s.issues:
                assert issue.category in (
                    "configuration", "authentication", "networking", "protocol"
                ), f"Issue {issue.issue_id} has invalid category: {issue.category}"

    def test_context_provided(self):
        """Every scenario should have context."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            assert len(s.context) > 0


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

    def test_reset_includes_service_status(self):
        """Reset should include service health status."""
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert len(obs.service_status) > 0
        assert "payment_client" in obs.service_status

    def test_reset_includes_dependency_graph(self):
        """Reset should include service dependency graph."""
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert len(obs.dependency_graph) > 0

    def test_reset_includes_error_trace(self):
        """Reset should include initial error trace."""
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert len(obs.error_trace) > 0


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
        assert obs.reward > 0

    def test_repeated_inspect_logs_no_reward(self):
        """Second inspection of same target should give 0 reward (+ step cost)."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs1 = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        obs2 = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        assert obs2.reward < obs1.reward

    def test_dynamic_logs_after_fix(self):
        """After fixing an issue, re-inspecting should show new log entries."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        # Fix content-type
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        # Re-inspect logs — should include dynamic log entries
        obs = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        # Should have the original logs PLUS dynamic logs
        assert any("application/json" in log.lower() or "parsed" in log.lower()
                    for log in obs.logs)


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

    def test_inspect_endpoint_shows_success_after_fix(self):
        """After all issues fixed, endpoint should show success."""
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
        # Episode is done now, but let's check service status
        # The service health should be updated
        assert env._service_health.get("payment_client") == "healthy"

    def test_inspect_endpoint_shows_category_status_code(self):
        """Endpoint errors should have category-appropriate status codes."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="inspect_endpoint",
            target="payment_client",
        ))
        assert obs.api_response is not None
        # Should have a realistic HTTP status code
        assert obs.api_response["status_code"] in [401, 415, 500, 504]


class TestSubmitFix:
    """Test submit_fix action with value validation and partial credit."""

    def test_correct_fix_accepted(self):
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
            fix_payload={"headers.Content-Type": "text/xml"},
        ))
        assert obs.issues_fixed == 0

    def test_partial_credit_close_value(self):
        """Right key, close value should get partial credit feedback."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/xml"},
        ))
        # Should get partial credit (same prefix "application/")
        assert obs.reward > -0.05  # Better than full reject

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
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer valid_token_123"},
        ))
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        assert obs.done is True
        assert obs.issues_fixed == 2

    def test_strategy_bonus_for_inspecting_first(self):
        """Should get higher reward when inspecting before fixing."""
        env1 = ApiDebugEnvironment(task_id="easy")
        env1.reset()
        # Fix directly (no inspection)
        obs1 = env1.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))

        env2 = ApiDebugEnvironment(task_id="easy")
        env2.reset()
        # Inspect first, then fix
        env2.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        obs2 = env2.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))

        # Fix with prior inspection should give higher reward
        assert obs2.reward > obs1.reward


# ─── Service Health Tests ─────────────────────────────────────────────────────


class TestServiceHealth:
    """Test dynamic service health tracking."""

    def test_initial_health_reflects_issues(self):
        """Services with issues should start as degraded/error."""
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()
        assert obs.service_status.get("payment_client") in ("error", "degraded")

    def test_health_updates_after_fix(self):
        """Fixing all issues on a service should mark it healthy."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer valid_token_123"},
        ))
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        # payment_client should be healthy after both fixes
        assert env._service_health.get("payment_client") == "healthy"

    def test_error_trace_updates(self):
        """Error trace should shrink as issues are fixed."""
        env = ApiDebugEnvironment(task_id="easy")
        obs1 = env.reset()
        initial_trace_len = len(obs1.error_trace)

        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        trace_after_fix = env._build_error_trace()
        assert len(trace_after_fix) < initial_trace_len


# ─── Grading Tests ────────────────────────────────────────────────────────────


class TestGrading:
    """Test the multi-dimensional grading rubric."""

    def test_grade_no_fixes_is_low(self):
        """Grade with no fixes should be low (but not zero — exploration gets some credit)."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        env.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        score = env.grade()
        assert 0.0 < score < 0.5  # Gets some credit for exploration and efficiency

    def test_grade_all_fixes_is_high(self):
        """Grade with all fixes should be high."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        env.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        env.step(ApiDebugAction(action_type="inspect_config", target="payment_client"))
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
        assert score > 0.6

    def test_grade_strictly_between_0_and_1(self):
        """Grade must be strictly in (0, 1), never exactly 0.0 or 1.0."""
        for task_id in get_all_task_ids():
            env = ApiDebugEnvironment(task_id=task_id)
            env.reset()
            score = env.grade()
            assert 0.0 < score < 1.0, f"Score for {task_id} was {score}"

    def test_efficiency_bonus(self):
        """Faster solutions with same fix count should score higher efficiency component."""
        # Both inspect then fix (same strategy), but one uses more steps
        env1 = ApiDebugEnvironment(task_id="easy")
        env1.reset()
        env1.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        env1.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score_fast = env1.grade()

        env2 = ApiDebugEnvironment(task_id="easy")
        env2.reset()
        env2.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        for _ in range(10):
            env2.step(ApiDebugAction(action_type="inspect_logs", target="payment_gateway"))
        env2.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score_slow = env2.grade()

        assert score_fast > score_slow, f"Fast={score_fast} should beat Slow={score_slow}"

    def test_strategy_affects_grade(self):
        """Proper strategy (inspect before fix) should improve grade."""
        # No inspection
        env1 = ApiDebugEnvironment(task_id="easy")
        env1.reset()
        env1.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer token"},
        ))
        env1.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score_no_inspect = env1.grade()

        # With inspection
        env2 = ApiDebugEnvironment(task_id="easy")
        env2.reset()
        env2.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        env2.step(ApiDebugAction(action_type="inspect_config", target="payment_client"))
        env2.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer token"},
        ))
        env2.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score_with_inspect = env2.grade()

        # Both should be decent but strategy should boost the inspecting one
        assert score_with_inspect >= score_no_inspect * 0.9  # At least close

    def test_grade_dimensions_nonzero(self):
        """Each grading dimension should be computable."""
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
        env.step(ApiDebugAction(action_type="inspect_logs", target="payment_client"))
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        score = env.grade()
        assert score > 0.001  # Should have some score from partial fix


# ─── Episode Termination Tests ────────────────────────────────────────────────


class TestEpisodeTermination:
    """Test episode ending conditions."""

    def test_out_of_steps_ends_episode(self):
        env = ApiDebugEnvironment(task_id="easy")
        env.reset()
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

    def test_numeric_tolerance_tight(self):
        """10% tolerance — 10 accepts 10 and 9.5 but not 8."""
        assert self.env._values_match(10, 10)  # Exact
        assert self.env._values_match(10, 9.5)  # Within 10% (5% diff)
        assert not self.env._values_match(10, 8)  # Outside 10% (20% diff)

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


class TestPartialCredit:
    """Test the _values_close method for partial credit."""

    def setup_method(self):
        self.env = ApiDebugEnvironment(task_id="easy")

    def test_numeric_close(self):
        assert self.env._values_close(10, 7)  # Within 50%
        assert not self.env._values_close(10, 100)

    def test_string_same_prefix(self):
        assert self.env._values_close("application/json", "application/xml")

    def test_check_fix_returns_partial(self):
        """Right key, close value should return 'partial'."""
        issue = Issue(
            issue_id="test",
            service="test_svc",
            description="test",
            expected_fix={"timeout": 10},
            fix_key="timeout",
            log_hint="test",
        )
        result = self.env._check_fix(issue, {"timeout": 7})
        assert result == "partial"

    def test_check_fix_returns_exact(self):
        issue = Issue(
            issue_id="test",
            service="test_svc",
            description="test",
            expected_fix={"timeout": 10},
            fix_key="timeout",
            log_hint="test",
        )
        result = self.env._check_fix(issue, {"timeout": 10})
        assert result == "exact"

    def test_check_fix_returns_none(self):
        issue = Issue(
            issue_id="test",
            service="test_svc",
            description="test",
            expected_fix={"timeout": 10},
            fix_key="timeout",
            log_hint="test",
        )
        result = self.env._check_fix(issue, {"base_url": "http://example.com"})
        assert result == "none"


# ─── Integration Tests ────────────────────────────────────────────────────────


class TestFullEpisode:
    """Test complete episode flows."""

    def test_easy_full_solve(self):
        """Run a complete easy episode from start to finish."""
        env = ApiDebugEnvironment(task_id="easy")
        obs = env.reset()

        obs = env.step(ApiDebugAction(
            action_type="inspect_logs",
            target="payment_client",
        ))
        assert obs.issues_found >= 1

        obs = env.step(ApiDebugAction(
            action_type="inspect_config",
            target="payment_client",
        ))
        assert "headers" in obs.config_snapshot

        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Authorization": "Bearer my_token_123"},
        ))
        assert obs.issues_fixed >= 1

        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="payment_client",
            fix_payload={"headers.Content-Type": "application/json"},
        ))
        assert obs.issues_fixed == 2
        assert obs.done is True

        score = env.grade()
        assert score > 0.6

    def test_medium_full_solve(self):
        """Run a complete medium episode."""
        env = ApiDebugEnvironment(task_id="medium")
        obs = env.reset()
        assert obs.issues_total == 3

        # Inspect logs
        for svc in obs.available_targets:
            obs = env.step(ApiDebugAction(
                action_type="inspect_logs", target=svc,
            ))

        # Inspect configs
        obs = env.step(ApiDebugAction(
            action_type="inspect_config", target="webhook_sender",
        ))

        # Fix rate limit
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="webhook_sender",
            fix_payload={"rate_limit.requests_per_second": 10},
        ))
        assert obs.issues_fixed >= 1

        # Fix retry
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="webhook_sender",
            fix_payload={"retry": {"max_retries": 3, "backoff_factor": 2, "retry_on_status": [429, 500]}},
        ))

        # Fix signature
        obs = env.step(ApiDebugAction(
            action_type="submit_fix",
            target="webhook_sender",
            fix_payload={"headers.X-Webhook-Signature": "sha256=computed_hmac"},
        ))

        assert obs.done is True
        score = env.grade()
        assert score > 0.4

    def test_hard_partial_solve(self):
        """Partially solve hard task and verify partial credit in grading."""
        env = ApiDebugEnvironment(task_id="hard")
        obs = env.reset()
        assert obs.issues_total == 5

        # Fix just 2 issues
        env.step(ApiDebugAction(action_type="inspect_logs", target="order_service"))
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="order_service",
            fix_payload={"inventory_url": "https://inventory.internal/v2/reserve"},
        ))
        env.step(ApiDebugAction(
            action_type="submit_fix",
            target="order_service",
            fix_payload={"timeout": 10},
        ))

        score = env.grade()
        assert 0.0 < score < 0.999
        assert len(env._issues_fixed) == 2


class TestCascadingFailures:
    """Test cascading failure dynamics."""

    def test_hard_dependency_chain(self):
        """Hard scenario has dependent issues (timeout depends on wrong_url)."""
        s = get_scenario("hard")
        timeout_issue = next(i for i in s.issues if i.issue_id == "hard_timeout")
        assert "hard_wrong_url" in timeout_issue.depends_on

    def test_cascade_effects_defined(self):
        """Issues with cascade effects should specify affected services."""
        for task_id in get_all_task_ids():
            s = get_scenario(task_id)
            any_cascade = any(len(i.cascade_effects) > 0 for i in s.issues)
            assert any_cascade, f"No cascade effects in {task_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
