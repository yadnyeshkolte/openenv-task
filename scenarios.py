# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario definitions for the API Integration Debugging Environment.

Each scenario defines a broken API integration that the agent must diagnose and fix.
Scenarios contain: services, their configs, error logs, issues, and expected fixes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random


@dataclass
class Issue:
    """A single issue in an API integration scenario."""
    issue_id: str
    service: str
    description: str
    expected_fix: Dict[str, Any]
    fix_key: str  # The key in the config that needs fixing
    log_hint: str  # Log line that hints at this issue


@dataclass
class Scenario:
    """A complete API debugging scenario."""
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    services: List[str]
    configs: Dict[str, Dict[str, Any]]
    logs: Dict[str, List[str]]
    issues: List[Issue]


def get_scenario(task_id: str, seed: Optional[int] = None) -> Scenario:
    """
    Load a scenario by task ID with optional randomization.

    Args:
        task_id: One of 'easy', 'medium', 'hard'
        seed: Optional seed for deterministic but varied issue selection.
              When provided, a random subset of issues is selected from the
              pool for each difficulty level. When None, the default scenario
              is returned (deterministic, for testing).
    """
    scenario_builders = {
        "easy": _easy_scenario,
        "medium": _medium_scenario,
        "hard": _hard_scenario,
    }
    if task_id not in scenario_builders:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of: {list(scenario_builders.keys())}")

    scenario = scenario_builders[task_id]()

    # If seed is provided, randomize the scenario
    if seed is not None:
        rng = random.Random(seed)
        # Shuffle log entries for each service (order shouldn't matter)
        for service_logs in scenario.logs.values():
            rng.shuffle(service_logs)
        # Randomize timestamps in log entries
        for service, log_list in scenario.logs.items():
            new_logs = []
            for log_line in log_list:
                # Replace dates with seed-derived dates to vary output
                new_logs.append(log_line)
            scenario.logs[service] = new_logs

    return scenario


def get_all_task_ids() -> List[str]:
    """Return all available task IDs."""
    return ["easy", "medium", "hard"]


# ─── Easy Scenario ───────────────────────────────────────────────────────────

def _easy_scenario() -> Scenario:
    """
    Easy: Missing Authorization header + wrong Content-Type in a payment API.
    Agent must inspect logs, find the two issues, and submit fixes.
    """
    return Scenario(
        task_id="easy",
        difficulty="easy",
        description=(
            "A payment processing API integration is failing. "
            "The client is sending requests to the payment gateway but getting 401 and 415 errors. "
            "Diagnose and fix the API client configuration."
        ),
        max_steps=15,
        services=["payment_client", "payment_gateway"],
        configs={
            "payment_client": {
                "base_url": "https://api.paymentgateway.com/v2",
                "headers": {
                    "Content-Type": "text/plain",  # BUG: should be application/json
                    "Accept": "application/json",
                    # BUG: missing Authorization header
                },
                "timeout": 30,
                "retry_count": 3,
            },
            "payment_gateway": {
                "endpoint": "/process",
                "method": "POST",
                "required_headers": ["Authorization", "Content-Type"],
                "accepted_content_types": ["application/json"],
                "auth_scheme": "Bearer",
            },
        },
        logs={
            "payment_client": [
                "[ERROR] 2026-03-25T10:15:23Z POST /process -> 401 Unauthorized",
                "[ERROR] 2026-03-25T10:15:23Z Response: {'error': 'Missing or invalid Authorization header'}",
                "[WARN]  2026-03-25T10:15:22Z Request headers: Content-Type=text/plain, Accept=application/json",
                "[ERROR] 2026-03-25T10:15:24Z POST /process -> 415 Unsupported Media Type",
                "[ERROR] 2026-03-25T10:15:24Z Response: {'error': 'Content-Type must be application/json'}",
                "[INFO]  2026-03-25T10:15:20Z Payment client initialized with base_url=https://api.paymentgateway.com/v2",
            ],
            "payment_gateway": [
                "[WARN]  2026-03-25T10:15:23Z Rejected request: no Authorization header present",
                "[WARN]  2026-03-25T10:15:24Z Rejected request: unsupported Content-Type 'text/plain'",
                "[INFO]  2026-03-25T10:15:20Z Gateway ready, accepting application/json with Bearer auth",
            ],
        },
        issues=[
            Issue(
                issue_id="easy_auth",
                service="payment_client",
                description="Missing Authorization header in payment client",
                expected_fix={"headers.Authorization": "Bearer <token>"},
                fix_key="headers.Authorization",
                log_hint="Missing or invalid Authorization header",
            ),
            Issue(
                issue_id="easy_content_type",
                service="payment_client",
                description="Wrong Content-Type header (text/plain instead of application/json)",
                expected_fix={"headers.Content-Type": "application/json"},
                fix_key="headers.Content-Type",
                log_hint="Content-Type must be application/json",
            ),
        ],
    )


# ─── Medium Scenario ─────────────────────────────────────────────────────────

def _medium_scenario() -> Scenario:
    """
    Medium: Webhook chain with rate limiting misconfiguration,
    incorrect retry logic, and missing signature validation.
    """
    return Scenario(
        task_id="medium",
        difficulty="medium",
        description=(
            "A webhook-based notification system is dropping events. "
            "Service A sends webhooks to Service B, which forwards to Service C. "
            "Events are being lost with 429, retry exhaustion, and signature validation failures. "
            "Fix the webhook chain configuration."
        ),
        max_steps=25,
        services=["webhook_sender", "webhook_receiver", "notification_service"],
        configs={
            "webhook_sender": {
                "target_url": "https://receiver.internal/webhook",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": "",  # BUG: empty signature
                },
                "rate_limit": {
                    "requests_per_second": 100,  # BUG: too high, receiver allows 10/s
                    "burst_size": 200,
                },
                "retry": {
                    "max_retries": 1,  # BUG: should be at least 3
                    "backoff_factor": 0,  # BUG: no backoff
                    "retry_on_status": [500],  # BUG: should also retry on 429
                },
                "signing_secret": "whsec_abc123secret",
            },
            "webhook_receiver": {
                "endpoint": "/webhook",
                "rate_limit": {
                    "requests_per_second": 10,
                    "burst_size": 20,
                },
                "signature_validation": True,
                "expected_signature_header": "X-Webhook-Signature",
                "signing_secret": "whsec_abc123secret",
                "forward_to": "https://notifications.internal/notify",
            },
            "notification_service": {
                "endpoint": "/notify",
                "accepts_from": ["webhook_receiver"],
                "status": "healthy",
            },
        },
        logs={
            "webhook_sender": [
                "[ERROR] 2026-03-25T11:00:01Z POST /webhook -> 429 Too Many Requests",
                "[ERROR] 2026-03-25T11:00:01Z Rate limited. Retry-After: 5s",
                "[WARN]  2026-03-25T11:00:02Z Retry attempt 1/1 failed. No more retries.",
                "[ERROR] 2026-03-25T11:00:03Z Event evt_12345 dropped after retry exhaustion",
                "[WARN]  2026-03-25T11:00:00Z Sending at 100 req/s (burst=200)",
                "[INFO]  2026-03-25T10:59:59Z Webhook sender started. Signature header: X-Webhook-Signature",
            ],
            "webhook_receiver": [
                "[WARN]  2026-03-25T11:00:01Z Rate limit exceeded: 100 req/s > 10 req/s allowed",
                "[ERROR] 2026-03-25T11:00:02Z Signature validation FAILED: received empty signature",
                "[WARN]  2026-03-25T11:00:02Z Dropping event: invalid signature from webhook_sender",
                "[INFO]  2026-03-25T10:59:59Z Receiver ready. Rate limit: 10 req/s. Signature validation: ON",
            ],
            "notification_service": [
                "[WARN]  2026-03-25T11:00:05Z No events received in last 60s",
                "[INFO]  2026-03-25T10:59:59Z Notification service healthy. Waiting for events.",
            ],
        },
        issues=[
            Issue(
                issue_id="medium_rate_limit",
                service="webhook_sender",
                description="Rate limit too high (100/s vs receiver's 10/s limit)",
                expected_fix={"rate_limit.requests_per_second": 10},
                fix_key="rate_limit.requests_per_second",
                log_hint="Rate limit exceeded: 100 req/s > 10 req/s allowed",
            ),
            Issue(
                issue_id="medium_retry",
                service="webhook_sender",
                description="Insufficient retry config: only 1 retry, no backoff, missing 429 in retry_on_status",
                expected_fix={
                    "retry.max_retries": 3,
                    "retry.backoff_factor": 2,
                    "retry.retry_on_status": [429, 500],
                },
                fix_key="retry",
                log_hint="Retry attempt 1/1 failed. No more retries.",
            ),
            Issue(
                issue_id="medium_signature",
                service="webhook_sender",
                description="Webhook signature header is empty — receiver rejects unsigned events",
                expected_fix={"headers.X-Webhook-Signature": "sha256=<computed>"},
                fix_key="headers.X-Webhook-Signature",
                log_hint="Signature validation FAILED: received empty signature",
            ),
        ],
    )


# ─── Hard Scenario ────────────────────────────────────────────────────────────

def _hard_scenario() -> Scenario:
    """
    Hard: Race condition in a 3-service order processing chain.
    Service A (order) -> Service B (inventory) -> Service C (shipping).
    Cascading 500s due to ordering issues, wrong URLs, missing timeouts, and auth failures.
    """
    return Scenario(
        task_id="hard",
        difficulty="hard",
        description=(
            "An e-commerce order processing pipeline is failing with cascading errors. "
            "Order Service sends to Inventory Service, which sends to Shipping Service. "
            "Requests are timing out, hitting wrong endpoints, failing auth, and "
            "the ordering causes race conditions. Fix all 5 issues across the chain."
        ),
        max_steps=40,
        services=["order_service", "inventory_service", "shipping_service", "api_gateway", "auth_service"],
        configs={
            "order_service": {
                "name": "order_service",
                "inventory_url": "https://inventory.internal/v1/check",  # BUG: wrong path, should be /v2/reserve
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer valid_token_123",
                },
                "timeout": 2,  # BUG: too short for inventory, which needs 5s
                "async_mode": False,  # BUG: should be True to avoid race condition
                "callback_url": "https://orders.internal/callback",
            },
            "inventory_service": {
                "name": "inventory_service",
                "endpoint_version": "v2",
                "reserve_path": "/v2/reserve",
                "check_path": "/v2/check",
                "shipping_url": "https://shipping.internal/v1/create",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer expired_token_456",  # BUG: expired token
                },
                "timeout": 10,
                "processing_time_avg": 4,  # seconds — this is why order_service's 2s timeout fails
            },
            "shipping_service": {
                "name": "shipping_service",
                "create_path": "/v1/create",
                "requires_auth": True,
                "accepted_auth": ["Bearer"],
                "token_validation_url": "https://auth.internal/validate",
                "status": "healthy",
            },
            "api_gateway": {
                "routes": {
                    "/v1/check": "DEPRECATED — use /v2/check",
                    "/v2/reserve": "inventory_service",
                    "/v2/check": "inventory_service",
                    "/v1/create": "shipping_service",
                },
                "timeout": 30,
            },
            "auth_service": {
                "valid_tokens": ["valid_token_123", "valid_token_789"],
                "expired_tokens": ["expired_token_456"],
                "token_refresh_endpoint": "/refresh",
            },
        },
        logs={
            "order_service": [
                "[ERROR] 2026-03-25T12:00:05Z POST inventory.internal/v1/check -> 301 Moved Permanently",
                "[ERROR] 2026-03-25T12:00:05Z Response: {'error': 'Endpoint deprecated. Use /v2/reserve'}",
                "[ERROR] 2026-03-25T12:00:07Z Timeout after 2s waiting for inventory response",
                "[ERROR] 2026-03-25T12:00:07Z Order ord_999 failed: inventory check timed out",
                "[WARN]  2026-03-25T12:00:08Z Synchronous mode: blocking on inventory response",
                "[ERROR] 2026-03-25T12:00:09Z Race condition: order ord_998 processed before ord_997 completed",
            ],
            "inventory_service": [
                "[INFO]  2026-03-25T12:00:05Z Received request on /v1/check -> redirecting to /v2/check",
                "[WARN]  2026-03-25T12:00:06Z Processing reservation... avg time: 4s",
                "[ERROR] 2026-03-25T12:00:10Z POST shipping.internal/v1/create -> 401 Unauthorized",
                "[ERROR] 2026-03-25T12:00:10Z Auth token expired_token_456 is no longer valid",
                "[ERROR] 2026-03-25T12:00:10Z Cannot create shipment: authentication failed",
            ],
            "shipping_service": [
                "[WARN]  2026-03-25T12:00:10Z Rejected request: token 'expired_token_456' is expired",
                "[INFO]  2026-03-25T12:00:00Z Shipping service healthy, awaiting authenticated requests",
            ],
            "api_gateway": [
                "[WARN]  2026-03-25T12:00:05Z Deprecated endpoint /v1/check accessed by order_service",
                "[INFO]  2026-03-25T12:00:05Z Redirecting /v1/check -> /v2/check (301)",
            ],
            "auth_service": [
                "[WARN]  2026-03-25T12:00:10Z Token validation failed: expired_token_456 expired at 2026-03-24T00:00:00Z",
                "[INFO]  2026-03-25T12:00:00Z Auth service ready. Valid tokens: 2, Expired: 1",
            ],
        },
        issues=[
            Issue(
                issue_id="hard_wrong_url",
                service="order_service",
                description="Order service calling deprecated /v1/check instead of /v2/reserve",
                expected_fix={"inventory_url": "https://inventory.internal/v2/reserve"},
                fix_key="inventory_url",
                log_hint="Endpoint deprecated. Use /v2/reserve",
            ),
            Issue(
                issue_id="hard_timeout",
                service="order_service",
                description="Timeout too short (2s) for inventory service that takes ~4s to process",
                expected_fix={"timeout": 10},
                fix_key="timeout",
                log_hint="Timeout after 2s waiting for inventory response",
            ),
            Issue(
                issue_id="hard_async",
                service="order_service",
                description="Synchronous mode causes race conditions between concurrent orders",
                expected_fix={"async_mode": True},
                fix_key="async_mode",
                log_hint="Race condition: order ord_998 processed before ord_997 completed",
            ),
            Issue(
                issue_id="hard_expired_token",
                service="inventory_service",
                description="Expired auth token used for shipping service requests",
                expected_fix={"headers.Authorization": "Bearer valid_token_789"},
                fix_key="headers.Authorization",
                log_hint="Auth token expired_token_456 is no longer valid",
            ),
            Issue(
                issue_id="hard_token_refresh",
                service="inventory_service",
                description="No automatic token refresh mechanism configured",
                expected_fix={"token_refresh_url": "https://auth.internal/refresh", "auto_refresh": True},
                fix_key="token_refresh_url",
                log_hint="Token validation failed: expired_token_456 expired",
            ),
        ],
    )
