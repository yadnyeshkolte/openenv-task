# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario definitions for the API Integration Debugging Environment.

Each scenario models a realistic multi-service API ecosystem with:
- Service dependency graphs (upstream/downstream relationships)
- Cascading failures (upstream bugs propagate downstream)
- Dynamic logs that update when issues are fixed
- Expanded issue pools for seed-based random subset selection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
    # --- New fields for cascading failures ---
    depends_on: List[str] = field(default_factory=list)
    # Issues that must be fixed before this one can be diagnosed
    cascade_effects: Dict[str, str] = field(default_factory=dict)
    # service -> error message caused by this issue being unfixed
    category: str = "configuration"
    # Issue category: configuration, authentication, networking, protocol
    severity: str = "error"
    # Severity: error, warning, critical
    root_cause_explanation: str = ""
    # Detailed explanation of why this issue occurs (for grading diagnosis quality)


@dataclass
class ServiceNode:
    """A node in the service dependency graph."""
    name: str
    depends_on: List[str] = field(default_factory=list)
    # Services this one calls (upstream dependencies)
    health_status: str = "degraded"
    # healthy, degraded, error, unreachable


@dataclass
class Scenario:
    """A complete API debugging scenario with dependency graph."""
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    services: List[str]
    configs: Dict[str, Dict[str, Any]]
    logs: Dict[str, List[str]]
    issues: List[Issue]
    # --- New fields ---
    service_graph: Dict[str, ServiceNode] = field(default_factory=dict)
    # Service dependency graph
    dynamic_logs: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # service -> {issue_id: [new logs when fixed]}
    optimal_fix_order: List[str] = field(default_factory=list)
    # Optimal order to fix issues (for strategy scoring)
    context: str = ""
    # Additional scenario context for the agent


def get_scenario(task_id: str, seed: Optional[int] = None) -> Scenario:
    """
    Load a scenario by task ID with optional randomization.

    Args:
        task_id: One of 'easy', 'medium', 'hard'
        seed: Optional seed for deterministic but varied scenarios.
              When provided, selects a random subset of issues from the pool
              and randomizes log order. When None, returns the canonical scenario.
    """
    scenario_builders = {
        "easy": _easy_scenario,
        "medium": _medium_scenario,
        "hard": _hard_scenario,
    }
    if task_id not in scenario_builders:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of: {list(scenario_builders.keys())}")

    scenario = scenario_builders[task_id](seed=seed)
    return scenario


def get_all_task_ids() -> List[str]:
    """Return all available task IDs."""
    return ["easy", "medium", "hard"]


def _select_issues(pool: List[Issue], count: int, rng: random.Random) -> List[Issue]:
    """Select a random subset of issues from a pool, respecting dependencies."""
    if count >= len(pool):
        selected = list(pool)
    else:
        # Build dependency-aware selection
        available = list(pool)
        selected = []
        while len(selected) < count and available:
            # Pick a random issue
            issue = rng.choice(available)
            available.remove(issue)
            # Add its dependencies too if not already selected
            deps_satisfied = all(
                any(s.issue_id == dep for s in selected)
                for dep in issue.depends_on
            )
            if deps_satisfied or not issue.depends_on:
                selected.append(issue)
            else:
                # Add dependencies first
                for dep_id in issue.depends_on:
                    dep_issue = next((i for i in pool if i.issue_id == dep_id), None)
                    if dep_issue and dep_issue not in selected:
                        selected.append(dep_issue)
                        if dep_issue in available:
                            available.remove(dep_issue)
                selected.append(issue)

    # Shuffle log order for selected issues
    rng.shuffle(selected)
    return selected[:count]


def _randomize_scenario(scenario: Scenario, seed: int) -> Scenario:
    """Apply seed-based randomization to a scenario."""
    rng = random.Random(seed)

    # Shuffle log entries for each service
    for service_logs in scenario.logs.values():
        rng.shuffle(service_logs)

    # Vary timestamps in log entries
    base_hour = rng.randint(8, 16)
    base_minute = rng.randint(0, 59)
    for service, log_list in scenario.logs.items():
        new_logs = []
        for i, log_line in enumerate(log_list):
            # Replace the timestamp portion  
            minute = (base_minute + i * rng.randint(1, 5)) % 60
            hour = base_hour + (base_minute + i * rng.randint(1, 5)) // 60
            new_log = log_line
            if "2026-" in new_log:
                # Replace date with varied date
                day = rng.randint(20, 28)
                new_log = new_log.replace(
                    "2026-03-25",
                    f"2026-03-{day:02d}"
                ).replace(
                    "2026-03-24",
                    f"2026-03-{day-1:02d}"
                )
            new_logs.append(new_log)
        scenario.logs[service] = new_logs

    return scenario


# ─── Easy Scenario ───────────────────────────────────────────────────────────

def _easy_scenario(seed: Optional[int] = None) -> Scenario:
    """
    Easy: Payment API integration failures.
    Agent must diagnose auth + content-type issues with clear log signals.

    Issue pool has 4 possible issues; canonical scenario uses 2.
    """
    # Full issue pool (4 issues, canonical uses 2)
    issue_pool = [
        Issue(
            issue_id="easy_auth",
            service="payment_client",
            description="Missing Authorization header — payment gateway requires Bearer token authentication",
            expected_fix={"headers.Authorization": "Bearer <token>"},
            fix_key="headers.Authorization",
            log_hint="Missing or invalid Authorization header",
            category="authentication",
            severity="critical",
            root_cause_explanation=(
                "The payment_client is missing the Authorization header entirely. "
                "The payment_gateway requires Bearer token auth on all /process requests. "
                "This results in HTTP 401 on every payment attempt."
            ),
            cascade_effects={
                "payment_gateway": "All requests from payment_client rejected with 401"
            },
        ),
        Issue(
            issue_id="easy_content_type",
            service="payment_client",
            description="Wrong Content-Type header (text/plain instead of application/json)",
            expected_fix={"headers.Content-Type": "application/json"},
            fix_key="headers.Content-Type",
            log_hint="Content-Type must be application/json",
            category="protocol",
            severity="error",
            root_cause_explanation=(
                "The payment_client sends Content-Type: text/plain, but the gateway "
                "only accepts application/json. This causes HTTP 415 Unsupported Media Type. "
                "The gateway cannot parse the request body."
            ),
            cascade_effects={
                "payment_gateway": "Request body parsing fails for payment_client requests"
            },
        ),
        Issue(
            issue_id="easy_timeout",
            service="payment_client",
            description="Timeout set too low (5s) for payment processing that takes 8-12s",
            expected_fix={"timeout": 30},
            fix_key="timeout",
            log_hint="Request timed out after 5s",
            category="networking",
            severity="error",
            root_cause_explanation=(
                "The payment_client has timeout=5s, but payment processing at the gateway "
                "takes 8-12s for fraud checks. Legitimate payments are timing out."
            ),
        ),
        Issue(
            issue_id="easy_base_url",
            service="payment_client",
            description="Base URL pointing to deprecated v1 endpoint instead of v2",
            expected_fix={"base_url": "https://api.paymentgateway.com/v2"},
            fix_key="base_url",
            log_hint="API v1 is deprecated",
            category="configuration",
            severity="warning",
            root_cause_explanation=(
                "The payment_client uses /v1 which is deprecated and returning 301 redirects. "
                "The gateway v2 endpoint has different request schemas, causing deserialization errors."
            ),
        ),
    ]

    # Select issues based on seed
    if seed is not None:
        rng = random.Random(seed)
        issues = _select_issues(issue_pool, 2, rng)
    else:
        issues = issue_pool[:2]  # Canonical: auth + content_type

    # Build logs based on selected issues
    client_logs = [
        "[INFO]  2026-03-25T10:15:20Z Payment client initialized with base_url=https://api.paymentgateway.com/v2",
    ]
    gateway_logs = [
        "[INFO]  2026-03-25T10:15:20Z Gateway ready, accepting application/json with Bearer auth",
    ]

    for issue in issues:
        if issue.issue_id == "easy_auth":
            client_logs.extend([
                "[ERROR] 2026-03-25T10:15:23Z POST /process -> 401 Unauthorized",
                "[ERROR] 2026-03-25T10:15:23Z Response: {'error': 'Missing or invalid Authorization header'}",
                "[WARN]  2026-03-25T10:15:22Z Request headers: Content-Type=text/plain, Accept=application/json",
            ])
            gateway_logs.append(
                "[WARN]  2026-03-25T10:15:23Z Rejected request: no Authorization header present"
            )
        elif issue.issue_id == "easy_content_type":
            client_logs.extend([
                "[ERROR] 2026-03-25T10:15:24Z POST /process -> 415 Unsupported Media Type",
                "[ERROR] 2026-03-25T10:15:24Z Response: {'error': 'Content-Type must be application/json'}",
            ])
            gateway_logs.append(
                "[WARN]  2026-03-25T10:15:24Z Rejected request: unsupported Content-Type 'text/plain'"
            )
        elif issue.issue_id == "easy_timeout":
            client_logs.extend([
                "[ERROR] 2026-03-25T10:15:30Z POST /process -> Request timed out after 5s",
                "[WARN]  2026-03-25T10:15:30Z Payment processing takes 8-12s for fraud verification",
            ])
            gateway_logs.append(
                "[INFO]  2026-03-25T10:15:30Z Processing payment... estimated time: 10s"
            )
        elif issue.issue_id == "easy_base_url":
            client_logs.extend([
                "[ERROR] 2026-03-25T10:15:21Z GET /v1/status -> 301 Moved Permanently",
                "[WARN]  2026-03-25T10:15:21Z API v1 is deprecated, migrate to /v2",
            ])
            gateway_logs.append(
                "[WARN]  2026-03-25T10:15:21Z Deprecated v1 endpoint accessed"
            )

    # Determine initial config based on selected issues
    configs = {
        "payment_client": {
            "base_url": "https://api.paymentgateway.com/v2",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
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
            "processing_time_ms": "8000-12000",
        },
    }

    # Apply broken config for each selected issue
    for issue in issues:
        if issue.issue_id == "easy_auth":
            # Remove auth header (it shouldn't exist)
            configs["payment_client"]["headers"].pop("Authorization", None)
        elif issue.issue_id == "easy_content_type":
            configs["payment_client"]["headers"]["Content-Type"] = "text/plain"
        elif issue.issue_id == "easy_timeout":
            configs["payment_client"]["timeout"] = 5
        elif issue.issue_id == "easy_base_url":
            configs["payment_client"]["base_url"] = "https://api.paymentgateway.com/v1"

    # Dynamic logs: what changes after fixing each issue
    dynamic_logs = {}
    for issue in issues:
        if issue.issue_id == "easy_auth":
            dynamic_logs["easy_auth"] = {
                "payment_client": ["[INFO]  Authorization header set. Retrying request..."],
                "payment_gateway": ["[INFO]  Authentication successful for payment_client"],
            }
        elif issue.issue_id == "easy_content_type":
            dynamic_logs["easy_content_type"] = {
                "payment_client": ["[INFO]  Content-Type set to application/json. Request body parsed."],
                "payment_gateway": ["[INFO]  Request body parsed successfully as JSON"],
            }
        elif issue.issue_id == "easy_timeout":
            dynamic_logs["easy_timeout"] = {
                "payment_client": ["[INFO]  Timeout increased to 30s. Payment processing completing normally."],
            }
        elif issue.issue_id == "easy_base_url":
            dynamic_logs["easy_base_url"] = {
                "payment_client": ["[INFO]  Migrated to v2 API endpoint. Requests routing correctly."],
            }

    # Service dependency graph
    service_graph = {
        "payment_client": ServiceNode(
            name="payment_client",
            depends_on=["payment_gateway"],
            health_status="error",
        ),
        "payment_gateway": ServiceNode(
            name="payment_gateway",
            depends_on=[],
            health_status="healthy",
        ),
    }

    scenario = Scenario(
        task_id="easy",
        difficulty="easy",
        description=(
            "A payment processing API integration is failing. "
            "The client is sending requests to the payment gateway but getting error responses. "
            "Diagnose the root causes by inspecting error logs and service configurations, "
            "then submit the correct configuration fixes."
        ),
        max_steps=15,
        services=["payment_client", "payment_gateway"],
        configs=configs,
        logs={"payment_client": client_logs, "payment_gateway": gateway_logs},
        issues=issues,
        service_graph=service_graph,
        dynamic_logs=dynamic_logs,
        optimal_fix_order=[i.issue_id for i in issues],
        context=(
            "The payment_client sends HTTP requests to payment_gateway. "
            "payment_gateway requires Bearer authentication and JSON content type."
        ),
    )

    if seed is not None:
        scenario = _randomize_scenario(scenario, seed)

    return scenario


# ─── Medium Scenario ─────────────────────────────────────────────────────────

def _medium_scenario(seed: Optional[int] = None) -> Scenario:
    """
    Medium: Webhook chain with cascading failures.
    Service A -> Service B -> Service C, with rate limiting, retry, and auth issues.

    Issue pool has 5 possible issues; canonical scenario uses 3.
    Issues have dependencies — fixing rate_limit reveals the real retry issue.
    """
    issue_pool = [
        Issue(
            issue_id="medium_rate_limit",
            service="webhook_sender",
            description="Rate limit too high (100/s vs receiver's 10/s limit) causing 429 responses",
            expected_fix={"rate_limit.requests_per_second": 10},
            fix_key="rate_limit.requests_per_second",
            log_hint="Rate limit exceeded: 100 req/s > 10 req/s allowed",
            category="networking",
            severity="error",
            root_cause_explanation=(
                "webhook_sender fires at 100 req/s but webhook_receiver only accepts 10 req/s. "
                "The excess requests get 429 Too Many Requests, and with only 1 retry, most events are dropped."
            ),
            cascade_effects={
                "webhook_receiver": "Overwhelmed with requests, dropping 90% of events",
                "notification_service": "No events arriving downstream",
            },
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
            depends_on=["medium_rate_limit"],
            # The retry issue is masked by the rate limit issue — even with retries,
            # 100 req/s would still overwhelm the receiver
            category="configuration",
            severity="error",
            root_cause_explanation=(
                "Even after fixing the rate limit, the sender only retries once with no backoff. "
                "Transient 429s during bursts aren't retried because 429 isn't in retry_on_status. "
                "This causes event loss on any temporary load spike."
            ),
        ),
        Issue(
            issue_id="medium_signature",
            service="webhook_sender",
            description="Webhook signature header is empty — receiver rejects unsigned events",
            expected_fix={"headers.X-Webhook-Signature": "sha256=<computed>"},
            fix_key="headers.X-Webhook-Signature",
            log_hint="Signature validation FAILED: received empty signature",
            category="authentication",
            severity="critical",
            root_cause_explanation=(
                "webhook_sender has signing_secret configured but the X-Webhook-Signature header "
                "is empty string. webhook_receiver validates signatures and drops all unsigned "
                "events as potential spoofing attempts."
            ),
            cascade_effects={
                "webhook_receiver": "Dropping all events as unsigned/spoofed",
                "notification_service": "Zero events forwarded from receiver",
            },
        ),
        Issue(
            issue_id="medium_target_url",
            service="webhook_sender",
            description="Target URL pointing to wrong receiver endpoint (/webhook vs /hooks/incoming)",
            expected_fix={"target_url": "https://receiver.internal/hooks/incoming"},
            fix_key="target_url",
            log_hint="404 Not Found on /webhook endpoint",
            category="configuration",
            severity="error",
            root_cause_explanation=(
                "webhook_sender posts to /webhook but the receiver listens on /hooks/incoming. "
                "All requests get 404 Not Found."
            ),
        ),
        Issue(
            issue_id="medium_content_encoding",
            service="webhook_sender",
            description="Payload compression enabled but receiver doesn't support gzip",
            expected_fix={"compression": "none"},
            fix_key="compression",
            log_hint="Unsupported Content-Encoding: gzip",
            category="protocol",
            severity="warning",
            root_cause_explanation=(
                "webhook_sender compresses payloads with gzip but webhook_receiver "
                "doesn't have a decompression middleware. Requests fail with 415."
            ),
        ),
    ]

    if seed is not None:
        rng = random.Random(seed)
        issues = _select_issues(issue_pool, 3, rng)
    else:
        issues = issue_pool[:3]  # Canonical: rate_limit, retry, signature

    # Build configs
    configs = {
        "webhook_sender": {
            "target_url": "https://receiver.internal/hooks/incoming",
            "headers": {
                "Content-Type": "application/json",
                "X-Webhook-Signature": "sha256=computed_hmac",
            },
            "rate_limit": {
                "requests_per_second": 10,
                "burst_size": 20,
            },
            "retry": {
                "max_retries": 3,
                "backoff_factor": 2,
                "retry_on_status": [429, 500],
            },
            "signing_secret": "whsec_abc123secret",
            "compression": "none",
        },
        "webhook_receiver": {
            "endpoint": "/hooks/incoming",
            "rate_limit": {
                "requests_per_second": 10,
                "burst_size": 20,
            },
            "signature_validation": True,
            "expected_signature_header": "X-Webhook-Signature",
            "signing_secret": "whsec_abc123secret",
            "forward_to": "https://notifications.internal/notify",
            "supported_encodings": ["identity"],
        },
        "notification_service": {
            "endpoint": "/notify",
            "accepts_from": ["webhook_receiver"],
            "status": "healthy",
        },
    }

    # Apply broken config for each selected issue
    for issue in issues:
        if issue.issue_id == "medium_rate_limit":
            configs["webhook_sender"]["rate_limit"]["requests_per_second"] = 100
            configs["webhook_sender"]["rate_limit"]["burst_size"] = 200
        elif issue.issue_id == "medium_retry":
            configs["webhook_sender"]["retry"] = {
                "max_retries": 1,
                "backoff_factor": 0,
                "retry_on_status": [500],
            }
        elif issue.issue_id == "medium_signature":
            configs["webhook_sender"]["headers"]["X-Webhook-Signature"] = ""
        elif issue.issue_id == "medium_target_url":
            configs["webhook_sender"]["target_url"] = "https://receiver.internal/webhook"
        elif issue.issue_id == "medium_content_encoding":
            configs["webhook_sender"]["compression"] = "gzip"

    # Build logs based on selected issues
    sender_logs = [
        "[INFO]  2026-03-25T10:59:59Z Webhook sender started. Signature header: X-Webhook-Signature",
    ]
    receiver_logs = [
        "[INFO]  2026-03-25T10:59:59Z Receiver ready. Rate limit: 10 req/s. Signature validation: ON",
    ]
    notif_logs = [
        "[INFO]  2026-03-25T10:59:59Z Notification service healthy. Waiting for events.",
    ]

    for issue in issues:
        if issue.issue_id == "medium_rate_limit":
            sender_logs.extend([
                "[ERROR] 2026-03-25T11:00:01Z POST /hooks/incoming -> 429 Too Many Requests",
                "[ERROR] 2026-03-25T11:00:01Z Rate limited. Retry-After: 5s",
                "[WARN]  2026-03-25T11:00:00Z Sending at 100 req/s (burst=200)",
            ])
            receiver_logs.append(
                "[WARN]  2026-03-25T11:00:01Z Rate limit exceeded: 100 req/s > 10 req/s allowed"
            )
        elif issue.issue_id == "medium_retry":
            sender_logs.extend([
                "[WARN]  2026-03-25T11:00:02Z Retry attempt 1/1 failed. No more retries.",
                "[ERROR] 2026-03-25T11:00:03Z Event evt_12345 dropped after retry exhaustion",
            ])
        elif issue.issue_id == "medium_signature":
            receiver_logs.extend([
                "[ERROR] 2026-03-25T11:00:02Z Signature validation FAILED: received empty signature",
                "[WARN]  2026-03-25T11:00:02Z Dropping event: invalid signature from webhook_sender",
            ])
        elif issue.issue_id == "medium_target_url":
            sender_logs.extend([
                "[ERROR] 2026-03-25T11:00:01Z POST /webhook -> 404 Not Found on /webhook endpoint",
                "[WARN]  2026-03-25T11:00:01Z Receiver endpoint may have changed",
            ])
        elif issue.issue_id == "medium_content_encoding":
            receiver_logs.extend([
                "[ERROR] 2026-03-25T11:00:02Z Unsupported Content-Encoding: gzip",
                "[WARN]  2026-03-25T11:00:02Z Cannot decompress payload from webhook_sender",
            ])

    notif_logs.append("[WARN]  2026-03-25T11:00:05Z No events received in last 60s")

    # Dynamic logs
    dynamic_logs = {
        "medium_rate_limit": {
            "webhook_sender": ["[INFO]  Rate limit adjusted to 10 req/s. 429 errors resolved."],
            "webhook_receiver": ["[INFO]  Incoming request rate normalized. Processing events."],
        },
        "medium_retry": {
            "webhook_sender": ["[INFO]  Retry config updated: 3 retries with backoff. 429 now retried."],
        },
        "medium_signature": {
            "webhook_sender": ["[INFO]  Webhook signature computed and attached to requests."],
            "webhook_receiver": ["[INFO]  Signature validation passed for incoming events."],
        },
        "medium_target_url": {
            "webhook_sender": ["[INFO]  Target URL corrected to /hooks/incoming. Requests routing OK."],
        },
        "medium_content_encoding": {
            "webhook_sender": ["[INFO]  Compression disabled. Receiver parsing payloads correctly."],
        },
    }

    service_graph = {
        "webhook_sender": ServiceNode(
            name="webhook_sender",
            depends_on=["webhook_receiver"],
            health_status="error",
        ),
        "webhook_receiver": ServiceNode(
            name="webhook_receiver",
            depends_on=["notification_service"],
            health_status="degraded",
        ),
        "notification_service": ServiceNode(
            name="notification_service",
            depends_on=[],
            health_status="healthy",
        ),
    }

    # Determine optimal fix order (respect dependencies)
    issue_ids = [i.issue_id for i in issues]
    optimal_order = []
    # Rate limit should be fixed before retry (dependency)
    if "medium_rate_limit" in issue_ids:
        optimal_order.append("medium_rate_limit")
    if "medium_retry" in issue_ids:
        optimal_order.append("medium_retry")
    for iid in issue_ids:
        if iid not in optimal_order:
            optimal_order.append(iid)

    scenario = Scenario(
        task_id="medium",
        difficulty="medium",
        description=(
            "A webhook-based notification system is dropping events. "
            "webhook_sender sends webhooks to webhook_receiver, which forwards to notification_service. "
            "Events are being lost due to multiple cascading failures in the webhook chain. "
            "Fix the webhook_sender configuration to restore event delivery."
        ),
        max_steps=25,
        services=["webhook_sender", "webhook_receiver", "notification_service"],
        configs=configs,
        logs={
            "webhook_sender": sender_logs,
            "webhook_receiver": receiver_logs,
            "notification_service": notif_logs,
        },
        issues=issues,
        service_graph=service_graph,
        dynamic_logs=dynamic_logs,
        optimal_fix_order=optimal_order,
        context=(
            "Event flow: webhook_sender -> webhook_receiver -> notification_service. "
            "webhook_receiver validates signatures and enforces rate limits. "
            "Fixing upstream issues may reveal additional downstream problems."
        ),
    )

    if seed is not None:
        scenario = _randomize_scenario(scenario, seed)

    return scenario


# ─── Hard Scenario ────────────────────────────────────────────────────────────

def _hard_scenario(seed: Optional[int] = None) -> Scenario:
    """
    Hard: E-commerce order processing pipeline with cascading failures.
    order_service -> inventory_service -> shipping_service
    Plus api_gateway and auth_service.

    Issue pool has 7 possible issues; canonical scenario uses 5.
    Multiple dependency chains make this genuinely challenging.
    """
    issue_pool = [
        Issue(
            issue_id="hard_wrong_url",
            service="order_service",
            description="Order service calling deprecated /v1/check instead of /v2/reserve",
            expected_fix={"inventory_url": "https://inventory.internal/v2/reserve"},
            fix_key="inventory_url",
            log_hint="Endpoint deprecated. Use /v2/reserve",
            category="configuration",
            severity="error",
            root_cause_explanation=(
                "order_service calls /v1/check which was deprecated. The API gateway returns "
                "301 Moved Permanently. The redirect goes to /v2/check (read-only) instead of "
                "/v2/reserve (write). Inventory is never actually reserved."
            ),
            cascade_effects={
                "inventory_service": "Receiving read-only check requests instead of reservation requests",
                "api_gateway": "Generating 301 redirect responses for deprecated endpoints",
            },
        ),
        Issue(
            issue_id="hard_timeout",
            service="order_service",
            description="Timeout too short (2s) for inventory service that takes ~4s to process",
            expected_fix={"timeout": 10},
            fix_key="timeout",
            log_hint="Timeout after 2s waiting for inventory response",
            depends_on=["hard_wrong_url"],
            # Timeout issue is masked by wrong URL — fix URL first to see real timeout
            category="networking",
            severity="error",
            root_cause_explanation=(
                "order_service has timeout=2s but inventory_service takes ~4s for reservation "
                "(including DB lock + stock validation). After fixing the URL, requests now reach "
                "inventory but time out before completion."
            ),
            cascade_effects={
                "inventory_service": "Connections killed mid-processing, leaving orphaned DB locks",
            },
        ),
        Issue(
            issue_id="hard_async",
            service="order_service",
            description="Synchronous mode causes race conditions between concurrent orders",
            expected_fix={"async_mode": True},
            fix_key="async_mode",
            log_hint="Race condition: order ord_998 processed before ord_997 completed",
            category="configuration",
            severity="critical",
            root_cause_explanation=(
                "order_service runs in sync mode, blocking the main thread on each inventory call. "
                "Concurrent orders queue up and when timeouts occur, orders are processed out of "
                "order, causing double-reservation and stock inconsistencies."
            ),
        ),
        Issue(
            issue_id="hard_expired_token",
            service="inventory_service",
            description="Expired auth token used for shipping service requests",
            expected_fix={"headers.Authorization": "Bearer valid_token_789"},
            fix_key="headers.Authorization",
            log_hint="Auth token expired_token_456 is no longer valid",
            category="authentication",
            severity="critical",
            root_cause_explanation=(
                "inventory_service uses Bearer expired_token_456 to authenticate with "
                "shipping_service. This token expired on 2026-03-24. All shipment creation "
                "requests fail with 401, so reserved inventory is never shipped."
            ),
            cascade_effects={
                "shipping_service": "Rejecting all requests from inventory_service",
                "auth_service": "Logging repeated failed token validations",
            },
        ),
        Issue(
            issue_id="hard_token_refresh",
            service="inventory_service",
            description="No automatic token refresh mechanism configured",
            expected_fix={"token_refresh_url": "https://auth.internal/refresh", "auto_refresh": True},
            fix_key="token_refresh_url",
            log_hint="Token validation failed: expired_token_456 expired",
            depends_on=["hard_expired_token"],
            # Token refresh is only relevant after fixing the expired token
            category="configuration",
            severity="error",
            root_cause_explanation=(
                "Even after replacing the expired token, there's no auto-refresh mechanism. "
                "Tokens expire every 24h, so without auto_refresh=True and a refresh URL, "
                "the same issue will recur tomorrow."
            ),
        ),
        Issue(
            issue_id="hard_circuit_breaker",
            service="order_service",
            description="No circuit breaker — failed requests keep hammering inventory_service",
            expected_fix={"circuit_breaker.enabled": True, "circuit_breaker.failure_threshold": 5},
            fix_key="circuit_breaker",
            log_hint="Circuit breaker not configured",
            category="configuration",
            severity="warning",
            root_cause_explanation=(
                "Without a circuit breaker, order_service keeps sending requests to "
                "inventory_service even when it's consistently failing. This wastes resources "
                "and can cause a cascading overload."
            ),
        ),
        Issue(
            issue_id="hard_idempotency",
            service="order_service",
            description="Missing idempotency key — retried requests create duplicate orders",
            expected_fix={"headers.Idempotency-Key": "order-{order_id}"},
            fix_key="headers.Idempotency-Key",
            log_hint="Duplicate order detected: ord_997 submitted twice",
            depends_on=["hard_async"],
            category="protocol",
            severity="error",
            root_cause_explanation=(
                "When async retries fire, there's no Idempotency-Key header to deduplicate "
                "requests. inventory_service creates duplicate reservations for the same order."
            ),
        ),
    ]

    if seed is not None:
        rng = random.Random(seed)
        issues = _select_issues(issue_pool, 5, rng)
    else:
        issues = issue_pool[:5]  # Canonical: first 5

    configs = {
        "order_service": {
            "name": "order_service",
            "inventory_url": "https://inventory.internal/v2/reserve",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer valid_token_123",
            },
            "timeout": 10,
            "async_mode": True,
            "callback_url": "https://orders.internal/callback",
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
            },
        },
        "inventory_service": {
            "name": "inventory_service",
            "endpoint_version": "v2",
            "reserve_path": "/v2/reserve",
            "check_path": "/v2/check",
            "shipping_url": "https://shipping.internal/v1/create",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer valid_token_789",
            },
            "timeout": 10,
            "processing_time_avg": 4,
            "token_refresh_url": "https://auth.internal/refresh",
            "auto_refresh": True,
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
            "token_ttl_hours": 24,
        },
    }

    # Apply broken config for each selected issue
    for issue in issues:
        if issue.issue_id == "hard_wrong_url":
            configs["order_service"]["inventory_url"] = "https://inventory.internal/v1/check"
        elif issue.issue_id == "hard_timeout":
            configs["order_service"]["timeout"] = 2
        elif issue.issue_id == "hard_async":
            configs["order_service"]["async_mode"] = False
        elif issue.issue_id == "hard_expired_token":
            configs["inventory_service"]["headers"]["Authorization"] = "Bearer expired_token_456"
        elif issue.issue_id == "hard_token_refresh":
            configs["inventory_service"].pop("token_refresh_url", None)
            configs["inventory_service"]["auto_refresh"] = False
        elif issue.issue_id == "hard_circuit_breaker":
            configs["order_service"]["circuit_breaker"] = {"enabled": False}
        elif issue.issue_id == "hard_idempotency":
            configs["order_service"]["headers"].pop("Idempotency-Key", None)

    # Build logs
    order_logs = []
    inventory_logs = []
    shipping_logs = []
    gateway_logs = []
    auth_logs = [
        "[INFO]  2026-03-25T12:00:00Z Auth service ready. Valid tokens: 2, Expired: 1",
    ]

    for issue in issues:
        if issue.issue_id == "hard_wrong_url":
            order_logs.extend([
                "[ERROR] 2026-03-25T12:00:05Z POST inventory.internal/v1/check -> 301 Moved Permanently",
                "[ERROR] 2026-03-25T12:00:05Z Response: {'error': 'Endpoint deprecated. Use /v2/reserve'}",
            ])
            inventory_logs.append(
                "[INFO]  2026-03-25T12:00:05Z Received request on /v1/check -> redirecting to /v2/check"
            )
            gateway_logs.extend([
                "[WARN]  2026-03-25T12:00:05Z Deprecated endpoint /v1/check accessed by order_service",
                "[INFO]  2026-03-25T12:00:05Z Redirecting /v1/check -> /v2/check (301)",
            ])
        elif issue.issue_id == "hard_timeout":
            order_logs.extend([
                "[ERROR] 2026-03-25T12:00:07Z Timeout after 2s waiting for inventory response",
                "[ERROR] 2026-03-25T12:00:07Z Order ord_999 failed: inventory check timed out",
            ])
            inventory_logs.append(
                "[WARN]  2026-03-25T12:00:06Z Processing reservation... avg time: 4s"
            )
        elif issue.issue_id == "hard_async":
            order_logs.extend([
                "[WARN]  2026-03-25T12:00:08Z Synchronous mode: blocking on inventory response",
                "[ERROR] 2026-03-25T12:00:09Z Race condition: order ord_998 processed before ord_997 completed",
            ])
        elif issue.issue_id == "hard_expired_token":
            inventory_logs.extend([
                "[ERROR] 2026-03-25T12:00:10Z POST shipping.internal/v1/create -> 401 Unauthorized",
                "[ERROR] 2026-03-25T12:00:10Z Auth token expired_token_456 is no longer valid",
                "[ERROR] 2026-03-25T12:00:10Z Cannot create shipment: authentication failed",
            ])
            shipping_logs.append(
                "[WARN]  2026-03-25T12:00:10Z Rejected request: token 'expired_token_456' is expired"
            )
            auth_logs.append(
                "[WARN]  2026-03-25T12:00:10Z Token validation failed: expired_token_456 expired at 2026-03-24T00:00:00Z"
            )
        elif issue.issue_id == "hard_token_refresh":
            auth_logs.append(
                "[WARN]  2026-03-25T12:00:11Z Token validation failed: expired_token_456 expired. No refresh configured."
            )
        elif issue.issue_id == "hard_circuit_breaker":
            order_logs.extend([
                "[WARN]  2026-03-25T12:00:12Z Circuit breaker not configured, continuing to send requests after 10 failures",
                "[ERROR] 2026-03-25T12:00:12Z System overload: 50 pending requests to inventory_service",
            ])
        elif issue.issue_id == "hard_idempotency":
            order_logs.append(
                "[ERROR] 2026-03-25T12:00:13Z Duplicate order detected: ord_997 submitted twice"
            )
            inventory_logs.append(
                "[WARN]  2026-03-25T12:00:13Z Duplicate reservation request for order ord_997"
            )

    if not shipping_logs:
        shipping_logs.append(
            "[INFO]  2026-03-25T12:00:00Z Shipping service healthy, awaiting authenticated requests"
        )

    dynamic_logs = {
        "hard_wrong_url": {
            "order_service": ["[INFO]  URL corrected to /v2/reserve. Inventory requests routing correctly."],
            "api_gateway": ["[INFO]  order_service now using correct /v2/reserve endpoint."],
        },
        "hard_timeout": {
            "order_service": ["[INFO]  Timeout increased to 10s. Inventory responses completing."],
            "inventory_service": ["[INFO]  Reservations completing successfully within timeout."],
        },
        "hard_async": {
            "order_service": ["[INFO]  Async mode enabled. Orders processing concurrently without blocking."],
        },
        "hard_expired_token": {
            "inventory_service": ["[INFO]  Auth token refreshed. Shipping service requests authenticated."],
            "shipping_service": ["[INFO]  Authentication successful for inventory_service."],
        },
        "hard_token_refresh": {
            "inventory_service": ["[INFO]  Auto token refresh configured. Tokens will be refreshed before expiry."],
        },
        "hard_circuit_breaker": {
            "order_service": ["[INFO]  Circuit breaker enabled. Will stop sending after 5 consecutive failures."],
        },
        "hard_idempotency": {
            "order_service": ["[INFO]  Idempotency keys set. Duplicate requests will be safely deduplicated."],
        },
    }

    service_graph = {
        "order_service": ServiceNode(
            name="order_service",
            depends_on=["inventory_service", "api_gateway"],
            health_status="error",
        ),
        "inventory_service": ServiceNode(
            name="inventory_service",
            depends_on=["shipping_service", "auth_service"],
            health_status="degraded",
        ),
        "shipping_service": ServiceNode(
            name="shipping_service",
            depends_on=[],
            health_status="healthy",
        ),
        "api_gateway": ServiceNode(
            name="api_gateway",
            depends_on=[],
            health_status="healthy",
        ),
        "auth_service": ServiceNode(
            name="auth_service",
            depends_on=[],
            health_status="healthy",
        ),
    }

    # Build optimal fix order respecting dependencies
    issue_ids = [i.issue_id for i in issues]
    optimal_order = []
    ordered_preference = [
        "hard_wrong_url", "hard_timeout", "hard_async",
        "hard_expired_token", "hard_token_refresh",
        "hard_circuit_breaker", "hard_idempotency",
    ]
    for iid in ordered_preference:
        if iid in issue_ids:
            optimal_order.append(iid)
    for iid in issue_ids:
        if iid not in optimal_order:
            optimal_order.append(iid)

    scenario = Scenario(
        task_id="hard",
        difficulty="hard",
        description=(
            "An e-commerce order processing pipeline is failing with cascading errors. "
            "Order Service calls Inventory Service, which calls Shipping Service. "
            "Multiple issues span the pipeline: wrong endpoints, timeouts, race conditions, "
            "expired authentication tokens, and missing resilience patterns. "
            "Some issues are masked by upstream failures — you must fix issues in the right "
            "order to diagnose the full chain."
        ),
        max_steps=40,
        services=["order_service", "inventory_service", "shipping_service", "api_gateway", "auth_service"],
        configs=configs,
        logs={
            "order_service": order_logs,
            "inventory_service": inventory_logs,
            "shipping_service": shipping_logs,
            "api_gateway": gateway_logs,
            "auth_service": auth_logs,
        },
        issues=issues,
        service_graph=service_graph,
        dynamic_logs=dynamic_logs,
        optimal_fix_order=optimal_order,
        context=(
            "Request flow: order_service -> api_gateway -> inventory_service -> shipping_service. "
            "auth_service provides token validation for all inter-service calls. "
            "Some issues are masked by upstream failures — fixing upstream issues may reveal "
            "new errors downstream. Pay attention to service dependencies."
        ),
    )

    if seed is not None:
        scenario = _randomize_scenario(scenario, seed)

    return scenario
