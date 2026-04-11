---
title: API Debug Env
emoji: 🔧
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
---

# 🔧 API Integration Debugging Environment

> A real-world OpenEnv environment where an AI agent diagnoses and fixes broken API integrations across multi-service systems with **cascading failures**, **dynamic state**, and **multi-dimensional rubric grading**.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-70%20passed-brightgreen)]()
[![HF Space](https://img.shields.io/badge/HF%20Space-Live-orange)](https://huggingface.co/spaces/yadnyeshkolte/api-debug-env)

---

## Table of Contents

- [Motivation — Why API Debugging?](#motivation--why-api-debugging)
- [Environment Overview](#environment-overview)
- [Key Design Features](#key-design-features)
- [Tasks (Easy / Medium / Hard)](#tasks)
- [Multi-Dimensional Grading Rubric](#multi-dimensional-grading-rubric)
- [Reward Shaping](#reward-shaping)
- [Action & Observation Spaces](#action--observation-spaces)
- [Example Transcript](#example-transcript)
- [Setup & Usage](#setup--usage)
- [API Endpoints](#api-endpoints)
- [Running Inference](#running-inference)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Design Philosophy](#design-philosophy)

---

## Motivation — Why API Debugging?

API integration failures are one of the **most common and expensive issues** in production software engineering. When microservices communicate — Service A calls Service B which calls Service C — a single misconfiguration can cascade through the entire system, producing confusing error chains that take hours to diagnose.

Real-world API debugging requires:

- **Structured diagnosis** — reading error logs and configs across multiple services
- **Dependency awareness** — understanding which upstream failure is causing downstream errors
- **Strategic reasoning** — fixing root causes first to unmask hidden downstream bugs
- **Precision** — submitting exact configuration corrections, not approximate guesses

This environment simulates **real-world cascading API failures** with dynamic state that changes as the agent acts — not a static lookup puzzle.

---

## Environment Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                     Agent Debugging Loop                          │
│                                                                   │
│  1. reset(task_id)      → Initial observation with broken state   │
│  2. step(inspect_logs)  → Error logs with diagnostic clues        │
│  3. step(inspect_config)→ Current (broken) service configuration  │
│  4. step(inspect_endpoint) → Simulated API response (401, 504..)  │
│  5. step(submit_fix)    → Strict fix validation + cascade update  │
│  6. grade()             → Multi-dimensional rubric score [0,1]    │
│                                                                   │
│  State updates dynamically: service health changes, new logs      │
│  appear, error cascades resolve as the agent fixes issues.        │
└───────────────────────────────────────────────────────────────────┘
```

The agent interacts through the standard OpenEnv API:
- **`reset()`** → returns initial observation with broken service state
- **`step(action)`** → executes one debugging action, returns observation + reward
- **`state()`** → returns current environment state (episode_id, step_count)
- **`grade()`** → returns final score using multi-dimensional rubric

---

## Key Design Features

### 1. Cascading Failures with Service Dependency Graphs

Each task models a real multi-service ecosystem. Services depend on each other, and a bug in an upstream service **cascades** to all downstream services:

```
Hard Task Dependency Graph:

  order_service ──┬──→ inventory_service ──┬──→ shipping_service
                  │                        └──→ auth_service
                  └──→ api_gateway

  [ERROR]              [DEGRADED]               [HEALTHY]
```

- Fixing `order_service`'s wrong URL unmasks `inventory_service`'s timeout issue
- Fixing `inventory_service`'s expired token allows `shipping_service` to respond
- **Some issues are intentionally masked by upstream failures** — the agent must fix in the right order

### 2. Dynamic State

Unlike static environments, the state **changes as the agent acts**:

| What changes | How |
|---|---|
| **Service health** | Fixing issues updates service status: `error` → `degraded` → `healthy` |
| **Logs** | After a fix, re-inspecting logs shows **new entries** (e.g., "Authorization header set. Retrying...") |
| **Error traces** | The cascade chain shrinks as upstream issues are resolved |
| **Endpoint responses** | `inspect_endpoint` returns different HTTP errors based on current fix state |

### 3. Seed-Based Scenario Randomization

Each difficulty level has an **expanded issue pool** (more issues than are selected per episode):

| Difficulty | Pool Size | Selected Per Episode |
|---|---|---|
| Easy | 4 issues | 2 |
| Medium | 5 issues | 3 |
| Hard | 7 issues | 5 |

Passing a `seed` to `reset()` produces a **deterministic but varied** scenario — different seeds select different subsets from the pool and randomize log order. This prevents agents from memorizing fixed patterns.

### 4. Strict Fix Validation with Partial Credit

The grader validates both **keys and values** of submitted fixes:

- **Exact match** → Full credit (+0.25 reward)
- **Right key, close value** (e.g., timeout=7 when expected=10) → Partial credit (+0.03)
- **Right key, wrong value** (e.g., timeout=100 when expected=10) → Rejected
- **Wrong key entirely** → Penalized (-0.1)
- **Bearer token pattern matching** — `Bearer <any_valid_token>` is accepted
- **Numeric tolerance** — strict 10% tolerance
- **Boolean coercion** — `"true"`, `"1"`, `"yes"` all match `True`

---

## Tasks

### Easy: Payment API Integration (2 issues, 15 max steps)

**Scenario**: A payment processing client is failing to connect to the payment gateway. The agent must diagnose authentication and protocol errors.

- **Services**: `payment_client`, `payment_gateway`
- **Issue pool** (4 possible, 2 selected):
  - Missing `Authorization` header (HTTP 401)
  - Wrong `Content-Type` header — `text/plain` instead of `application/json` (HTTP 415)
  - Timeout too low for payment processing (HTTP 504)
  - Base URL pointing to deprecated v1 endpoint (HTTP 301)
- **Dependencies**: None — straightforward diagnosis

### Medium: Webhook Event Chain (3 issues, 25 max steps)

**Scenario**: A webhook notification system is dropping events across a 3-service chain. Events flow from sender → receiver → notification service, but multiple configuration issues are causing failures.

- **Services**: `webhook_sender`, `webhook_receiver`, `notification_service`
- **Issue pool** (5 possible, 3 selected):
  - Rate limit mismatch (sender at 100/s, receiver accepts 10/s) → 429 errors
  - Insufficient retry config (only 1 retry, no backoff, 429 not in retry list)
  - Empty webhook signature header → receiver drops all events as unsigned
  - Wrong target URL (`/webhook` vs `/hooks/incoming`) → 404 errors
  - Payload compression enabled but receiver doesn't support gzip → 415 errors
- **Dependencies**: Retry issue is **masked** by rate limit — must fix rate limit first to see the retry problem

### Hard: E-Commerce Order Pipeline (5 issues, 40 max steps)

**Scenario**: A complex e-commerce order processing pipeline is failing with cascading errors across 5 services. Multiple dependency chains make this genuinely challenging for frontier models.

- **Services**: `order_service`, `inventory_service`, `shipping_service`, `api_gateway`, `auth_service`
- **Issue pool** (7 possible, 5 selected):
  - Deprecated URL (`/v1/check` → should be `/v2/reserve`) → 301 redirect
  - Timeout too short (2s vs 4s processing time) — masked by wrong URL
  - Synchronous mode causing race conditions between concurrent orders
  - Expired auth token on inventory→shipping calls → 401
  - No auto token refresh configured — masked by expired token
  - No circuit breaker → failed requests hammer inventory service
  - Missing idempotency key → retries create duplicate orders
- **Dependencies**: `timeout` depends on `wrong_url` fix; `token_refresh` depends on `expired_token` fix; `idempotency` depends on `async` fix

---

## Multi-Dimensional Grading Rubric

The grader uses a **4-dimension weighted rubric**, not a simple `issues_fixed / total` ratio:

| Dimension | Weight | What It Measures |
|---|---|---|
| **Fix Score** | 40% | `issues_fixed / total_issues` — how many bugs were actually resolved |
| **Strategy Score** | 25% | Did the agent follow a logical approach? Inspect before fix, avoid repeats, follow dependency order, use all action types |
| **Diagnosis Score** | 20% | Did the agent inspect the service (logs/config) **before** submitting a fix for it? |
| **Efficiency Score** | 15% | `remaining_steps / max_steps` — faster solutions score higher |

```
Final Score = fix × 0.40 + strategy × 0.25 + diagnosis × 0.20 + efficiency × 0.15
Clamped to (0.001, 0.999) — never exactly 0.0 or 1.0
```

**Strategy scoring details:**
- Did the agent inspect logs/config before submitting any fix? (+1)
- Ratio of unique inspections to total inspections (no wasteful repeats) (+1)
- Did fixes follow the optimal dependency order? (+1)
- Did the agent use a variety of action types? (+1)

### Baseline Scores (Rule-Based Heuristic Agent)

| Task | Score | Steps Used | Issues Fixed |
|---|---|---|---|
| Easy | ~0.75 | 7 | 2/2 |
| Medium | ~0.55 | 10 | 3/3 |
| Hard | ~0.45 | 15 | 5/5 |

*The baseline uses a deterministic heuristic (inspect all logs → inspect all configs → submit known fixes). An LLM-based agent following good debugging strategy can score higher.*

---

## Reward Shaping

Every action produces a meaningful reward signal — not just sparse end-of-episode feedback:

| Action | Reward | Condition |
|---|---|---|
| `inspect_logs` (first time, finds error patterns) | **+0.15** | New issue-related log patterns found |
| `inspect_logs` (first time, no issues here) | +0.05 | Valid inspection, no errors in this service |
| `inspect_logs` (repeat, no new info) | 0.00 | Already inspected, nothing changed |
| `inspect_logs` (repeat, after a fix) | +0.05 | Dynamic logs appeared after a recent fix |
| `inspect_config` (service has issues) | +0.05 | Relevant config retrieved |
| `inspect_config` (service is clean) | +0.01 | Config retrieved but no issues here |
| `inspect_config` (repeat) | 0.00 | Already inspected |
| `inspect_endpoint` | +0.02 to +0.05 | Simulated endpoint test |
| `submit_fix` (correct fix) | **+0.25** | Issue resolved, service health updated |
| `submit_fix` (correct + inspected first) | **+0.30** | Fix + strategy bonus for diagnosis |
| `submit_fix` (partial — close but not exact) | +0.03 | Right key, approximately right value |
| `submit_fix` (wrong fix) | **-0.10** | Incorrect fix payload |
| `submit_fix` (empty payload) | -0.10 | Empty fix_payload submitted |
| All issues fixed | **+0.20** | Episode completion bonus |
| Invalid target / invalid action | -0.05 | Bad input |
| Every step | **-0.01** | Step cost — encourages efficiency |

---

## Action & Observation Spaces

### Action Schema (Pydantic model: `ApiDebugAction`)

```json
{
  "action_type": "inspect_logs | inspect_config | inspect_endpoint | submit_fix",
  "target": "<service_name>",
  "fix_payload": {
    "config_key": "corrected_value"
  }
}
```

- `action_type` (required): One of the 4 debugging actions
- `target` (required): The service to act on (from `available_targets` in the observation)
- `fix_payload` (optional): Required only for `submit_fix` — the configuration correction

**Fix payload formats:**
```json
// Simple key-value fix
{"timeout": 10}

// Nested key fix (dot notation)
{"headers.Authorization": "Bearer my_api_key"}

// Complex nested object fix
{"retry": {"max_retries": 3, "backoff_factor": 2, "retry_on_status": [429, 500]}}
```

### Observation Schema (Pydantic model: `ApiDebugObservation`)

```json
{
  "task_id": "easy",
  "task_description": "A payment processing API integration is failing...",
  "logs": ["[ERROR] 2026-03-25T10:15:23Z POST /process -> 401 Unauthorized", "..."],
  "config_snapshot": {"headers": {"Content-Type": "text/plain"}, "timeout": 30},
  "api_response": {"status": "error", "status_code": 401, "error": "Missing Authorization"},
  "service_status": {"payment_client": "error", "payment_gateway": "healthy"},
  "dependency_graph": {"payment_client": ["payment_gateway"], "payment_gateway": []},
  "error_trace": [
    "[CRITICAL] payment_client: Missing Authorization header",
    "  └─> payment_gateway: All requests rejected with 401"
  ],
  "hints": ["Check headers.Authorization"],
  "remaining_steps": 14,
  "issues_found": 1,
  "issues_fixed": 0,
  "issues_total": 2,
  "action_result": "Inspected logs for 'payment_client'. Found relevant error patterns!",
  "available_targets": ["payment_client", "payment_gateway"],
  "done": false,
  "reward": 0.15
}
```

**Key observation fields for agent reasoning:**
- `service_status` — shows which services are healthy/degraded/error (updates dynamically)
- `dependency_graph` — shows service relationships (agent should fix upstream first)
- `error_trace` — shows active error cascades (shrinks as issues are fixed)
- `hints` — progressive hints that get more specific as steps are used

---

## Example Transcript

```
>>> reset(task_id="easy")
task_description: "A payment processing API integration is failing..."
service_status: {payment_client: "error", payment_gateway: "healthy"}
error_trace:
  [CRITICAL] payment_client: Missing Authorization header
    └─> payment_gateway: All requests rejected with 401
  [ERROR] payment_client: Wrong Content-Type (text/plain instead of application/json)
    └─> payment_gateway: Request body parsing fails
issues_total: 2, remaining_steps: 15

>>> step(action_type="inspect_logs", target="payment_client")
logs: [
  "[INFO]  Payment client initialized...",
  "[ERROR] POST /process -> 401 Unauthorized",
  "[ERROR] Response: {'error': 'Missing or invalid Authorization header'}",
  "[WARN]  Request headers: Content-Type=text/plain",
  "[ERROR] POST /process -> 415 Unsupported Media Type",
]
issues_found: 2, reward: +0.15

>>> step(action_type="inspect_config", target="payment_client")
config_snapshot: {
  "base_url": "https://api.paymentgateway.com/v2",
  "headers": {"Content-Type": "text/plain", "Accept": "application/json"},
  "timeout": 30
}
reward: +0.05   // Service has issues, first inspection

>>> step(action_type="submit_fix", target="payment_client",
         fix_payload={"headers.Authorization": "Bearer sk_live_my_key"})
action_result: "Fix accepted! Fixed 1 issue(s). Total: 1/2"
service_status: {payment_client: "degraded", payment_gateway: "healthy"}
reward: +0.30   // Fix (+0.25) + strategy bonus (+0.05) for inspecting first

>>> step(action_type="inspect_logs", target="payment_client")
logs: [...original logs...,
  "[INFO]  Authorization header set. Retrying request..."   // NEW dynamic log!
]
reward: +0.05   // Re-inspection has new dynamic logs

>>> step(action_type="submit_fix", target="payment_client",
         fix_payload={"headers.Content-Type": "application/json"})
action_result: "Fix accepted! All issues fixed! Episode complete."
service_status: {payment_client: "healthy", payment_gateway: "healthy"}
error_trace: ["All issues resolved. No error cascades active."]
reward: +0.50   // Fix (+0.25) + strategy (+0.05) + completion bonus (+0.20)
done: true

>>> grade()
score: 0.82
  fix_score: 1.00 (2/2 fixed)
  diagnosis_score: 1.00 (inspected before every fix)
  efficiency_score: 0.67 (5/15 steps used)
  strategy_score: 0.80 (inspected first, used multiple action types)
```

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker (for containerized deployment)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yadnyeshkolte/openenv-task.git
cd openenv-task

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Run the Server Locally

```bash
# From the project root (openenv-task/)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Quick Test

```bash
# Reset environment
curl -X POST http://localhost:8000/reset

# Inspect logs
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_logs", "target": "payment_client"}'

# Submit a fix
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_fix", "target": "payment_client", "fix_payload": {"headers.Authorization": "Bearer my_key"}}'
```

### Docker Build & Run

```bash
# From the project root (openenv-task/)
docker build -t api_debug_env -f Dockerfile .
docker run -p 8000:8000 api_debug_env
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info, version, and feature list |
| `/reset` | POST | Reset environment (accepts `task_id` and `seed` params) |
| `/step` | POST | Execute a debugging action |
| `/state` | GET | Get current state (episode_id, step_count) |
| `/schema` | GET | Get action/observation Pydantic schemas |
| `/tasks` | GET | List all 3 tasks with action schema and service dependencies |
| `/grader` | POST | Get multi-dimensional grader score for current episode |
| `/baseline` | POST | Run the rule-based baseline agent on all 3 tasks |
| `/health` | GET | Health check endpoint |
| `/docs` | GET | Interactive Swagger UI documentation |

---

## Running Inference

The `inference.py` script at the project root uses the OpenAI API client to run an LLM agent against all 3 tasks:

```bash
# Set your API credentials
export HF_TOKEN=your_huggingface_token
# Optional: override model and API base
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

# Run inference from the project root
python inference.py
```

**Output format** (stdout):
```
[START] task=easy env=api_debug_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=inspect_logs(target=payment_client) reward=0.15 done=false error=null
[STEP]  step=2 action=submit_fix(target=payment_client, fix={...}) reward=0.30 done=false error=null
...
[END]   success=true steps=5 score=0.820 rewards=0.15,0.30,...
```

The inference script:
- Uses `openai.OpenAI` client for all LLM calls
- Reads `HF_TOKEN` (or `API_KEY`) from environment variables
- Includes retry logic with exponential backoff
- Emits `[START]`, `[STEP]`, `[END]` lines to stdout

---

## Running Tests

```bash
# From the project root (openenv-task/)
python -m pytest tests/ -v --tb=short
```

**70 tests** across 12 test classes covering:
- Scenario loading, seed randomization, and issue pool selection
- Environment reset and initialization
- All 4 action types: `inspect_logs`, `inspect_config`, `inspect_endpoint`, `submit_fix`
- Dynamic state: service health updates, dynamic log injection, error trace changes
- Multi-dimensional grading rubric (fix, diagnosis, efficiency, strategy)
- Strict fix validation with partial credit
- Value matching (strings, numbers, booleans, lists, Bearer tokens)
- Full episode integration tests (easy, medium, hard)
- Cascading failure mechanics and dependency chains
- Episode termination conditions

### Validate OpenEnv Compliance

```bash
openenv validate
```

---

## Project Structure

```
openenv-task/                         # Project root
├── __init__.py                       # Package init (exports ApiDebugEnv, Action, Observation)
├── client.py                         # OpenEnv client (WebSocket connection to server)
├── models.py                         # Pydantic Action & Observation type definitions
├── scenarios.py                      # Task scenarios with dependency graphs & issue pools
├── inference.py                      # MANDATORY inference script (LLM agent, OpenAI client)
├── openenv.yaml                      # OpenEnv metadata (spec v1)
├── pyproject.toml                    # Python project config & dependencies
├── Dockerfile                        # Docker build for HF Spaces deployment
├── LICENSE                           # BSD license
├── README.md                         # This file
├── PROGRESS.md                       # Development session log
├── AGENTS.md                         # Instructions for AI coding agents
├── server/
│   ├── __init__.py                   # Server package init
│   ├── api_debug_env_environment.py  # Core environment (reset/step/grade logic)
│   ├── app.py                        # FastAPI endpoints (/reset, /step, /tasks, etc.)
│   ├── Dockerfile                    # Alternate Dockerfile (same as root)
│   └── requirements.txt             # Server-specific requirements
├── scripts/
│   └── baseline_inference.py         # Alternate baseline script
└── tests/
    └── test_environment.py           # 70 unit & integration tests
```

### Key Files

| File | Purpose |
|---|---|
| `server/api_debug_env_environment.py` | **Core logic** — `reset()`, `step()`, `grade()`, dynamic state, cascading failures |
| `scenarios.py` | **Task definitions** — issue pools, dependency graphs, dynamic logs, service configs |
| `models.py` | **Type definitions** — `ApiDebugAction` and `ApiDebugObservation` Pydantic models |
| `inference.py` | **Mandatory** — LLM-based agent using OpenAI client with `[START]/[STEP]/[END]` output |
| `openenv.yaml` | **Mandatory** — OpenEnv spec v1 metadata with task definitions |
| `server/app.py` | **FastAPI server** — all HTTP endpoints including `/baseline` and `/grader` |

---

## Design Philosophy

This environment is designed to be useful for **RL/agent training and evaluation**, not just a one-off benchmark:

1. **Dense Reward Signal** — every action type produces positive or negative reward, enabling gradient-based training (GRPO, DPO, PPO). Not just a sparse binary score at the end.

2. **Progressive Difficulty** — Easy (2 services, 2 issues) → Medium (3 services, 3 issues with 1 dependency) → Hard (5 services, 5 issues with multiple dependency chains). Difficulty comes from complexity, not ambiguity.

3. **Partial Credit** — close-but-wrong fixes get constructive feedback instead of just rejection. This provides learning signal for agents that are on the right track.

4. **Strategy Incentives** — the multi-dimensional rubric rewards **how** the agent solves (inspect before fix, follow dependencies, avoid waste), not just **what** it solves. This encourages emergent debugging strategies.

5. **Stochastic Scenarios** — seed-based randomization from expanded issue pools prevents policy overfitting to memorized scenarios while maintaining reproducibility.

6. **Cascading Dynamics** — upstream fixes change downstream state, requiring **multi-step causal reasoning**. The agent can't just pattern-match each issue independently — it must understand the system architecture.

7. **Real-World Relevance** — API integration debugging is a genuine, high-value task that software engineers spend significant time on. The scenarios model actual failure patterns (expired tokens, rate limiting, missing headers, deprecated endpoints, race conditions).

---

## OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| OpenEnv spec v1 (`openenv.yaml`) | ✅ |
| Typed Pydantic models (Action, Observation) | ✅ |
| `reset()` / `step()` / `state()` API | ✅ |
| 3+ tasks with difficulty range | ✅ (easy, medium, hard) |
| Programmatic graders (0.0–1.0) | ✅ (multi-dimensional rubric) |
| Meaningful reward function | ✅ (dense, not sparse) |
| Baseline inference script | ✅ (`inference.py` at root) |
| OpenAI client for LLM calls | ✅ |
| `[START]/[STEP]/[END]` stdout format | ✅ |
| Dockerfile builds and runs | ✅ |
| HF Space deploys and responds | ✅ |
| `openenv validate` passes | ✅ |

---

## Hackathon Submission

- **HF Space**: [yadnyeshkolte/api-debug-env](https://huggingface.co/spaces/yadnyeshkolte/api-debug-env)
- **GitHub**: [yadnyeshkolte/openenv-task](https://github.com/yadnyeshkolte/openenv-task)
- **Hackathon**: Meta PyTorch OpenEnv Hackathon × Scaler School of Technology
