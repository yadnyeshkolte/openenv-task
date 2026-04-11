---
title: Api Debug Env
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---
# API Integration Debugging Environment

An OpenEnv environment where AI agents diagnose and fix broken API integrations — a real-world task that developers face daily.

## Overview

Agents interact with a simulated multi-service API ecosystem that has various misconfigurations. Through a `step()/reset()/state()` API, the agent must:

1. **Inspect error logs** to identify failure patterns
2. **Inspect service configurations** to find misconfigurations
3. **Test endpoints** to observe current behavior
4. **Submit fixes** with corrected configuration payloads

The environment features:
- **3 difficulty levels** with increasing complexity (2, 3, and 5 issues)
- **Strict value validation** on fixes (grader checks both key AND value)
- **Seed-based randomization** for reproducible yet varied episodes
- **Penalty for repeated inspections** to encourage efficient exploration
- **Comprehensive test suite** with 30+ unit tests

## Action Space

```python
class ApiDebugAction(Action):
    action_type: str   # "inspect_logs" | "inspect_config" | "inspect_endpoint" | "submit_fix"
    target: str        # Service name (e.g. "payment_client", "webhook_sender")
    fix_payload: dict  # Required when action_type="submit_fix"
```

| Action | Description | Reward |
|--------|-------------|--------|
| `inspect_logs` | Read error logs for a service | +0.15 (finds new issue) / +0.05 (first time, no issue) / 0.0 (repeat) |
| `inspect_config` | View current config of a service | +0.05 (has issues) / +0.01 (no issues) / 0.0 (repeat) |
| `inspect_endpoint` | Test-call an endpoint | +0.02 to +0.05 |
| `submit_fix` | Submit a configuration fix | +0.25 (correct) / -0.1 (wrong) |
| *step cost* | Applied every step | -0.01 |

## Observation Space

```python
class ApiDebugObservation(Observation):
    task_id: str              # "easy", "medium", or "hard"
    task_description: str     # Human-readable task description
    logs: List[str]           # Error log lines from inspected service
    config_snapshot: dict     # Configuration of inspected service
    api_response: dict        # Response from endpoint test
    hints: List[str]          # Progressive hints based on step count
    remaining_steps: int      # Steps before episode timeout
    issues_found: int         # Issues identified so far
    issues_fixed: int         # Issues correctly fixed so far
    issues_total: int         # Total issues in scenario
    action_result: str        # Feedback on last action
    available_targets: List   # Valid service names
```

## Tasks

### Task 1: Easy — Payment API Auth Fix
- **Issues**: 2 (missing `Authorization` header, wrong `Content-Type`)
- **Max Steps**: 15
- **Services**: `payment_client`, `payment_gateway`
- **Scenario**: Payment gateway rejects requests with 401/415 errors

### Task 2: Medium — Webhook Chain Debugging
- **Issues**: 3 (rate limit too high, insufficient retries, empty webhook signature)
- **Max Steps**: 25
- **Services**: `webhook_sender`, `webhook_receiver`, `notification_service`
- **Scenario**: Events are dropped across a webhook notification pipeline

### Task 3: Hard — Microservice Cascade Failure
- **Issues**: 5 (wrong endpoint URL, timeout too short, sync mode race condition, expired auth token, missing token refresh)
- **Max Steps**: 40
- **Services**: `order_service`, `inventory_service`, `shipping_service`, `api_gateway`, `auth_service`
- **Scenario**: E-commerce order processing pipeline fails with cascading 500s

## Reward Function

- **Step cost**: -0.01 per step to encourage efficiency
- **Partial progress**: First useful inspection earns reward (+0.05 to +0.15)
- **Repeated inspection**: 0 reward (prevents reward farming)
- **Fix rewards**: +0.25 per correctly fixed issue (strict key+value validation)
- **Completion bonus**: +0.2 when all issues are resolved
- **Penalties**: -0.1 for wrong fixes, -0.05 for invalid actions

## Grading

```
Score = (issues_fixed / issues_total) × efficiency_bonus + exploration_bonus
efficiency_bonus = 1.0 + (remaining_steps / max_steps × 0.3)
exploration_bonus = issues_found / issues_total × 0.1
```

Faster fixes earn up to 30% bonus. Scores strictly clamped to (0.001, 0.999).

## Baseline Scores (Rule-Based Agent)

| Task | Score | Issues Fixed | Issues Total | Steps |
|------|-------|-------------|-------------|-------|
| Easy | ~0.85 | 2/2 | 2 | 6 |
| Medium | ~0.65 | 3/3 | 3 | 9 |
| Hard | ~0.55 | 5/5 | 5 | 15 |

> The rule-based baseline inspects logs/configs then submits known fixes. An LLM agent with proper reasoning can achieve higher scores by solving issues more efficiently.

## Example Interaction (Easy Task)

```text
[START] task=easy env=api_debug_env model=Qwen/Qwen2.5-72B-Instruct

# Agent inspects logs and finds Auth error
[STEP] step=1 action=inspect_logs(target=payment_client) reward=0.14 done=false error=null

# Agent checks config to understand current headers
[STEP] step=2 action=inspect_config(target=payment_client) reward=0.04 done=false error=null

# Agent fixes the authorization header
[STEP] step=3 action=submit_fix(target=payment_client,fix={"headers.Authorization":"Bearer sk_live_token123"}) reward=0.24 done=false error=null

# Agent fixes the content type
[STEP] step=4 action=submit_fix(target=payment_client,fix={"headers.Content-Type":"application/json"}) reward=0.44 done=true error=null

[END] success=true steps=4 score=0.899 rewards=0.14,0.04,0.24,0.44
```

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
cd api_debug_env

# Install dependencies
uv sync

# Run server
uv run server
# or
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
cd api_debug_env
docker build -t api_debug_env:latest -f server/Dockerfile .
docker run -p 8000:8000 api_debug_env:latest
```

### Run Inference

```bash
# Set API credentials
export HF_TOKEN=your-key

# Run inference on all tasks
python inference.py
```

### Run Tests

```bash
cd api_debug_env
pytest tests/ -v --tb=short
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root — environment info and links |
| `/reset` | POST | Reset environment, start new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks with action schemas |
| `/grader` | POST | Get grader score for completed episode |
| `/baseline` | POST | Run baseline inference on all tasks |
| `/schema` | GET | Get action/observation JSON schemas |
| `/health` | GET | Health check endpoint |

## Project Structure

```
api_debug_env/
├── inference.py        # ★ MANDATORY hackathon inference script
├── models.py           # Pydantic Action & Observation models
├── scenarios.py        # 3 task scenarios with randomization support
├── client.py           # WebSocket client for the environment
├── openenv.yaml        # OpenEnv metadata (spec v1)
├── pyproject.toml      # Dependencies & build config
├── server/
│   ├── app.py                        # FastAPI application
│   ├── api_debug_env_environment.py  # Core environment logic
│   └── Dockerfile                    # Container build
├── tests/
│   └── test_environment.py           # 30+ unit & integration tests
└── scripts/
    └── baseline_inference.py         # Original baseline agent script
```

## Randomization & Reproducibility

The environment supports seed-based randomization via `reset(seed=42)`. This:
- Shuffles log entry order so agents can't memorize positions
- Ensures reproducible episodes for consistent evaluation
- When `seed=None` (default), returns the canonical scenario for testing

## License

BSD-style license. See LICENSE file.
