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

## Action Space

```python
class ApiDebugAction(Action):
    action_type: str   # "inspect_logs" | "inspect_config" | "inspect_endpoint" | "submit_fix"
    target: str        # Service name (e.g. "payment_client", "webhook_sender")
    fix_payload: dict  # Required when action_type="submit_fix"
```

| Action | Description | Reward |
|--------|-------------|--------|
| `inspect_logs` | Read error logs for a service | +0.05 (relevant) / +0.15 (finds new issue) |
| `inspect_config` | View current config of a service | +0.02 to +0.05 |
| `inspect_endpoint` | Test-call an endpoint | +0.02 to +0.05 |
| `submit_fix` | Submit a configuration fix | +0.25 (correct) / -0.1 (wrong) |

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

- **Partial progress**: Every useful inspection earns reward (+0.05 to +0.15)
- **Fix rewards**: +0.25 per correctly fixed issue
- **Completion bonus**: +0.2 when all issues are resolved
- **Penalties**: -0.1 for wrong fixes, -0.05 for invalid actions

## Grading

```
Score = (issues_fixed / issues_total) × efficiency_bonus
efficiency_bonus = 1.0 + (remaining_steps / max_steps × 0.3)
```

Faster fixes earn up to 30% bonus. Score capped at 1.0.

## Baseline Scores

| Task | Score | Reward | Issues Found | Issues Fixed | Steps |
|------|-------|--------|-------------|-------------|-------|
| Easy | 0.0000 | 0.34 | 2/2 | 0/2 | 6 |
| Medium | 0.0000 | 0.53 | 3/3 | 0/3 | 9 |
| Hard | 0.0000 | 0.87 | 5/5 | 0/5 | 15 |

> The rule-based baseline only explores (inspects) without submitting fixes, establishing a floor. An LLM agent that also fixes issues will score significantly higher.

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

### Run Baseline

```bash
# Rule-based baseline (no API key needed)
python scripts/baseline_inference.py --mode rule

# LLM-powered baseline
export OPENAI_API_KEY=your-key
python scripts/baseline_inference.py --mode llm
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, start new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks with action schemas |
| `/grader` | POST | Get grader score for completed episode |
| `/baseline` | POST | Run baseline inference on all tasks |
| `/schema` | GET | Get action/observation JSON schemas |
| `/ws` | WS | WebSocket for persistent sessions |

## Project Structure

```
api_debug_env/
├── inference.py        # ★ MANDATORY hackathon inference script
├── models.py           # Pydantic Action & Observation models
├── scenarios.py        # 3 task scenarios with issues, logs, configs
├── client.py           # WebSocket client for the environment
├── openenv.yaml        # OpenEnv metadata
├── pyproject.toml      # Dependencies & build config
├── server/
│   ├── app.py                        # FastAPI application
│   ├── api_debug_env_environment.py  # Core environment logic
│   └── Dockerfile                    # Container build
└── scripts/
    └── baseline_inference.py         # Original baseline agent script
```

## License

BSD-style license. See LICENSE file.
