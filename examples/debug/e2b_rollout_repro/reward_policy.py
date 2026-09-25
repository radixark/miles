"""Classify budget exhaustion separately from infrastructure failures."""
import json
from pathlib import Path
from typing import Any


def apply_reward_policy(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    exception_type = None
    trial_dir = result.get("trial_dir")
    if trial_dir:
        path = Path(trial_dir) / "result.json"
        if path.exists():
            trial = json.loads(path.read_text())
            exception_type = (trial.get("exception_info") or {}).get("exception_type")
    agent_timeout = exception_type == "AgentTimeoutError"
    truncated = result.get("exit_status") == "SequenceLengthLimitExceeded"
    if truncated:
        result["reward"] = 0.0
        result["reward_source"] = "token_budget_exhausted"
    elif agent_timeout:
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        result["reward"] = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))
        result["reward_source"] = "verifier" if rewards else "agent_time_budget_exhausted"
        result["pilot_agent_timeout"] = True
    result["pilot_valid_for_training"] = truncated or agent_timeout or (
        result.get("exit_status") == "Submitted" and bool(result.get("eval_report"))
    )
    return result
