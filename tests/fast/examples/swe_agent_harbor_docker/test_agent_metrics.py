"""Agent metrics must reflect what an agent measured, not a default of zero.

The example directory name is not a Python identifier, so the module is loaded
by path the same way the other example tests do it.
"""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATE_SCRIPT = REPO_ROOT / "examples" / "swe-agent-harbor-docker" / "generate.py"

# An agent that instruments every turn reports the full key set.
FULLY_INSTRUMENTED = {
    "turns": 30,
    "tool_calls": 47,
    "model_query_time_sum": 120.0,
    "env_execution_time_sum": 60.0,
    "eval_time": 12.0,
    "agent_run_time": 400.0,
    "time_per_turn": 13.3,
    "model_query_time_avg": 4.0,
    "env_execution_time_avg": 2.0,
    "model_time_ratio": 0.3,
    "env_time_ratio": 0.15,
    "eval_time_ratio": 0.03,
    "total_time": 500.0,
}
# Another agent reports only wall-clock totals: no turn, tool-call or timing breakdown.
TOTALS_ONLY = {"eval_time": 42.8, "agent_run_time": 486.8, "total_time": 559.3}

UNREPORTED_BY_TOTALS_ONLY = [
    "agent/turns_mean",
    "agent/turns_sum",
    "agent/tool_calls_mean",
    "agent/tool_calls_sum",
    "agent/model_query_time_sum_mean",
    "agent/env_execution_time_sum_mean",
    "agent/time_per_turn",
    "agent/model_query_time_avg",
    "agent/env_execution_time_avg",
    "agent/model_time_ratio",
    "agent/env_time_ratio",
    "agent/eval_time_ratio",
]


@pytest.fixture(scope="module")
def generate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swe_agent_harbor_docker_generate", GENERATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_with(agent_metrics: dict) -> SimpleNamespace:
    return SimpleNamespace(metadata={"agent_metrics": agent_metrics})


def test_fully_instrumented_agent_reports_every_metric(generate_module: ModuleType) -> None:
    metrics = generate_module.aggregate_agent_metrics([sample_with(FULLY_INSTRUMENTED)])

    assert metrics["agent/turns_mean"] == 30
    assert metrics["agent/tool_calls_mean"] == 47
    assert metrics["agent/tool_calls_sum"] == 47
    assert metrics["agent/time_per_turn"] == pytest.approx(13.3)
    assert metrics["agent/total_time_mean"] == pytest.approx(500.0)


def test_unreported_keys_are_omitted_rather_than_logged_as_zero(generate_module: ModuleType) -> None:
    """A zero here is indistinguishable from an agent that made no tool calls."""
    metrics = generate_module.aggregate_agent_metrics([sample_with(TOTALS_ONLY)])

    for key in UNREPORTED_BY_TOTALS_ONLY:
        assert key not in metrics, f"{key} was logged despite never being measured"

    # What the agent did report still comes through.
    assert metrics["agent/eval_time_mean"] == pytest.approx(42.8)
    assert metrics["agent/agent_run_time_mean"] == pytest.approx(486.8)
    assert metrics["agent/total_time_mean"] == pytest.approx(559.3)


def test_mixed_batch_averages_only_over_agents_that_measured_the_key(generate_module: ModuleType) -> None:
    """One silent agent must not halve the tool-call count of the batch."""
    metrics = generate_module.aggregate_agent_metrics([sample_with(FULLY_INSTRUMENTED), sample_with(TOTALS_ONLY)])

    assert metrics["agent/tool_calls_mean"] == 47
    assert metrics["agent/turns_mean"] == 30
    assert metrics["agent/time_per_turn"] == pytest.approx(13.3)
    # Keys both agents report still average across the whole batch.
    assert metrics["agent/total_time_mean"] == pytest.approx((500.0 + 559.3) / 2)
    assert metrics["agent/total_time_min"] == pytest.approx(500.0)
    assert metrics["agent/total_time_max"] == pytest.approx(559.3)


def test_zero_is_kept_when_an_agent_actually_measures_zero(generate_module: ModuleType) -> None:
    """Omitting unreported keys must not swallow a genuine zero measurement."""
    metrics = generate_module.aggregate_agent_metrics([sample_with({**FULLY_INSTRUMENTED, "tool_calls": 0})])

    assert metrics["agent/tool_calls_mean"] == 0
    assert metrics["agent/tool_calls_sum"] == 0


def test_samples_without_agent_metrics_are_ignored(generate_module: ModuleType) -> None:
    assert generate_module.aggregate_agent_metrics([sample_with({})]) == {}
    assert generate_module.aggregate_agent_metrics([SimpleNamespace(metadata={})]) == {}
    assert generate_module.aggregate_agent_metrics([]) == {}
