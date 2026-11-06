"""Offline tests for the tau2-bench adapter's episode classification (no tau2, no engine).

``tau2_agent_function.run`` records in the metadata it returns whether the
episode is valid; ``tau2_generate.generate`` turns that into the sample status.
The ``tau2`` package is faked in ``sys.modules`` so these run where it is not
installed, and the simulation itself is replaced per test.
"""

import asyncio
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.types import Sample

from . import EXAMPLE_DIR

_FAKE_TAU2_MODULES = {
    "tau2": {},
    "tau2.data_model": {},
    "tau2.data_model.semantic_communication": {"CommunicateScorer": object, "CommunicateScoring": object},
    "tau2.data_model.simulation": {"SimulationRun": object, "TextRunConfig": object},
    "tau2.evaluator": {},
    "tau2.evaluator.evaluator_nl_assertions": {},
    "tau2.evaluator.evaluator": {"EvaluationType": SimpleNamespace(ALL="all")},
    "tau2.run": {"get_tasks": None, "run_single_task": None},
}


def _load_example(monkeypatch: pytest.MonkeyPatch, filename: str):
    for name, attributes in _FAKE_TAU2_MODULES.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        f"tau2_example_{filename.removesuffix('.py')}", EXAMPLE_DIR / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EndpointError(Exception):
    """The shape litellm gives an HTTP error from the policy, the user simulator or the judge."""

    def __init__(self, status_code: int):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


# the generate wrapper adds the router URL the user simulator and the judge default to
_METADATA = {
    "domain": "retail",
    "task_split": "train",
    "task_id": "7",
    "tau2_user_api_base": "http://10.0.0.1:30000/v1",
}


def _run(monkeypatch: pytest.MonkeyPatch, module, simulate):
    monkeypatch.setattr(module, "_run_simulation", simulate)
    return asyncio.run(module.run("http://127.0.0.1:1/sessions/s", prompt="retail/7", metadata=dict(_METADATA)))


class TestRun:
    def test_a_scored_episode_keeps_tau2s_reason_and_reward(self, monkeypatch):
        module = _load_example(monkeypatch, "tau2_agent_function.py")
        simulation = SimpleNamespace(
            reward_info=SimpleNamespace(reward=1.0),
            termination_reason=SimpleNamespace(value="user_stop"),
            messages=[None] * 12,
        )

        result = _run(monkeypatch, module, lambda *args: simulation)

        assert result == {
            "reward": 1.0,
            "tau2_domain": "retail",
            "tau2_task_id": "7",
            "tau2_episode_valid": True,
            "tau2_termination_reason": "user_stop",
            "tau2_num_messages": 12,
        }

    def test_tau2_rejecting_an_empty_reply_is_a_scored_zero(self, monkeypatch):
        """The policy's thinking spent the whole turn: the policy's outcome, not a failure of the harness."""
        module = _load_example(monkeypatch, "tau2_agent_function.py")

        def simulate(*args):
            raise ValueError("AssistantMessage must have either content or tool_calls. Got AssistantMessage\n...")

        result = _run(monkeypatch, module, simulate)

        assert result == {
            "reward": 0.0,
            "tau2_domain": "retail",
            "tau2_task_id": "7",
            "tau2_episode_valid": True,
            "tau2_termination_reason": "empty_reply",
        }

    @pytest.mark.parametrize(
        "exc",
        [
            _EndpointError(400),  # the judge's model is not served, or the engine refuses the prompt length
            _EndpointError(502),  # the engine behind the session
            ValueError("Simulation is done"),  # another tau2 ValueError
            json.JSONDecodeError("AssistantMessage must have either content or tool_calls", "", 0),  # a subclass
            RuntimeError("The agent process exited with code 1"),
        ],
        ids=["endpoint_400", "policy_502", "other_value_error", "value_error_subclass", "agent_process"],
    )
    def test_anything_else_raised_invalidates_the_episode(self, monkeypatch, exc):
        module = _load_example(monkeypatch, "tau2_agent_function.py")

        def simulate(*args):
            raise exc

        result = _run(monkeypatch, module, simulate)

        assert result == {"reward": 0.0, "tau2_domain": "retail", "tau2_task_id": "7", "tau2_episode_valid": False}

    def test_the_episode_timeout_is_unchanged(self, monkeypatch):
        module = _load_example(monkeypatch, "tau2_agent_function.py")
        monkeypatch.setenv("TAU2_EPISODE_TIMEOUT_S", "0.01")

        def simulate(*args):
            import time

            time.sleep(0.5)
            return SimpleNamespace(reward_info=SimpleNamespace(reward=1.0))

        result = _run(monkeypatch, module, simulate)

        assert result["tau2_termination_reason"] == "episode_timeout"
        assert result["tau2_episode_valid"] is True
        assert result["reward"] == 0.0


class TestGenerate:
    @staticmethod
    def _generate(monkeypatch, samples):
        module = _load_example(monkeypatch, "tau2_generate.py")
        output = GenerateFnOutput(samples=samples)

        async def collect(input: GenerateFnInput) -> GenerateFnOutput:
            return output

        monkeypatch.setattr(agentic_tool_call, "generate", collect)
        args = SimpleNamespace(sglang_router_ip="10.0.0.1", sglang_router_port=30000)
        input = GenerateFnInput(
            state=SimpleNamespace(args=args), sample=Sample(prompt="retail/7"), sampling_params={}, evaluation=False
        )
        assert asyncio.run(module.generate(input)) is output
        assert input.sample.metadata["tau2_user_api_base"] == "http://10.0.0.1:30000/v1"

    @staticmethod
    def _sample(status: Sample.Status, metadata: dict) -> Sample:
        return Sample(tokens=[1, 2, 3], response_length=1, status=status, metadata=metadata)

    @pytest.mark.parametrize(
        ("assembled", "metadata", "expected"),
        [
            (
                Sample.Status.COMPLETED,
                {"tau2_episode_valid": True, "tau2_termination_reason": "user_stop"},
                Sample.Status.COMPLETED,
            ),
            # the episode ran past --max-seq-len: sample assembly cut it, the reward is still tau2's
            (
                Sample.Status.TRUNCATED,
                {"tau2_episode_valid": True, "tau2_termination_reason": "user_stop", "reward": 1.0},
                Sample.Status.TRUNCATED,
            ),
            (
                Sample.Status.COMPLETED,
                {"tau2_episode_valid": True, "tau2_termination_reason": "episode_timeout"},
                Sample.Status.COMPLETED,
            ),
            # the policy's thinking spent the turn: the session cut the turn, tau2 rejected the empty reply
            (
                Sample.Status.TRUNCATED,
                {"tau2_episode_valid": True, "tau2_termination_reason": "empty_reply", "reward": 0.0},
                Sample.Status.TRUNCATED,
            ),
            (Sample.Status.COMPLETED, {"tau2_episode_valid": False}, Sample.Status.ABORTED),
            (Sample.Status.TRUNCATED, {"tau2_episode_valid": False}, Sample.Status.ABORTED),
            # the agent process never reported (it died before returning)
            (Sample.Status.COMPLETED, {}, Sample.Status.ABORTED),
            # collect_samples already gave up on the session
            (Sample.Status.ABORTED, {}, Sample.Status.ABORTED),
        ],
        ids=[
            "scored",
            "scored_past_max_seq_len",
            "episode_timeout",
            "empty_reply",
            "invalid",
            "invalid_length_cut",
            "never_reported",
            "collect_aborted",
        ],
    )
    def test_the_status_follows_how_the_episode_ended(self, monkeypatch, assembled, metadata, expected):
        sample = self._sample(assembled, metadata)

        self._generate(monkeypatch, sample)

        assert sample.status == expected

    def test_every_leaf_of_a_v2_session_is_classified(self, monkeypatch):
        scored = self._sample(
            Sample.Status.COMPLETED, {"tau2_episode_valid": True, "tau2_termination_reason": "user_stop"}
        )
        invalid = self._sample(Sample.Status.COMPLETED, {"tau2_episode_valid": False})

        self._generate(monkeypatch, [scored, invalid])

        assert (scored.status, invalid.status) == (Sample.Status.COMPLETED, Sample.Status.ABORTED)
