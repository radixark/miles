import json
import os
import pickle
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import chz
import httpx
import pytest
from tinker_agent import harbor
from tinker_agent import train as recipe
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl import train
from tinker_cookbook.rl.data_processing import trajectory_to_data

TEMPLATE = json.dumps(
    {
        "environment": {"type": "docker"},
        "agent": {
            "name": "terminus-2",
            "model_name": "openai/policy",
            "kwargs": {
                "api_base": "{proxy_url}/v1",
                "llm_kwargs": {"api_key": "{proxy_api_key}"},
            },
        },
    }
)


@pytest.mark.asyncio
async def test_harbor_http_calls_become_unmodified_training_data(
    monkeypatch, renderer, sampler_factory, capture_store
):
    sampler = sampler_factory()
    sampler.get_tokenizer = lambda: renderer.tokenizer
    policy = TinkerTokenCompleter(sampler, max_tokens=32)
    configs = []

    async def create(config):
        configs.append(config)
        reward = float(len(configs) % 2)

        async def run():
            async with httpx.AsyncClient() as http:
                for text in ["original", "compacted", "compacted"]:
                    response = await http.post(
                        f"{config.agent.kwargs['api_base']}/chat/completions",
                        headers={"Authorization": f"Bearer {config.agent.kwargs['llm_kwargs']['api_key']}"},
                        json={"messages": [{"role": "user", "content": text}]},
                    )
                    response.raise_for_status()
            return SimpleNamespace(exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": reward}))

        return SimpleNamespace(run=run)

    monkeypatch.setattr(harbor.Trial, "create", create)
    strategy = harbor.HarborRolloutStrategy(
        "role_colon",
        TEMPLATE,
        capture_store.base_url,
        "test",
        harbor.HarborConfig(trial_config="unused.json", max_parallel_trials_per_group=1),
    )
    strategy = pickle.loads(pickle.dumps(strategy))
    group = harbor.HarborGroup(Path("/tasks/task"), 2)
    first = await strategy.execute(group, policy)
    result = await strategy.execute(group, policy)
    assert len({c.agent.kwargs["api_base"] for c in configs}) == 4
    assert all(c.task.path == Path("/tasks/task") for c in configs)
    assert await group.compute_group_rewards(result.trajectories, result.envs) == [
        (1, {"stop/completed": 1.0}),
        (0, {"stop/completed": 1.0}),
    ]
    for trajectory in [*first.trajectories, *result.trajectories]:
        datums = trajectory_to_data(trajectory, traj_advantage=1.0)
        assert len(datums) == 3
        assert [datum.loss_fn_inputs["target_tokens"].data[-1] for datum in datums] == [1000] * 3
        assert [datum.loss_fn_inputs["logprobs"].data[-1] for datum in datums] == [-0.25] * 3
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.ConnectError):
            await http.get(configs[0].agent.kwargs["api_base"])


@pytest.mark.parametrize(
    "failure,reason",
    [
        (None, "completed"),
        ("AgentTimeoutError", "rollout_timeout"),
        ("ContextWindowExceededError", "context_overflow"),
        ("OutputTokenExceededError", "max_tokens"),
    ],
)
def test_only_scored_trial_outcomes_are_accepted(failure, reason):
    result = SimpleNamespace(
        exception_info=None if failure is None else SimpleNamespace(exception_type=failure, exception_message=""),
        verifier_result=SimpleNamespace(rewards={"reward": 0.5}),
    )
    assert harbor.trial_outcome(result, "reward") == (0.5, reason)


@pytest.mark.parametrize(
    "failure,rewards",
    [
        ("VerifierTimeoutError", {"reward": 0}),
        ("NonZeroAgentExitCodeError", {"reward": 0}),
        (None, None),
        ("AgentTimeoutError", None),
        ("ContextWindowExceededError", None),
        ("OutputTokenExceededError", None),
        ("ApiRateLimitError", {"reward": 0}),
        (None, {"reward": float("nan")}),
    ],
)
def test_failed_or_unscored_trials_cannot_turn_into_zero_rewards(failure, rewards):
    result = SimpleNamespace(
        exception_info=(
            None if failure is None else SimpleNamespace(exception_type=failure, exception_message="failed")
        ),
        verifier_result=SimpleNamespace(rewards=rewards),
    )
    with pytest.raises((RuntimeError, ValueError)):
        harbor.trial_outcome(result, "reward")


def test_harbor_config_uses_native_defaults(tmp_path):
    trial = harbor.trial_config(TEMPLATE, tmp_path, "http://proxy/r/traj/7", 'key"quoted')
    assert trial.agent.kwargs["api_base"] == "http://proxy/r/traj/7/v1"
    assert trial.agent.kwargs["llm_kwargs"]["api_key"] == 'key"quoted'
    assert "response_length_exceeded_policy" not in trial.agent.kwargs
    assert "max_turns" not in trial.agent.kwargs


@pytest.mark.parametrize("exception_type", ["NonZeroAgentExitCodeError", "NetworkConnectionError"])
@pytest.mark.parametrize("suffix", ["", "\nConnection refused"])
def test_official_proxy_context_error_is_scored_only_when_it_is_the_terminal_failure(suffix, exception_type):
    message = (
        "Command failed (exit 1): mini-swe-agent\nstdout: traceback\n"
        "BadRequestError: litellm.BadRequestError: OpenAIException - prompt is too long: \n"
        "66070 prompt tokens plus 8192 max_tokens exceed the model's context window\n\n"
        f"stderr: None{suffix}"
    )
    result = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=exception_type, exception_message=message),
        verifier_result=SimpleNamespace(rewards={"reward": 0}),
    )
    if suffix:
        with pytest.raises(RuntimeError, match="Harbor trial failed"):
            harbor.trial_outcome(result, "reward")
    else:
        assert harbor.trial_outcome(result, "reward") == (0, "context_overflow")
        result.verifier_result = None
        with pytest.raises(RuntimeError, match="no verifier reward"):
            harbor.trial_outcome(result, "reward")


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        [
            "loss_fn=ppo",
            'loss_fn_config={"clip_low_threshold": 0.1}',
            "load_checkpoint_path=tinker://checkpoint",
            "rolling_save_every=3",
            "ttl_seconds=None",
            "lora_rank=8",
            "async_config.max_steps_off_policy=2",
            "async_config.groups_per_batch=3",
            "dataset_builder.epochs=2",
        ],
    ],
)
def test_cli_preserves_official_config_and_capture_lifetime(tmp_path, monkeypatch, overrides):
    config_path = tmp_path / "trial.json"
    config_path.write_text(TEMPLATE)
    arguments = [
        "model_name=model",
        "renderer_name=role_colon",
        "learning_rate=0.00001",
        "max_tokens=32",
        f"log_path={tmp_path / 'logs'}",
        f"dataset_builder.tasks_dir={tmp_path}",
        *overrides,
    ]
    expected = (
        chz.Blueprint(train.Config)
        .apply({"recipe_name": "harbor-tinker", "dataset_builder": harbor.HarborDatasetBuilder})
        .make_from_argv(arguments)
    )
    events = []
    seen = []

    @contextmanager
    def capture_to_store(run_id, *, data_dir):
        assert data_dir == Path(expected.log_path) / "capture"
        events.append("capture-enter")
        yield SimpleNamespace(base_url="http://store")
        events.append("capture-exit")

    async def train_main(config):
        events.append("train")
        seen.append(config)

    monkeypatch.setattr(recipe, "capture_to_store", capture_to_store)
    monkeypatch.setattr(train, "main", train_main)
    recipe.cli([*["config." + arg for arg in arguments], f"harbor.trial_config={config_path}"])
    assert events == ["capture-enter", "train", "capture-exit"]
    (actual,) = seen
    assert type(actual) is train.Config
    assert isinstance(actual.rollout_error_tolerance, harbor.HarborRolloutStrategy)
    assert chz.asdict(chz.replace(actual, rollout_error_tolerance=None)) == chz.asdict(expected)


@pytest.mark.parametrize(
    "argument,match",
    [
        ("config.rollout_error_tolerance=True", "recipe owns config.rollout_error_tolerance"),
        ("config.temperature=0.5", "Set temperature in the Harbor agent config"),
    ],
)
def test_conflicting_rollout_options_fail_before_training(tmp_path, argument, match):
    with pytest.raises(ValueError, match=match):
        recipe.cli(
            [
                "config.model_name=model",
                "config.renderer_name=role_colon",
                "config.learning_rate=0.00001",
                "config.max_tokens=32",
                f"config.log_path={tmp_path}",
                f"config.dataset_builder.tasks_dir={tmp_path}",
                "harbor.trial_config=unused.json",
                argument,
            ]
        )


def test_installed_cli_help_lists_official_and_harbor_options():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "tinker_agent.train", "--help"],
        capture_output=True,
        text=True,
        # chz re-raises CLI help under pytest; exercise its normal shell behavior.
        env={key: value for key, value in os.environ.items() if key != "PYTEST_VERSION"},
    )
    assert result.stderr == "", result.stderr
    assert "config.rolling_save_every" in result.stdout
    assert "config.dataset_builder.tasks_dir" in result.stdout
    assert "harbor.trial_config" in result.stdout
