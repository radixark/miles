import shlex
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_deterministic as scenario

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig


def test_each_takeover_changes_only_the_rollout_data_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real relaunch callback rewrites only the rollout path and preserves every other argv token."""
    dump_dir = str(tmp_path / "target")
    installed = "--lr=0.125 --save /fixed --save-debug-rollout-data /gen0 --custom 'two words'"
    driver = Mock(spec=scenario.HotRestartDriver)
    driver.records = []
    constructor = Mock(return_value=driver)
    relaunch = Mock()
    monkeypatch.setattr(scenario, "HotRestartDriver", constructor)
    monkeypatch.setattr(scenario, "driving_hot_restarts", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(scenario, "relaunch_with_hot_restart", relaunch)
    monkeypatch.setattr(scenario, "read_installed_args", lambda _dump_dir: installed)
    monkeypatch.setattr(scenario, "assert_generations_recorded_their_steps", Mock())
    monkeypatch.setattr(scenario, "assert_take_overs_carried_rollout_only_args", Mock())

    with scenario._driving_take_overs_of(
        restart_mode=scenario.CHECKPOINTED,
        mode=scenario._MODE,
        dump_dir=dump_dir,
        config=ExecuteTrainConfig(run_id="unit-test"),
    ):
        callback = constructor.call_args.kwargs["relaunch"]
        for _generation in (1, 2):
            driver.records.append(object())
            callback(None)

    expected_tail = ["--custom", "two words"]
    for generation, call in enumerate(relaunch.call_args_list, start=1):
        assert shlex.split(call.kwargs["train_args"]) == [
            "--lr=0.125",
            "--save",
            "/fixed",
            "--save-debug-rollout-data",
            f"{dump_dir}/rollout_data/generation_{generation}/{{rollout_id}}.pt",
            *expected_tail,
        ]
    assert relaunch.call_count == 2
