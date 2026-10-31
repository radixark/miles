from dataclasses import replace
from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import RELEASE
from tests.fast.utils.soak.deploy.deploy_fakes import _launch_spec
from tests.utils.soak.deploy.utils import compute_checkpoint_dir, compute_hot_restart_config

from miles.utils.workers.types import HotRestartComponent


class TestComputeHotRestartConfig:
    def test_the_relaunch_of_the_installed_release_restarts_both_components(self) -> None:
        """Only hot_restart changes, naming the orchestration and rollout executor components."""
        config = _launch_spec().config

        relaunch = compute_hot_restart_config(config, installed_release=RELEASE)

        assert relaunch.parsed_hot_restart == [HotRestartComponent.ORCHESTRATION, HotRestartComponent.ROLLOUT_EXECUTOR]
        assert replace(relaunch, hot_restart="") == config
        assert config.hot_restart == ""

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"run_id": "other"}, id="run_id"),
            pytest.param({"deploy_instance_id": "b"}, id="instance"),
        ],
    )
    def test_a_config_that_would_install_another_release_is_refused(self, overrides: dict[str, str]) -> None:
        """A relaunch under its own release would leave the watched trainers behind."""
        with pytest.raises(AssertionError, match="upgrades the release that is already up"):
            compute_hot_restart_config(_launch_spec(**overrides).config, installed_release=RELEASE)


class TestComputeCheckpointDir:
    def test_checkpoints_live_in_the_dump_dir(self) -> None:
        """The observer and the run must agree on one checkpoint directory under the dump dir."""
        assert compute_checkpoint_dir("/dumps/run") == Path("/dumps/run/checkpoints")
