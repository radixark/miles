import shlex
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import scaling

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


class TestScalingRelaunch:
    @pytest.mark.parametrize(
        "flag,replicas,gpus_per_cell,expected",
        [("--actor-num-gpus-per-node", 3, 2, "6"), ("--rollout-num-gpus", 2, 1, "2")],
    )
    async def test_relaunch_replaces_only_the_scaled_pool_size(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
        replicas: int,
        gpus_per_cell: int,
        expected: str,
    ) -> None:
        """A resize converts replicas to GPUs and preserves the other pool and launch identity."""
        mode = scaling.compute_scaling_mode(("train",))
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="test", run_id="same-run")
        observed: list[dict[str, Any]] = []
        monkeypatch.setattr(scaling, "run_training", lambda **kwargs: observed.append(kwargs))

        await scaling._relaunch_at_size(
            replicas,
            train_args="--actor-num-gpus-per-node 4 --rollout-num-gpus 3 --save /checkpoints/same",
            mode=mode,
            config=config,
            scaling_flag=flag,
            num_gpus_per_cell=gpus_per_cell,
        )

        assert len(observed) == 1
        call = observed[0]
        assert call["mode"] is mode
        assert call["config"] is config
        tokens = shlex.split(call["train_args"])
        assert tokens.count(flag) == 1
        assert tokens[tokens.index(flag) + 1] == expected
        other = "--rollout-num-gpus" if flag == "--actor-num-gpus-per-node" else "--actor-num-gpus-per-node"
        assert tokens[tokens.index(other) + 1] == ("3" if other == "--rollout-num-gpus" else "4")
        assert tokens[tokens.index("--save") + 1] == "/checkpoints/same"


class TestScalingPrerequisites:
    @pytest.mark.parametrize(
        "backend,namespace,message",
        [(ClusterBackend.RAY, "test", "only"), (ClusterBackend.KUBERNETES, None, "namespace")],
    )
    def test_unsupported_cluster_or_missing_namespace_fails_before_preparation(
        self, monkeypatch: pytest.MonkeyPatch, backend: ClusterBackend, namespace: str | None, message: str
    ) -> None:
        """A scaling scenario cannot begin preparation without an addressable Kubernetes pool."""
        config = ExecuteTrainConfig(cluster_backend=backend, namespace=namespace)
        monkeypatch.setattr(scaling.command_utils, "default_config", lambda: config)
        with pytest.raises(AssertionError, match=message):
            scaling.run_scaling_scenario(
                test_name="scale",
                mode=scaling.compute_scaling_mode(("train",)),
                pool_id="actor",
                cell_type="actor",
                scaling_flag="--actor-num-gpus-per-node",
                num_gpus_per_cell=2,
                read_counts_of_rollout_id=lambda events, path: {},
                counted="cells",
            )
