from typing import Any

import pytest
from tests.fast.utils.external_utils.command_utils.fake_launch_guard import RecordingLaunchGuard

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend import backend
from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend
from miles.utils.workers.types import ClusterBackend


class TestApiServerHost:
    @pytest.mark.parametrize(
        ("run_id", "namespace"),
        [
            ("", "rl"),
            ("260101-000000-000", ""),
        ],
    )
    def test_an_api_server_host_requires_both_run_id_and_namespace(self, run_id: str, namespace: str) -> None:
        """An api server host requires both parts that identify its Kubernetes service."""
        config = ExecuteTrainConfig(run_id=run_id, namespace=namespace)
        backend = KubernetesCommandBackend(config)

        with pytest.raises(AssertionError, match="run_id and namespace"):
            backend.api_server_host(config)


class TestExecuteTrainCarriesTheGuard:
    def test_the_launcher_is_handed_the_guard_and_config_of_the_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A kubernetes launch that dropped the guard would install through the unguarded helm calls."""
        launched: list[dict[str, Any]] = []
        monkeypatch.setattr(backend.entrypoint, "execute_train", lambda **kwargs: launched.append(kwargs))
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, run_id="run-a", namespace="rl")
        guard = RecordingLaunchGuard()

        config.create_backend().execute_train(
            train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None, guard=guard
        )

        assert len(launched) == 1
        assert launched[0]["guard"] is guard
        assert launched[0]["config"] is config
        assert guard.calls == []
