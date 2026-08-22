import textwrap
from pathlib import Path
from typing import Any

import pytest

from miles.utils.external_utils.command_utils.helm_backend import command_job

NAMESPACE = "rl"


def _backend(helm_values: tuple[str, ...]) -> Any:
    pytest.importorskip("torch")
    from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
    from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend

    return KubernetesCommandBackend(ExecuteTrainConfig(namespace=NAMESPACE, helm_values=helm_values))


def _values_pinned_to(tmp_path: Path, hostname: str) -> tuple[str, ...]:
    values = tmp_path / "infra-pinned.yaml"
    values.write_text(
        textwrap.dedent(
            f"""
            infra:
              scheduling:
                nodeSelector:
                  kubernetes.io/hostname: {hostname}
            """
        )
    )
    return (str(values),)


def _record_completions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    completions: list[int] = []

    def fake_run_on_nodes(context: Any, cmd: str, **kwargs: Any) -> list[str | None]:
        completions.append(kwargs["completions"])
        return [None] * kwargs["completions"]

    monkeypatch.setattr(command_job, "run_on_nodes", fake_run_on_nodes)
    return completions


class TestTheNodesACommandAsksFor:
    def test_refuses_a_multi_node_command_pinned_to_a_single_host(self, monkeypatch, tmp_path):
        """A two-node command under a single-host nodeSelector is refused before any Job reaches the cluster."""
        completions = _record_completions(monkeypatch)
        backend = _backend(_values_pinned_to(tmp_path, "gpu-1"))

        with pytest.raises(AssertionError, match="gpu-1"):
            backend.exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=2, num_gpus_per_node=1)

        assert completions == []

    def test_runs_a_single_node_command_on_a_pinned_deployment(self, monkeypatch, tmp_path):
        """One node always fits the host it is pinned to, so the command is installed as before."""
        completions = _record_completions(monkeypatch)
        backend = _backend(_values_pinned_to(tmp_path, "gpu-1"))

        backend.exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=1, num_gpus_per_node=1)

        assert completions == [1]

    def test_runs_a_multi_node_command_when_no_host_is_pinned(self, monkeypatch):
        """Without a host pin the backend keeps letting the scheduler place every completion."""
        completions = _record_completions(monkeypatch)
        backend = _backend(())

        backend.exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=2, num_gpus_per_node=1)

        assert completions == [2]
