import subprocess
from typing import Any

import pytest

from miles.utils.test_utils import kubectl_reads


class TestReadReplicas:
    @pytest.mark.parametrize("stdout,expected", [(" 3\n", 3), ("0\n", 0)])
    def test_workload_replica_reads_use_the_requested_namespace_and_checked_timeout(
        self, monkeypatch: pytest.MonkeyPatch, stdout: str, expected: int
    ) -> None:
        """Replica reads address the named workload and parse its scalar output without losing zero."""
        calls: list[tuple[list[str], dict[str, Any]]] = []

        def run_process(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout)

        monkeypatch.setattr(kubectl_reads, "run_process", run_process)

        assert kubectl_reads.read_replicas(namespace="isolated", workload="actor-pool") == expected
        assert calls == [
            (
                [
                    "kubectl",
                    "get",
                    "leaderworkersets.leaderworkerset.x-k8s.io",
                    "actor-pool",
                    "--namespace",
                    "isolated",
                    "-o",
                    "jsonpath={.spec.replicas}",
                ],
                {"capture_output": True, "check": True, "timeout": kubectl_reads.KUBECTL_TIMEOUT_SECONDS},
            )
        ]

    @pytest.mark.parametrize("failure", ["invalid", "command"])
    def test_failed_or_malformed_replica_reads_do_not_become_zero(
        self, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        """Unavailable replica evidence fails instead of silently claiming an empty pool."""

        def run_process(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if failure == "command":
                raise subprocess.CalledProcessError(returncode=1, cmd=argv)
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="not-a-number")

        monkeypatch.setattr(kubectl_reads, "run_process", run_process)
        with pytest.raises(ValueError if failure == "invalid" else subprocess.CalledProcessError):
            kubectl_reads.read_replicas(namespace="isolated", workload="actor-pool")
