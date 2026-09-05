from __future__ import annotations

import argparse

import pytest

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend, resolve_worker_comm_backend


class TestTheAutomaticChoice:
    def test_ray_keeps_talking_over_ray_until_the_default_flips(self):
        """The flag exists so both modes coexist; today an unset flag must change nothing for ray users."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested=None)

        assert chosen == WorkerCommBackend.RAY

    def test_kubernetes_talks_over_rpc(self):
        """A pod is not an actor, so the only way to call it is the server it serves."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested=None)

        assert chosen == WorkerCommBackend.RPC


class TestTheExplicitChoice:
    @pytest.mark.parametrize("requested", ["ray", "rpc"])
    def test_ray_accepts_both_modes(self, requested: str):
        """Ray is where the two modes coexist, which is what makes a gradual switch possible."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested=requested)

        assert chosen == WorkerCommBackend(requested)

    def test_kubernetes_accepts_rpc(self):
        """Naming the backend that is already in use must not be an error."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested="rpc")

        assert chosen == WorkerCommBackend.RPC

    def test_kubernetes_refuses_ray_communication(self):
        """There is no actor to call, so accepting the flag would fail much later and far less clearly."""
        with pytest.raises(AssertionError, match="worker-comm-backend"):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested="ray")

    def test_an_unknown_backend_is_rejected(self):
        """A typo must not silently fall back to the default."""
        with pytest.raises(ValueError):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested="grpc")


class TestTheChoiceThatIsStored:
    @pytest.mark.parametrize("requested, expected", [([], "ray"), (["--worker-comm-backend", "rpc"], "rpc")])
    def test_validation_leaves_the_resolved_wire_on_the_arguments(self, requested: list[str], expected: str):
        """The wire is resolved once, so everything downstream reads a value rather than resolving again."""
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args([*requested, "--rollout-batch-size", "64", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.worker_comm_backend == expected

    @pytest.mark.parametrize("requested, expected", [([], "rpc"), (["--worker-comm-backend", "rpc"], "rpc")])
    def test_kubernetes_is_resolved_through_the_public_validation_too(self, requested: list[str], expected: str):
        """Every downstream reader takes the wire off args, so the k8s path must be resolved there as well."""
        args = _validated_args([*requested, "--cluster-backend", "kubernetes"])

        assert args.worker_comm_backend == expected

    def test_asking_kubernetes_for_ray_fails_the_whole_validation(self):
        """A combination that cannot work must stop the run at argument time, not at the first call."""
        with pytest.raises(AssertionError, match="worker-comm-backend"):
            _validated_args(["--cluster-backend", "kubernetes", "--worker-comm-backend", "ray"])


def _validated_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args([*argv, "--rollout-batch-size", "64", "--num-rollout", "1"])
    miles_validate_args(args)
    return args
