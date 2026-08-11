import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import LAYOUT

from miles.utils.external_utils.command_utils.helm_backend.launcher.values import pool_entry
from miles.utils.workers.worker_spec import BaseWorkerSpec, SchedulingSpec


class TestBuildEntry:
    def test_refuses_a_spec_with_no_launch_mechanism(self):
        """A pool without a command or RPC server cannot launch a worker."""
        spec = BaseWorkerSpec(
            name="unsupported-worker",
            port_infos=[],
            env_var=lambda context: {},
            scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        )

        with pytest.raises(AssertionError, match="neither launched by a command nor served over rpc"):
            pool_entry.build_entry(spec, plan=LAYOUT, addresses={})
