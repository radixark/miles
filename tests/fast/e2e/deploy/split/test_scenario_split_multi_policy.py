import dataclasses

import pytest
from examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split import (
    build_deployment_train_args,
    compute_deployment_identities,
)
from examples.multi_policy.run_solver_verifier_gsm8k import ScriptArgs
from tests.e2e.deploy.conftest_deploy.split import scenario_split_multi_policy as scenario

from miles.utils.workers.types import ClusterBackend

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"


@pytest.fixture
def args() -> ScriptArgs:
    return ScriptArgs(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID)


class TestBuildDeployments:
    def test_the_scenario_installs_every_part_the_example_declares_and_no_other(self, args):
        """A scenario naming its own topology would deploy a run the example's README never describes."""
        deployments = scenario._build_deployments(args)

        assert [
            (one.deploy_component, one.deploy_instance_id) for one in deployments
        ] == compute_deployment_identities(args)

    def test_every_part_is_launched_with_the_arguments_the_example_composes_for_it(self, args):
        """Arguments rebuilt here rather than imported would drift from the commands a reader types."""
        for one in scenario._build_deployments(args):
            assert one.train_args == build_deployment_train_args(
                dataclasses.replace(
                    args, deploy_component=one.deploy_component, deploy_instance_id=one.deploy_instance_id
                )
            )
