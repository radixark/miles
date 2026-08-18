import dataclasses

import pytest
from examples.infra_features.split_deployment.address_book import (
    DEFAULT_TRAINER_ID,
    INIT_EXPECTED_NUM_CELLS_FLAG,
    RunAddressBook,
)
from examples.infra_features.split_deployment.run_qwen3_0_6b_split import (
    ScriptArgs,
    build_deployment_train_args,
    compute_deployment_identities,
)
from tests.fast.train_args import (
    FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON,
    ROLLOUT_NUM_GPUS_FLAG,
    shared_argv,
    value_of,
    values_after,
)

from miles.ray.specs.inference import INFERENCE_CONTROLLER_ADDR_FLAG
from miles.ray.specs.train import TRAINER_CONTROLLER_ADDRS_FLAG
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, OBJECT_STORE_BACKEND_FLAG
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"


@pytest.fixture
def args() -> ScriptArgs:
    return ScriptArgs(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID)


@pytest.fixture
def train_args_of_identity(args: ScriptArgs) -> dict[tuple[DeployComponent, str | None], str]:
    return {
        (component, instance_id): build_deployment_train_args(
            dataclasses.replace(args, deploy_component=component, deploy_instance_id=instance_id)
        )
        for component, instance_id in compute_deployment_identities(args)
    }


class TestComputeDeploymentIdentities:
    def test_one_command_of_this_example_installs_one_component(self, args):
        """The point of the example is the several commands a reader types, not a script typing them."""
        assert compute_deployment_identities(args) == [
            (DeployComponent.TRAINER, None),
            (DeployComponent.INFERENCE, "e0"),
            (DeployComponent.INFERENCE, "e1"),
            (DeployComponent.PRIMARY, None),
        ]

    def test_the_command_that_drives_the_run_is_the_one_typed_last(self, args):
        """That command blocks until the run ends, so anything after it is typed into a finished run."""
        component, _ = compute_deployment_identities(args)[-1]

        assert component is DeployComponent.PRIMARY

    def test_every_engine_of_a_run_is_named_apart_from_the_others(self, args):
        """Two engine commands under one name install one release, and the run loses half its engines."""
        engines = [one for one in compute_deployment_identities(args) if one[0] is DeployComponent.INFERENCE]

        assert len(engines) == args.num_engines
        assert len({instance_id for _, instance_id in engines}) == args.num_engines


class TestBuildDeploymentTrainArgs:
    def test_an_engine_command_is_given_the_gpus_of_exactly_one_engine(self, train_args_of_identity, args):
        """A command holding two engines is the shape an unsplit run already has, and splits nothing."""
        for (component, _), train_args in train_args_of_identity.items():
            if component is DeployComponent.INFERENCE:
                assert value_of(train_args, ROLLOUT_NUM_GPUS_FLAG) == str(args.gpus_per_engine)

    def test_the_other_commands_still_count_every_engine_the_run_registers(self, train_args_of_identity, args):
        """They wait for as many engine cells as their own arguments declare, and would start on half a fleet."""
        whole_run = str(args.num_engines * args.gpus_per_engine)

        for (component, _), train_args in train_args_of_identity.items():
            if component is not DeployComponent.INFERENCE:
                assert value_of(train_args, ROLLOUT_NUM_GPUS_FLAG) == whole_run

    def test_only_the_engine_commands_are_told_where_to_register(self, train_args_of_identity):
        """Every other command holds the controller itself and refuses to be pointed at one."""
        told = {
            component
            for (component, _), train_args in train_args_of_identity.items()
            if INFERENCE_CONTROLLER_ADDR_FLAG in train_args
        }

        assert told == {DeployComponent.INFERENCE}

    def test_only_the_driving_command_is_told_where_the_trainer_is(self, train_args_of_identity):
        """The command that carries the trainer reaches it in its own process and refuses the flag."""
        told = {
            component
            for (component, _), train_args in train_args_of_identity.items()
            if TRAINER_CONTROLLER_ADDRS_FLAG in train_args
        }

        assert told == {DeployComponent.PRIMARY}

    def test_the_driving_command_dials_the_trainer_release_of_this_run(self, train_args_of_identity, args):
        """The driver reaches the trainer by name alone, so a name that drifted reaches nothing at all."""
        address_book = RunAddressBook.of_config(args)
        expected = address_book.trainer_controller_addrs_arg(
            deploy_instance_id_of_trainer_id={DEFAULT_TRAINER_ID: None}
        )

        assert values_after(
            train_args_of_identity[(DeployComponent.PRIMARY, None)], TRAINER_CONTROLLER_ADDRS_FLAG
        ) == values_after(expected, TRAINER_CONTROLLER_ADDRS_FLAG)

    def test_every_command_redeems_its_references_at_one_object_store(self, train_args_of_identity):
        """Commands that disagree on the master hand each other references nothing can read back."""
        addresses = {value_of(one, MOONCAKE_INIT_KWARGS_FLAG) for one in train_args_of_identity.values()}

        assert len(addresses) == 1
        assert all(value_of(one, OBJECT_STORE_BACKEND_FLAG) == "mooncake" for one in train_args_of_identity.values())

    def test_the_commands_agree_on_everything_the_run_itself_declares(self, train_args_of_identity):
        """Only what a command carries may differ; a drifted model or batch shape trains something else."""
        shared = [
            shared_argv(one, differing_flags=FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON)
            for one in train_args_of_identity.values()
        ]

        assert all(one == shared[0] for one in shared)

    def test_a_command_naming_no_component_of_the_run_is_refused(self, args):
        """An unsplit launch installs everything under one release, which is the shape this example is not."""
        with pytest.raises(AssertionError, match="names no part"):
            build_deployment_train_args(dataclasses.replace(args, deploy_component=DeployComponent.ALL))


class TestTheEnginesThePrimaryWaitsFor:
    def test_the_primary_waits_for_every_engine_deployment_of_the_run(self, train_args_of_identity, args):
        """The engines register from releases of their own, so a primary left to guess waits for one."""
        engines = [one for one, _ in compute_deployment_identities(args) if one is DeployComponent.INFERENCE]

        assert value_of(train_args_of_identity[(DeployComponent.PRIMARY, None)], INIT_EXPECTED_NUM_CELLS_FLAG) == str(
            args.num_engines
        )
        assert len(engines) == args.num_engines

    def test_a_deployment_carrying_its_own_engines_declares_no_number_to_wait_for(self, train_args_of_identity):
        """--init-expected-num-cells is refused for a deployment that installs the engines itself."""
        others = [
            train_args
            for (component, _), train_args in train_args_of_identity.items()
            if component is not DeployComponent.PRIMARY
        ]

        assert not [one for one in others if INIT_EXPECTED_NUM_CELLS_FLAG in one]
