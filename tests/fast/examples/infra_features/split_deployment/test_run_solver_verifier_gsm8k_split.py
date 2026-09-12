import dataclasses

import pytest
import yaml
from examples.infra_features.split_deployment.address_book import INIT_EXPECTED_NUM_CELLS_FLAG, RunAddressBook
from examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split import (
    build_deployment_train_args,
    compute_deployment_identities,
    compute_num_engine_cells_per_model,
)
from examples.multi_policy.run_solver_verifier_gsm8k import (
    MODEL_IDS,
    ScriptArgs,
    compute_megatron_config,
    compute_sglang_config,
    compute_trainer_id,
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
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"
NUM_POLICIES: int = 2
NUM_DEPLOYMENTS_OF_THE_RUN: int = 5
ROLLOUT_NUM_GPUS_PER_MODEL: int = 2
MEGATRON_CONFIG_FLAG: str = "--megatron-config"
SGLANG_CONFIG_FLAG: str = "--sglang-config"

_FLAGS_A_COMPONENT_MAY_DIFFER_ON: tuple[str, ...] = (
    *FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON,
    MEGATRON_CONFIG_FLAG,
    SGLANG_CONFIG_FLAG,
)


@pytest.fixture
def args() -> ScriptArgs:
    return ScriptArgs(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace=NAMESPACE,
        run_id=RUN_ID,
        run_uuid=RUN_UUID,
        rollout_num_gpus_per_model=ROLLOUT_NUM_GPUS_PER_MODEL,
    )


@pytest.fixture
def train_args_of_identity(args: ScriptArgs) -> dict[tuple[DeployComponent, str | None], str]:
    return {
        (component, instance_id): build_deployment_train_args(
            dataclasses.replace(args, deploy_component=component, deploy_instance_id=instance_id)
        )
        for component, instance_id in compute_deployment_identities(args)
    }


class TestComputeDeploymentIdentities:
    def test_the_run_is_installed_by_one_command_per_policy_and_one_for_the_script(self, args):
        """Several trainer releases in one run is the shape this example exists to show."""
        assert compute_deployment_identities(args) == [
            *[(DeployComponent.TRAINER, compute_trainer_id(model_id)) for model_id in MODEL_IDS],
            *[(DeployComponent.INFERENCE, model_id) for model_id in MODEL_IDS],
            (DeployComponent.PRIMARY, None),
        ]

    def test_the_run_is_the_two_policy_five_release_shape_every_assertion_here_reads(self, args):
        """The comparisons below are written against MODEL_IDS, and would all hold for a run of one policy."""
        assert len(MODEL_IDS) == NUM_POLICIES
        assert len(compute_deployment_identities(args)) == NUM_DEPLOYMENTS_OF_THE_RUN

    def test_the_command_that_drives_the_run_is_the_one_typed_last(self, args):
        """That command blocks until the run ends, so anything after it is typed into a finished run."""
        component, _ = compute_deployment_identities(args)[-1]

        assert component is DeployComponent.PRIMARY


class TestBuildDeploymentTrainArgs:
    def test_a_trainer_command_carries_exactly_the_trainer_it_is_named_after(self, train_args_of_identity):
        """A command handed another policy's config would train a second copy of that policy instead."""
        carried = {
            instance_id: [one["trainer_id"] for one in _config_of(train_args, MEGATRON_CONFIG_FLAG)["trainers"]]
            for (component, instance_id), train_args in train_args_of_identity.items()
            if component is DeployComponent.TRAINER
        }

        assert carried == {compute_trainer_id(model_id): [compute_trainer_id(model_id)] for model_id in MODEL_IDS}

    def test_a_trainer_command_carries_every_argument_the_whole_run_gives_that_policy(
        self, train_args_of_identity, args
    ):
        """Splitting the config per command must copy it, not rewrite it: a dropped override trains another model."""
        whole_run = {one["trainer_id"]: one for one in compute_megatron_config(args)["trainers"]}
        carried = {
            instance_id: _config_of(train_args, MEGATRON_CONFIG_FLAG)["trainers"]
            for (component, instance_id), train_args in train_args_of_identity.items()
            if component is DeployComponent.TRAINER
        }

        assert len(whole_run) == NUM_POLICIES
        assert carried == {instance_id: [whole_run[instance_id]] for instance_id in whole_run}

    def test_an_engine_command_serves_exactly_the_policy_it_is_named_after(self, train_args_of_identity):
        """An engine command that also declared its neighbour's model would install those engines twice."""
        served = {
            instance_id: [one["name"] for one in _config_of(train_args, SGLANG_CONFIG_FLAG)["sglang"]]
            for (component, instance_id), train_args in train_args_of_identity.items()
            if component is DeployComponent.INFERENCE
        }

        assert served == {model_id: [model_id] for model_id in MODEL_IDS}

    def test_the_engine_commands_together_carry_what_one_run_declares(self, train_args_of_identity, args):
        """A run short of engines still starts, and only the policy that never generates shows it."""
        declared = [
            int(value_of(train_args, ROLLOUT_NUM_GPUS_FLAG))
            for (component, _), train_args in train_args_of_identity.items()
            if component is DeployComponent.INFERENCE
        ]

        assert declared == [args.rollout_num_gpus_per_model] * len(MODEL_IDS)
        assert sum(declared) == args.rollout_num_gpus

    def test_the_driving_command_expects_every_engine_the_run_registers(self, train_args_of_identity, args):
        """It counts the cells it waits for out of its own config, so a trimmed one would start on half a fleet."""
        driver = train_args_of_identity[(DeployComponent.PRIMARY, None)]
        models = _config_of(driver, SGLANG_CONFIG_FLAG)["sglang"]

        assert [one["name"] for one in models] == MODEL_IDS
        assert value_of(driver, ROLLOUT_NUM_GPUS_FLAG) == str(args.rollout_num_gpus)

    def test_the_driving_command_names_every_trainer_of_the_run(self, train_args_of_identity):
        """The orchestration script deploys no trainer, so a trainer it cannot address is one it never drives."""
        driver = train_args_of_identity[(DeployComponent.PRIMARY, None)]
        named = [entry.partition("=")[0] for entry in values_after(driver, TRAINER_CONTROLLER_ADDRS_FLAG)]

        assert sorted(named) == sorted(compute_trainer_id(model_id) for model_id in MODEL_IDS)

    def test_the_driving_command_reaches_each_trainer_in_the_release_that_carries_it(
        self, train_args_of_identity, args
    ):
        """One address per trainer is the only thing tying the driver to releases it did not install."""
        address_book = RunAddressBook.of_config(args)
        driver = train_args_of_identity[(DeployComponent.PRIMARY, None)]
        expected = address_book.trainer_controller_addrs_arg(
            deploy_instance_id_of_trainer_id={
                compute_trainer_id(model_id): compute_trainer_id(model_id) for model_id in MODEL_IDS
            }
        )

        assert values_after(driver, TRAINER_CONTROLLER_ADDRS_FLAG) == values_after(
            expected, TRAINER_CONTROLLER_ADDRS_FLAG
        )

    def test_only_the_engine_commands_are_told_where_to_register(self, train_args_of_identity):
        """Every other command holds the controller itself and refuses to be pointed at one."""
        told = {
            component
            for (component, _), train_args in train_args_of_identity.items()
            if INFERENCE_CONTROLLER_ADDR_FLAG in train_args
        }

        assert told == {DeployComponent.INFERENCE}

    def test_only_the_driving_command_is_told_where_the_trainers_are(self, train_args_of_identity):
        """A command that carries a trainer reaches it in its own process and refuses the flag."""
        told = {
            component
            for (component, _), train_args in train_args_of_identity.items()
            if TRAINER_CONTROLLER_ADDRS_FLAG in train_args
        }

        assert told == {DeployComponent.PRIMARY}

    def test_every_command_redeems_its_references_at_one_object_store(self, train_args_of_identity):
        """Commands that disagree on the master hand each other references nothing can read back."""
        addresses = {value_of(one, MOONCAKE_INIT_KWARGS_FLAG) for one in train_args_of_identity.values()}

        assert len(addresses) == 1
        assert all(value_of(one, OBJECT_STORE_BACKEND_FLAG) == "mooncake" for one in train_args_of_identity.values())

    def test_the_commands_agree_on_everything_the_run_itself_declares(self, train_args_of_identity):
        """Only what a command carries may differ; a drifted batch shape trains something else entirely."""
        shared = [
            shared_argv(one, differing_flags=_FLAGS_A_COMPONENT_MAY_DIFFER_ON)
            for one in train_args_of_identity.values()
        ]

        assert all(one == shared[0] for one in shared)

    def test_a_command_naming_a_policy_this_run_does_not_train_is_refused(self, args):
        """A misspelled policy would install a release the run waits for under another name for good."""
        with pytest.raises(AssertionError, match="--deploy-instance-id is one of"):
            build_deployment_train_args(
                dataclasses.replace(args, deploy_component=DeployComponent.TRAINER, deploy_instance_id="solver")
            )

    def test_a_run_id_too_long_to_name_a_release_after_every_policy_is_refused(self, args):
        """helm refuses such a release name too, but only once the releases installed before it are up."""
        with pytest.raises(AssertionError, match="before helm refuses the release name"):
            build_deployment_train_args(dataclasses.replace(args, run_id="r" * 64))

    def test_a_command_naming_no_component_of_the_run_is_refused(self, args):
        """An unsplit launch installs everything under one release, which is the shape this example is not."""
        with pytest.raises(AssertionError, match="names no part"):
            build_deployment_train_args(dataclasses.replace(args, deploy_component=DeployComponent.ALL))


def _config_of(train_args: str, flag: str) -> dict:
    return yaml.safe_load(resolve_file_arg(value_of(train_args, flag)))


class TestTheEnginesThePrimaryWaitsFor:
    def test_the_primary_waits_for_every_engine_cell_a_policy_registers(self, train_args_of_identity, args):
        """A primary left to guess starts on one cell of a policy and rolls out on a fraction of its engines."""
        [model] = compute_sglang_config(args, model_ids=[MODEL_IDS[0]])["sglang"]
        [group] = model["server_groups"]
        expected = group["num_gpus"] // model["num_gpus_per_engine"]

        assert value_of(train_args_of_identity[(DeployComponent.PRIMARY, None)], INIT_EXPECTED_NUM_CELLS_FLAG) == str(
            expected
        )
        assert expected > 1

    def test_a_deployment_carrying_its_own_engines_declares_no_number_to_wait_for(self, train_args_of_identity):
        """--init-expected-num-cells is refused for a deployment that installs the engines itself."""
        others = [
            train_args
            for (component, _), train_args in train_args_of_identity.items()
            if component is not DeployComponent.PRIMARY
        ]

        assert not [one for one in others if INIT_EXPECTED_NUM_CELLS_FLAG in one]

    def test_policies_registering_different_numbers_of_cells_are_refused(self, args, monkeypatch):
        """One number gates every policy, so a run whose policies differ would start on the smaller one."""
        config = compute_sglang_config(args)
        config["sglang"][1]["server_groups"][0]["num_gpus"] += 1
        monkeypatch.setattr(
            "examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split.compute_sglang_config",
            lambda *_args, **_kwargs: config,
        )

        with pytest.raises(AssertionError, match="gates every policy"):
            compute_num_engine_cells_per_model(args)
