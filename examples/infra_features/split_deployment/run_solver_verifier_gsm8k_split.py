import typer
from examples.infra_features.split_deployment.address_book import RunAddressBook, init_expected_num_cells_arg
from examples.multi_policy.run_solver_verifier_gsm8k import (
    MODEL_IDS,
    ScriptArgs,
    build_train_args,
    compute_megatron_config,
    compute_sglang_config,
    compute_trainer_id,
    launch_train,
    prepare,
)

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH
from miles.utils.workers.types import DeployComponent


def build_deployment_train_args(args: ScriptArgs) -> str:
    _assert_the_run_id_leaves_room_for_one_release_per_policy(args.run_id)
    address_book = RunAddressBook.of_config(args)
    wandb_args = command_utils.get_default_wandb_args(__file__, run_id=args.run_id)
    shared = address_book.shared_object_store_args()

    match args.deploy_component:
        case DeployComponent.TRAINER:
            model_id = _model_id_of_trainer_deployment(args.deploy_instance_id)
            return (
                build_train_args(
                    args, wandb_args=wandb_args, megatron_config=compute_megatron_config(args, model_ids=[model_id])
                )
                + shared
            )
        case DeployComponent.INFERENCE:
            model_id = _model_id_of_engine_deployment(args.deploy_instance_id)
            return (
                build_train_args(
                    args,
                    wandb_args=wandb_args,
                    sglang_config=compute_sglang_config(args, model_ids=[model_id]),
                    rollout_num_gpus=args.rollout_num_gpus_per_model,
                )
                + shared
                + address_book.inference_controller_addr_arg()
            )
        case DeployComponent.PRIMARY:
            return (
                build_train_args(args, wandb_args=wandb_args)
                + shared
                + address_book.trainer_controller_addrs_arg(
                    deploy_instance_id_of_trainer_id={one: one for one in map(compute_trainer_id, MODEL_IDS)}
                )
                + init_expected_num_cells_arg(compute_num_engine_cells_per_model(args))
            )
        case _:
            raise AssertionError(
                f"one command of this example installs one part of the run, and --deploy-component "
                f"{args.deploy_component.value} names no part; launch the parts "
                f"{[one.value for one, _ in compute_deployment_identities(args)]} one command at a time"
            )


def compute_num_engine_cells_per_model(args: ScriptArgs) -> int:
    counts = {
        sum(group["num_gpus"] // model["num_gpus_per_engine"] for group in model["server_groups"])
        for model in compute_sglang_config(args)["sglang"]
    }

    assert len(counts) == 1, (
        f"one --init-expected-num-cells gates every policy of the run, and these deployments register {counts} "
        f"cell(s) per policy"
    )
    return counts.pop()


def compute_deployment_identities(args: ScriptArgs) -> list[tuple[DeployComponent, str | None]]:
    return [
        *[(DeployComponent.TRAINER, compute_trainer_id(model_id)) for model_id in MODEL_IDS],
        *[(DeployComponent.INFERENCE, model_id) for model_id in MODEL_IDS],
        (DeployComponent.PRIMARY, None),
    ]


def _model_id_of_trainer_deployment(deploy_instance_id: str | None) -> str:
    model_id_of_trainer_id = {compute_trainer_id(model_id): model_id for model_id in MODEL_IDS}

    assert deploy_instance_id in model_id_of_trainer_id, (
        f"a trainer deployment is named after the one policy it trains, so its --deploy-instance-id is one of "
        f"{sorted(model_id_of_trainer_id)}, not {deploy_instance_id!r}"
    )
    return model_id_of_trainer_id[deploy_instance_id]


def _model_id_of_engine_deployment(deploy_instance_id: str | None) -> str:
    assert deploy_instance_id in MODEL_IDS, (
        f"an engine deployment serves the one policy it is named after, so its --deploy-instance-id is one of "
        f"{MODEL_IDS}, not {deploy_instance_id!r}"
    )
    return deploy_instance_id


def _assert_the_run_id_leaves_room_for_one_release_per_policy(run_id: str) -> None:
    longest = max(len(one) for one in [*MODEL_IDS, *map(compute_trainer_id, MODEL_IDS)])
    budget = RUN_ID_MAX_LENGTH - (longest + 1)

    assert len(run_id) <= budget, (
        f"run id {run_id!r} is {len(run_id)} characters, and this example suffixes a release with the longest "
        f"policy name ({longest} characters), leaving at most {budget} before helm refuses the release name"
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    train_args = build_deployment_train_args(args)
    prepare(args)
    launch_train(train_args, args)


# TODO: unify this launcher when the example scripts are refactored
if __name__ == "__main__":
    typer.run(main)
