from dataclasses import dataclass

import typer
from examples.infra_features.split_deployment.address_book import (
    DEFAULT_TRAINER_ID,
    RunAddressBook,
    init_expected_num_cells_arg,
)

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import DeployComponent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    model_name: str = "Qwen3-0.6B"
    megatron_model_type: str = "qwen3-0.6B"
    num_rollout: int = 3
    actor_num_gpus: int = 4
    num_engines: int = 2
    gpus_per_engine: int = 1
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    @property
    def num_gpus_per_node(self) -> int:
        return self.actor_num_gpus + self.num_engines * self.gpus_per_engine


def prepare(args: ScriptArgs) -> None:
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.actor_num_gpus,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def build_deployment_train_args(args: ScriptArgs) -> str:
    address_book = RunAddressBook.of_config(args)
    shared = address_book.shared_object_store_args()
    whole_run_rollout_num_gpus = args.num_engines * args.gpus_per_engine

    match args.deploy_component:
        case DeployComponent.TRAINER:
            return build_train_args(args, rollout_num_gpus=whole_run_rollout_num_gpus) + shared
        case DeployComponent.INFERENCE:
            return (
                build_train_args(args, rollout_num_gpus=args.gpus_per_engine)
                + shared
                + address_book.inference_controller_addr_arg()
            )
        case DeployComponent.PRIMARY:
            return (
                build_train_args(args, rollout_num_gpus=whole_run_rollout_num_gpus)
                + shared
                + address_book.trainer_controller_addrs_arg(
                    deploy_instance_id_of_trainer_id={DEFAULT_TRAINER_ID: None}
                )
                + init_expected_num_cells_arg(args.num_engines)
            )
        case _:
            raise AssertionError(
                f"one command of this example installs one part of the run, and --deploy-component "
                f"{args.deploy_component.value} names no part; launch "
                f"{[one.value for one, _ in compute_deployment_identities(args)]} one command at a time"
            )


def compute_deployment_identities(args: ScriptArgs) -> list[tuple[DeployComponent, str | None]]:
    return [
        (DeployComponent.TRAINER, None),
        *[(DeployComponent.INFERENCE, f"e{index}") for index in range(args.num_engines)],
        (DeployComponent.PRIMARY, None),
    ]


def build_train_args(args: ScriptArgs, *, rollout_num_gpus: int) -> str:
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deterministic_random "
        "--rollout-max-response-len 200 "
        "--rollout-temperature 0.8 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        f"--rollout-num-gpus-per-engine {args.gpus_per_engine} "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--accumulate-allreduce-grads-in-fp32 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--context-parallel-size 2 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus} "
        "--global-batch-size 256 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--advantage-estimator grpo "
        "--eps-clip 0.2 "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{misc_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args} "
    )


def launch_train(train_args: str, args: ScriptArgs) -> None:
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    train_args = build_deployment_train_args(args)
    prepare(args)
    launch_train(train_args, args)


# TODO: unify this launcher when the example scripts are refactored
if __name__ == "__main__":
    typer.run(main)
