import gc
import os
import shutil
from argparse import ArgumentParser, Namespace
from typing import Annotated, Literal

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.training_utils.model_companion import ModelCompanionInstallationUtils
from miles.utils.args.component_trainer import TrainerOnlyConfig
from miles.utils.args.configs.custom_megatron_plugins import CustomMegatronPluginsConfig, Dsv4MegatronPluginsConfig
from miles.utils.args.configs.debug import DebugConfig
from miles.utils.args.configs.mtp_training import MtpTrainingConfig
from miles.utils.args.configs.train import TrainConfig
from miles.utils.args.custom_function import add_user_provided_function_arguments, resolve_custom_function_configs
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.args.schema import Arg, BaseConfig, reset_arg
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.memory_utils import print_memory


class _ConversionCliConfig(BaseConfig):
    hf_checkpoint: Annotated[str, Arg(required=True, help="HuggingFace model path")]


class _ConversionConfig(
    BaseLeafConfig,
    _ConversionCliConfig,
    TrainerOnlyConfig,
    TrainConfig,
    DebugConfig,
    CustomMegatronPluginsConfig,
    Dsv4MegatronPluginsConfig,
    MtpTrainingConfig,
):
    trainer_id: Literal["actor"] = "actor"
    trainer_model_id: Literal[None] = None
    trainer_role: Literal["actor"] = "actor"
    trainer_actor_index: Literal[None] = None
    multi_lora_n_adapters: Literal[0] = 0
    use_rollout_routing_replay: Literal[False] = False

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        super().add_arguments(parser=parser)
        reset_arg(parser=parser, name="--padded-vocab-size", type=int, default=None)
        return add_user_provided_function_arguments(parser, config_class=cls)


def get_args() -> _ConversionConfig:
    args = parse_args(extra_args_provider=_ConversionConfig.add_arguments)
    args.multi_lora_n_adapters = 0
    args = set_default_megatron_args(args)

    args.debug_deterministic_collective = False
    args.enable_witness = False

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = world_size

    _configure_pipeline_parallel(args, world_size=world_size)
    validate_args(args)
    return _compute_conversion_config(args)


def _compute_conversion_config(args: Namespace) -> _ConversionConfig:
    resolve_custom_function_configs(args, config_class=_ConversionConfig)
    values = vars(args)
    return _ConversionConfig.model_validate(
        {name: value for name, value in values.items() if name in _ConversionConfig.model_fields}
        | {
            "backend": MegatronArgsNamespace(
                **{name: value for name, value in values.items() if name not in _ConversionConfig.model_fields}
            )
        }
    )


def _configure_pipeline_parallel(args: Namespace, *, world_size: int) -> None:
    assert args.pipeline_model_parallel_size <= args.num_layers, (
        f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be less than or equal to "
        f"number of layers {args.num_layers}."
    )

    def ceildiv(a, b):
        return -(a // -b)

    auto_pipeline_parallel = (
        args.pipeline_model_parallel_size == 1
        and args.tensor_model_parallel_size == 1
        and args.context_parallel_size == 1
        # ETP defaults to None (= TP) until validate_args resolves it.
        and (args.expert_tensor_parallel_size or args.tensor_model_parallel_size) == 1
        # Each pipeline stage must hold whole EP groups, so auto PP only fills the ranks EP leaves.
        and world_size % args.expert_model_parallel_size == 0
        and world_size > args.expert_model_parallel_size
        and not os.environ.get("CONVERT_KEEP_PP1")
    )
    if auto_pipeline_parallel:
        pp_size = world_size // args.expert_model_parallel_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )


def main():
    import miles_plugins.mbridge  # noqa: F401
    from mbridge import AutoBridge

    configure_logger_raw()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    args = get_args()
    with args.backend.mutable():
        args.backend.rank = dist.get_rank()
    init(args)
    with args.backend.mutable():
        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    enforce_marked_param_dtypes(model)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)

    with ModelCompanionInstallationUtils.hide(model):
        bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        source_dir = get_checkpoint_name(args.backend.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.backend.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)

    dist.barrier()

    # This modification must be the *last* step and after a `dist.barrier`
    # because the higher-level scripts consider this as a signal that the script has been executed successfully
    if dist.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.backend.save)
        with open(tracker_filename, "w") as f:
            f.write("release")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
