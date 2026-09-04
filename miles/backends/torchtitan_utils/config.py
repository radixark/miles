"""One ``Trainer.Config`` tree from miles arguments.

The config tree is the program: torchtitan's ``Trainer.__init__`` builds
everything from it. miles fills the tree and hands it over; nothing here
constructs a torchtitan object directly.
"""

import importlib
import json
import logging
import os
from argparse import Namespace

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils.components import EmptyDataLoader, TiedCheckpointManager
from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config

logger = logging.getLogger(__name__)


def _checkpoint_ties_embeddings(hf_assets_path: str) -> bool:
    config_path = os.path.join(hf_assets_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        return bool(json.load(f).get("tie_word_embeddings", False))


def resolve_model_spec(args: Namespace):
    """``torchtitan.models.<name>.model_registry(<flavor>)``."""
    module_name = f"torchtitan.models.{args.titan_model_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"--titan-model-name {args.titan_model_name!r} does not resolve to a torchtitan "
            f"model package ({module_name}). Check the pinned torchtitan checkout."
        ) from e
    registry = getattr(module, "model_registry", None)
    if registry is None:
        raise ValueError(f"{module_name} exposes no model_registry(); cannot build a ModelSpec")
    return registry(args.titan_model_flavor, attn_backend=args.titan_attn_backend)


def build_trainer_config(args: Namespace, *, hf_assets_path: str, lr_total_steps: int, dump_subdir: str):
    if args.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

    ties_embeddings = _checkpoint_ties_embeddings(hf_assets_path)
    if ties_embeddings and args.titan_pipeline_parallel_degree > 1:
        raise ValueError(
            "the checkpoint ties lm_head to the embedding, which torchtitan cannot do across pipeline "
            "stages: it would train a separate lm_head that the HF export then has no tensor to ship "
            "into. Use --titan-pipeline-parallel-degree 1 or an untied checkpoint."
        )

    config = Trainer.Config()
    config.model_spec = resolve_model_spec(args)
    if args.titan_num_layers:
        available = len(config.model_spec.model.layers)
        if args.titan_num_layers > available:
            raise ValueError(
                f"--titan-num-layers {args.titan_num_layers} exceeds the "
                f"{args.titan_model_flavor} flavor's {available} blocks"
            )
        config.model_spec.model.layers = config.model_spec.model.layers[: args.titan_num_layers]
        logger.info(f"Truncated {args.titan_model_flavor} to {args.titan_num_layers} of {available} blocks")
    if ties_embeddings and hasattr(config.model_spec.model, "enable_weight_tying"):
        config.model_spec.model.enable_weight_tying = True
        logger.info("Checkpoint ties lm_head to the embedding; excluding lm_head.weight from the HF export")

    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(args.save or "./outputs", "torchtitan", dump_subdir)

    config.parallelism.data_parallel_replicate_degree = args.titan_data_parallel_replicate_degree
    config.parallelism.data_parallel_shard_degree = args.titan_data_parallel_shard_degree
    config.parallelism.tensor_parallel_degree = args.titan_tensor_parallel_degree
    config.parallelism.pipeline_parallel_degree = args.titan_pipeline_parallel_degree
    config.parallelism.context_parallel_degree = args.titan_context_parallel_degree
    config.parallelism.expert_parallel_degree = args.titan_expert_parallel_degree
    if args.titan_pipeline_parallel_schedule:
        config.parallelism.pipeline_parallel_schedule = args.titan_pipeline_parallel_schedule
    config.parallelism.pipeline_parallel_microbatch_size = 1
    parallel_dims = parallel_dims_from_config(config.parallelism)
    dp_size = parallel_dims.dp_replicate * parallel_dims.dp_shard

    config.training.seq_len = args.titan_seq_len
    config.training.local_batch_size = max(args.global_batch_size // dp_size // args.micro_batch_size, 1)
    config.training.global_batch_size = config.training.local_batch_size * dp_size
    config.training.steps = max(lr_total_steps, 1)
    config.training.max_norm = args.clip_grad
    config.training.disable_cuda_graphs = True
    if args.fp16:
        config.training.dtype = "float16"

    config.optimizer.param_groups = [
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={
                "lr": args.lr,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            },
        )
    ]

    config.loss = RLLossAdapter.Config()
    config.dataloader = EmptyDataLoader.Config()
    config.checkpoint = TiedCheckpointManager.Config()
    config.activation_checkpoint = FullAC.Config() if getattr(args, "gradient_checkpointing", False) else None
    config.debug.seed = args.seed

    config.checkpoint.enable = True
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.initial_load_in_hf = True

    config.metrics.enable_tensorboard = False
    config.metrics.enable_wandb = False
    config.validator.enable = False
    return config
