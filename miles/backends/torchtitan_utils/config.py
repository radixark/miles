import importlib
import logging
import os
from argparse import Namespace

from miles.backends.torchtitan_utils import compat

compat.install()

from torchtitan.components.optimizer import ParamGroupConfig  # noqa: E402
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils.components import EmptyDataLoader, TiedCheckpointManager
from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config
from miles.utils.hf_config import load_hf_config

logger = logging.getLogger(__name__)


def _checkpoint_ties_embeddings(hf_assets_path: str) -> bool:
    return bool(getattr(load_hf_config(hf_assets_path), "tie_word_embeddings", False))


_MODEL_PACKAGE_ROOTS = ("miles.backends.torchtitan_utils.models", "torchtitan.models")


def _model_package(name: str):
    for root in _MODEL_PACKAGE_ROOTS:
        module_name = f"{root}.{name}"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise
    raise ValueError(
        f"--titan-model-name {name!r} is neither a miles model package nor a torchtitan one "
        f"(looked for {', '.join(f'{root}.{name}' for root in _MODEL_PACKAGE_ROOTS)})"
    )


def resolve_model_spec(args: Namespace):
    module = _model_package(args.titan_model_name)
    registry = getattr(module, "model_registry", None)
    if registry is None:
        raise ValueError(f"{module.__name__} exposes no model_registry(); cannot build a ModelSpec")
    return registry(args.titan_model_flavor, attn_backend="flex")


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
    if ties_embeddings and hasattr(config.model_spec.model, "enable_weight_tying"):
        config.model_spec.model.enable_weight_tying = True
        logger.info("Checkpoint ties lm_head to the embedding; excluding lm_head.weight from the HF export")

    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(args.save or "./outputs", "torchtitan", dump_subdir)

    config.parallelism.data_parallel_replicate_degree = args.titan_data_parallel_replicate_degree
    config.parallelism.tensor_parallel_degree = args.titan_tensor_parallel_degree
    config.parallelism.pipeline_parallel_degree = args.titan_pipeline_parallel_degree
    config.parallelism.context_parallel_degree = args.titan_context_parallel_degree
    config.parallelism.expert_parallel_degree = args.titan_expert_parallel_degree
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
