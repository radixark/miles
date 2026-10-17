import importlib
import logging
import os
import tempfile
from types import ModuleType

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils.components import EmptyDataLoader, TiedCheckpointManager
from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config
from miles.utils.args.runtime import TrainerConfig
from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)


_MODEL_PACKAGE_ROOTS = ("miles.backends.torchtitan_utils.models", "torchtitan.models")


def _model_package(name: str) -> ModuleType:
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


def resolve_model_spec(args: TrainerConfig) -> ModelSpec:
    module = _model_package(args.backend.titan_model_name)
    registry = getattr(
        module, "model_registry", None
    )  # config-access-exempt: imported TorchTitan model packages may omit the registry entry point
    if registry is None:
        raise ValueError(f"{module.__name__} exposes no model_registry(); cannot build a ModelSpec")
    return registry(args.backend.titan_model_flavor, attn_backend="flex")


def _latest_step_dir(load_root: str, dump_subdir: str) -> str | None:
    folder = os.path.join(load_root, "torchtitan", dump_subdir, "checkpoint")
    if not os.path.isdir(folder):
        return None
    steps = [int(name.removeprefix("step-")) for name in os.listdir(folder) if name.startswith("step-")]
    return os.path.join(folder, f"step-{max(steps)}") if steps else None


def build_trainer_config(
    args: TrainerConfig, *, hf_assets_path: str, lr_total_steps: int, dump_subdir: str
) -> Trainer.Config:
    if args.backend.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.backend.optimizer!r}")

    ties_embeddings = load_hf_config(hf_assets_path).tie_word_embeddings
    if ties_embeddings and args.backend.pipeline_model_parallel_size > 1:
        raise ValueError(
            "the checkpoint ties lm_head to the embedding, which torchtitan cannot do across pipeline "
            "stages: it would train a separate lm_head that the HF export then has no tensor to ship "
            "into. Use --pipeline-model-parallel-size 1 or an untied checkpoint."
        )

    config = Trainer.Config()
    config.model_spec = resolve_model_spec(args)
    if ties_embeddings:
        config.model_spec.model.enable_weight_tying = True
        logger.info("Checkpoint ties lm_head to the embedding; excluding lm_head.weight from the HF export")

    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(
        args.backend.save or tempfile.mkdtemp(prefix="miles-torchtitan-"), "torchtitan", dump_subdir
    )

    config.parallelism.data_parallel_replicate_degree = args.backend.dp_replicate_size
    config.parallelism.tensor_parallel_degree = args.backend.tensor_model_parallel_size
    config.parallelism.pipeline_parallel_degree = args.backend.pipeline_model_parallel_size
    config.parallelism.context_parallel_degree = args.backend.context_parallel_size
    config.parallelism.expert_parallel_degree = args.backend.expert_model_parallel_size
    config.parallelism.pipeline_parallel_microbatch_size = 1
    config.parallelism.spmd_backend = "partial_dtensor"
    parallel_dims = parallel_dims_from_config(config.parallelism)
    dp_size = parallel_dims.dp_replicate * parallel_dims.dp_shard

    config.training.seq_len = args.backend.seq_length
    if parallel_dims.pp_enabled and args.backend.global_batch_size % (dp_size * args.backend.micro_batch_size):
        raise ValueError(
            f"--global-batch-size {args.backend.global_batch_size} must be a multiple of dp * micro_batch_size "
            f"({dp_size} * {args.backend.micro_batch_size}) under pipeline parallelism"
        )
    config.training.local_batch_size = max(
        args.backend.global_batch_size // dp_size // args.backend.micro_batch_size, 1
    )
    config.training.global_batch_size = config.training.local_batch_size * dp_size
    config.training.steps = max(lr_total_steps, 1)
    config.training.max_norm = args.backend.clip_grad
    config.training.disable_cuda_graphs = True

    config.optimizer.param_groups = [
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={
                "lr": args.backend.lr,
                "betas": (args.backend.adam_beta1, args.backend.adam_beta2),
                "eps": args.backend.adam_eps,
                "weight_decay": args.backend.weight_decay,
            },
        )
    ]

    config.lr_scheduler.warmup_steps = args.backend.lr_warmup_iters
    if args.backend.lr_decay_style == "constant":
        config.lr_scheduler.min_lr_factor = 1.0
    else:
        config.lr_scheduler.decay_type = args.backend.lr_decay_style
        config.lr_scheduler.min_lr_factor = args.backend.min_lr / args.backend.lr

    config.loss = RLLossAdapter.Config()
    config.dataloader = EmptyDataLoader.Config()
    config.checkpoint = TiedCheckpointManager.Config()
    config.activation_checkpoint = FullAC.Config() if args.backend.gradient_checkpointing else None
    config.debug.seed = args.backend.seed

    config.checkpoint.enable = True
    config.checkpoint.last_save_model_only = False
    resume_from = _latest_step_dir(args.backend.load, dump_subdir) if args.backend.load else None
    if resume_from is None:
        config.checkpoint.initial_load_model_only = True
        config.checkpoint.initial_load_in_hf = True
    else:
        config.checkpoint.initial_load_path = resume_from
        config.checkpoint.initial_load_model_only = False
        config.checkpoint.initial_load_in_hf = False

    config.metrics.enable_tensorboard = False
    config.metrics.enable_wandb = False
    config.validator.enable = False
    return config
