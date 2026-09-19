import importlib
import logging
import os
import tempfile
from argparse import Namespace

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils.components import EmptyDataLoader, TiedCheckpointManager
from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config
from miles.utils.hf_config import load_hf_config

logger = logging.getLogger(__name__)


def resolve_model_spec(args: Namespace):
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
    return registry(args.titan_model_flavor, attn_backend="flex")


def _latest_step_dir(load_root: str, dump_subdir: str) -> str | None:
    folder = os.path.join(load_root, "torchtitan", dump_subdir, "checkpoint")
    if not os.path.isdir(folder):
        return None
    steps = [int(name.removeprefix("step-")) for name in os.listdir(folder) if name.startswith("step-")]
    return os.path.join(folder, f"step-{max(steps)}") if steps else None


def build_trainer_config(args: Namespace, *, hf_assets_path: str, lr_total_steps: int, dump_subdir: str):
    if args.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

    ties_embeddings = load_hf_config(hf_assets_path).tie_word_embeddings
    if ties_embeddings and args.titan_pipeline_parallel_degree > 1:
        raise ValueError(
            "the checkpoint ties lm_head to the embedding, which torchtitan cannot do across pipeline "
            "stages: it would train a separate lm_head that the HF export then has no tensor to ship "
            "into. Use --titan-pipeline-parallel-degree 1 or an untied checkpoint."
        )

    config = Trainer.Config()
    config.model_spec = resolve_model_spec(args)
    if ties_embeddings:
        config.model_spec.model.enable_weight_tying = True
        logger.info("Checkpoint ties lm_head to the embedding; excluding lm_head.weight from the HF export")

    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(
        args.save or tempfile.mkdtemp(prefix="miles-torchtitan-"), "torchtitan", dump_subdir
    )

    config.parallelism.data_parallel_replicate_degree = args.titan_data_parallel_replicate_degree
    config.parallelism.tensor_parallel_degree = args.titan_tensor_parallel_degree
    config.parallelism.pipeline_parallel_degree = args.titan_pipeline_parallel_degree
    config.parallelism.context_parallel_degree = args.titan_context_parallel_degree
    config.parallelism.expert_parallel_degree = args.titan_expert_parallel_degree
    config.parallelism.pipeline_parallel_microbatch_size = 1
    parallel_dims = parallel_dims_from_config(config.parallelism)
    dp_size = parallel_dims.dp_replicate * parallel_dims.dp_shard

    config.training.seq_len = args.titan_seq_len
    if parallel_dims.pp_enabled and args.global_batch_size % (dp_size * args.micro_batch_size):
        raise ValueError(
            f"--global-batch-size {args.global_batch_size} must be a multiple of dp * micro_batch_size "
            f"({dp_size} * {args.micro_batch_size}) under pipeline parallelism"
        )
    config.training.local_batch_size = max(args.global_batch_size // dp_size // args.micro_batch_size, 1)
    config.training.global_batch_size = config.training.local_batch_size * dp_size
    config.training.steps = max(lr_total_steps, 1)
    config.training.max_norm = args.clip_grad
    config.training.disable_cuda_graphs = True

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

    config.lr_scheduler.warmup_steps = args.lr_warmup_iters
    if args.lr_decay_style == "constant":
        config.lr_scheduler.min_lr_factor = 1.0
    else:
        config.lr_scheduler.decay_type = args.lr_decay_style
        config.lr_scheduler.min_lr_factor = args.min_lr / args.lr

    config.loss = RLLossAdapter.Config()
    config.dataloader = EmptyDataLoader.Config()
    config.checkpoint = TiedCheckpointManager.Config()
    config.activation_checkpoint = FullAC.Config() if args.gradient_checkpointing else None
    config.debug.seed = args.seed

    config.checkpoint.enable = True
    config.checkpoint.last_save_model_only = False
    resume_from = _latest_step_dir(args.load, dump_subdir) if args.load else None
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
