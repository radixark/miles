import importlib
import inspect
import logging
import os
import tempfile
from types import ModuleType

import torch.distributed as dist
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.config.transform.apply import transform_model_config_
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.base import BaseModel
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_attention import AllGatherKVFlexInnerAttention
from torchtitan.models.common.moe import MoE

from miles.backends.torchtitan_utils import lora
from miles.backends.torchtitan_utils.components import TiedCheckpointManager
from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.trainer import TitanTrainer
from miles.utils.args.runtime import TrainerConfig
from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)


_MODEL_PACKAGE_ROOTS = ("miles.backends.torchtitan_utils.models", "torchtitan.models")
_ATTN_BACKEND = "flex"


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


def build_model_config(
    args: TrainerConfig, *, parallelism: ParallelismConfig, hf_assets_path: str | None = None
) -> BaseModel.Config:
    module = _model_package(args.backend.titan_model_name)
    builder = getattr(
        module, "build_model_config", None
    )  # config-access-exempt: imported TorchTitan model packages may omit the config builder entry point
    if builder is None:
        raise ValueError(f"{module.__name__} exposes no build_model_config(); cannot build its model")
    kwargs = dict(parallelism=parallelism, seq_len=args.backend.seq_length, attn_backend=_ATTN_BACKEND)
    if "hf_config" in inspect.signature(builder).parameters:
        if hf_assets_path is None:
            raise ValueError(f"{module.__name__} sizes its model from the HF config; pass hf_assets_path")
        kwargs["hf_config"] = load_hf_config(hf_assets_path)
    model = builder(args.backend.titan_model_flavor, **kwargs)
    _disable_load_balancing(model)
    if parallelism.context_parallel_degree > 1:
        model = transform_model_config_(
            model, [ContextParallelTransform(inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention})]
        )
    return model


def _disable_load_balancing(model: BaseModel.Config) -> None:
    for _, moe, _, _ in model.traverse(MoE.Config):
        moe.router.aux_loss = None
        moe.freeze_expert_bias = True


def _latest_step_dir(load_root: str, dump_subdir: str) -> str | None:
    folder = os.path.join(load_root, "torchtitan", dump_subdir, "checkpoint")
    if not os.path.isdir(folder):
        return None
    steps = [int(name.removeprefix("step-")) for name in os.listdir(folder) if name.startswith("step-")]
    return os.path.join(folder, f"step-{max(steps)}") if steps else None


def _parallelism_config(args: TrainerConfig) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=args.backend.dp_replicate_size,
        tensor_parallel_degree=args.backend.tensor_model_parallel_size,
        pipeline_parallel_degree=args.backend.pipeline_model_parallel_size,
        context_parallel_degree=args.backend.context_parallel_size,
        expert_parallel_degree=args.backend.expert_model_parallel_size,
        num_pp_microbatches=_num_pipeline_microbatches(args),
    )


def _num_pipeline_microbatches(args: TrainerConfig) -> int:
    model_parallel_size = (
        args.backend.tensor_model_parallel_size
        * args.backend.pipeline_model_parallel_size
        * args.backend.context_parallel_size
    )
    dp_size = dist.get_world_size() // model_parallel_size
    if args.backend.pipeline_model_parallel_size > 1 and args.backend.global_batch_size % (
        dp_size * args.backend.micro_batch_size
    ):
        raise ValueError(
            f"--global-batch-size {args.backend.global_batch_size} must be a multiple of dp * micro_batch_size "
            f"({dp_size} * {args.backend.micro_batch_size}) under pipeline parallelism"
        )
    return max(args.backend.global_batch_size // dp_size // args.backend.micro_batch_size, 1)


def _optim_config(args: TrainerConfig) -> Optim.Config:
    if args.backend.lr_decay_style == "constant":
        lr_scheduler = LRSchedulersContainer.Config(warmup_steps=args.backend.lr_warmup_iters, min_lr_factor=1.0)
    else:
        lr_scheduler = LRSchedulersContainer.Config(
            warmup_steps=args.backend.lr_warmup_iters,
            decay_type=args.backend.lr_decay_style,
            min_lr_factor=args.backend.min_lr / args.backend.lr,
        )
    adamw = AdamW.Config(
        pattern=r".*",
        lr=args.backend.lr,
        betas=(args.backend.adam_beta1, args.backend.adam_beta2),
        eps=args.backend.adam_eps,
        weight_decay=args.backend.weight_decay,
    )
    return Optim.Config(
        optimizer=OptimizersContainer.Config(optimizers=[adamw]),
        lr_scheduler=lr_scheduler,
        max_norm=args.backend.clip_grad,
    )


def _checkpointer_config(args: TrainerConfig, *, dump_subdir: str) -> TiedCheckpointManager.Config:
    resume_from = _latest_step_dir(args.backend.load, dump_subdir) if args.backend.load else None
    if resume_from is None:
        return TiedCheckpointManager.Config(
            last_save_model_only=False, initial_load_model_only=True, initial_load_in_hf=True
        )
    return TiedCheckpointManager.Config(
        last_save_model_only=False,
        initial_load_path=resume_from,
        initial_load_model_only=False,
        initial_load_in_hf=False,
    )


def build_trainer_config(
    args: TrainerConfig,
    *,
    hf_assets_path: str,
    lr_total_steps: int,
    dump_subdir: str,
    with_lora: bool = False,
) -> TitanTrainer.Config:
    if args.backend.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.backend.optimizer!r}")

    ties_embeddings = load_hf_config(hf_assets_path).tie_word_embeddings
    if ties_embeddings and args.backend.pipeline_model_parallel_size > 1:
        raise ValueError(
            "the checkpoint ties lm_head to the embedding, which torchtitan cannot do across pipeline "
            "stages: it would train a separate lm_head that the HF export then has no tensor to ship "
            "into. Use --pipeline-model-parallel-size 1 or an untied checkpoint."
        )

    parallelism = _parallelism_config(args)
    model = build_model_config(args, parallelism=parallelism, hf_assets_path=hf_assets_path)
    model.enable_weight_tying = ties_embeddings
    if ties_embeddings:
        logger.info("Checkpoint ties lm_head to the embedding; excluding lm_head.weight from the HF export")
    if with_lora:
        model = lora.convert_model_config(
            model, hf_targets=args.hf_lora_targets, rank=args.lora_rank, alpha=args.lora_alpha
        )

    return TitanTrainer.Config(
        model=model,
        hf_assets_path=hf_assets_path,
        output_dir=os.path.join(
            args.backend.save or tempfile.mkdtemp(prefix="miles-torchtitan-"), "torchtitan", dump_subdir
        ),
        parallelism=parallelism,
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=args.backend.seq_length,
            max_context_length=args.backend.seq_length,
            steps=max(lr_total_steps, 1),
            enable_cpu_offload=args.backend.optimizer_cpu_offload,
            disable_cuda_graphs=True,
        ),
        optim=_optim_config(args),
        checkpointer=_checkpointer_config(args, dump_subdir=dump_subdir),
        activation_remat=FullActivationRematConfig() if args.backend.gradient_checkpointing else None,
        loss=RLLossAdapter.Config(),
        debug=DebugConfig(seed=args.backend.seed),
    )
