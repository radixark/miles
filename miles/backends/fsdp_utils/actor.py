import logging
import os
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import torch.distributed as dist

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.adaptations import routing_replay
from miles.backends.training_utils.model_assets import load_model_assets
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native_actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native_loop import LinearStepRunner, StepMetrics, clip_and_report
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils import train_dump_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.profile_utils import TrainProfiler
from miles.utils.replay_base import routing_replay_manager
from miles.utils.timer import Timer
from miles.utils.tracking_utils.tracking import init_tracking

from .adaptations.class_patches import apply_class_patches, apply_model_instance_patches
from .adaptations.packing import apply_packing
from .adaptations.post_load_fixups import apply_post_load_fixups
from .adaptations.precision import apply_fp32_master, precision_forward_context, resolve_precision_policy
from .hf_weight_iterator import get_hf_weight_iterator
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TorchNativeTrainRayActor):
    routing_replay = routing_replay

    """Simplified TrainRayActor for pure HF+FSDP training.

    Initializes the stock HF model on rank0 (others on meta), wraps it in FSDP2, and provides the
    train / save / update_weights hooks. Weight sync: rank0 gathers the full state_dict and broadcasts
    tensor-by-tensor.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
    ) -> int | None:  # type: ignore[override]
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        # Unsupported
        assert recv_ckpt_src_rank is None
        assert indep_dp_info.quorum_id == 0

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        # Setup ParallelState for both CP and non-CP cases
        set_parallel_state(create_fsdp_parallel_state(args))

        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        assets = load_model_assets(self.args, with_processor=True)
        self.hf_config = assets.hf_config
        self.tokenizer = assets.tokenizer
        if assets.processor is not None:
            self.processor = assets.processor

        self.precision_policy = resolve_precision_policy(self.hf_config, self.args)
        self._init_flops(self.hf_config)

        routing_replay.enable(args)

        # FSDP trains stock HF modeling: HF-compat patches + config-lifetime packing, before construction.
        apply_class_patches(self.hf_config, self.args)
        apply_packing(None, self.hf_config, "config")

        # backend-level true-on-policy setup (batch-invariant ops)
        self._enable_true_on_policy_optimizations(args)

        init_context = self._get_init_weight_context_manager()

        model, n = self._build_model_with_attn_bridge(self.args.hf_checkpoint, init_context)
        if n > 0:
            logger.info(f"FSDPTrainRayActor applied triton attention patch to {n} layer(s)")

        apply_model_instance_patches(model, self.hf_config, self.args)
        routing_replay.install(model, self.hf_config)
        if self.precision_policy.keep_fp32_master:
            model = apply_fp32_master(model, self.precision_policy.sync_dtype_resolver)

        # re-assert the checkpoint over any param from_pretrained clobbered post-load (arch-gated, else no-op)
        apply_post_load_fixups(model, self.hf_config, self.args.hf_checkpoint)

        # post-load packing patches that need the instantiated model (NemotronH); no-op for archs that don't
        apply_packing(model, self.hf_config, "post_load")

        model.train()

        full_state = model.state_dict()

        model = apply_fsdp2(
            model,
            mesh=get_parallel_state().get_mesh("fsdp"),
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            param_dtype=self.precision_policy.param_dtype,
            reduce_dtype=self.precision_policy.reduce_dtype,
        )

        model = self._fsdp2_load_full_state_dict(
            model,
            full_state,
            get_parallel_state().get_mesh("fsdp"),
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = model

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)
            self.ref_runner = LinearStepRunner(partial(self._logprob_forward, self.ref_model))
        self.model_parts = [self.model]
        self.optimizers = [self.optimizer]

        self.weight_updater = WeightUpdater(
            self.args,
            self.model,
            weights_getter=lambda: None,
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            iterator_factory=get_hf_weight_iterator,
            parallel_state=get_parallel_state(),
            is_lora=False,
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        self.max_tokens_per_gpu = args.max_tokens_per_gpu

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _has_image_text_to_text_impl(self) -> bool:
        if not hasattr(self.hf_config, "vision_config"):
            return False
        # A remote-code checkpoint only implements the Auto classes its own auto_map declares, and a
        # multimodal config does not imply AutoModelForImageTextToText is one of them: Kimi-K2.5 ships
        # a vision_config but maps only AutoModelForCausalLM. Native archs carry no auto_map and keep
        # resolving through the transformers registry.
        auto_map = getattr(self.hf_config, "auto_map", None)
        return not auto_map or "AutoModelForImageTextToText" in auto_map

    def get_model_cls(self):
        if self._has_image_text_to_text_impl():
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            import transformers
            from transformers import AutoModelForCausalLM
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

            # Resolve natively-supported archs by model_type string: AutoConfig/AutoModel registries can
            # be re-registered at runtime (sglang vendors a nemotron_h config whose hybrid_override_pattern
            # parsing mis-places the attention layers), which would silently train a mis-shaped model.
            native_cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(self.hf_config, "model_type", ""))
            if native_cls_name is not None:
                return getattr(transformers, native_cls_name)
            return AutoModelForCausalLM

    def _build_model_with_attn_bridge(self, checkpoint_path: str, init_context):
        """Build HF model and optionally apply Triton attention bridge patch."""
        # ROCm-only: on other platforms "triton" falls through to from_pretrained, which rejects
        # it exactly as it did before this path existed.
        use_triton_bridge = self.args.attn_implementation == "triton" and torch.version.hip is not None
        effective_attn = "eager" if use_triton_bridge else self.args.attn_implementation

        with init_context():
            model = self.get_model_cls().from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                attn_implementation=effective_attn,
            )

        patched_layers = 0
        if use_triton_bridge:
            from .sglang_attn_bridge.hf_sglang_triton_patch import apply_sglang_triton_attention_patch

            patched_layers = apply_sglang_triton_attention_patch(model)
        return model, patched_layers

    def _enable_true_on_policy_optimizations(self, args):
        """Backend-level true-on-policy setup (batch-invariant ops), gated on the run mode."""
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

    def _get_init_weight_context_manager(self):
        """Context manager for model init: meta device (no allocation) on non-rank-0, EXCEPT when
        tie_word_embeddings=True (meta tensors hang there) -- then full CPU load on all ranks.

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        """
        from accelerate import init_empty_weights

        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load the full state dict into the FSDP2 model, broadcasting rank-0 weights to all ranks
        (so only rank 0 reads from disk).

        Ref: verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        if self.args.debug_rollout_only or self.args.save is None:
            return

        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

    @contextmanager
    def _active_model(self, model_tag: str):
        """Yield the model that owns this pass.

        The reference model is a separate FSDP2 module, so both cannot be
        resident at once unless FSDP is already offloading: park the actor on the
        host for the duration of the reference pass and bring it back after. The
        barriers keep the ranks from racing each other's device moves.
        """
        if model_tag != "ref" or self.ref_model is None:
            yield self.model
            return

        if not self.fsdp_cpu_offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            dist.barrier(group=get_gloo_group())
        self.ref_model.eval()
        try:
            yield self.ref_model
        finally:
            torch.cuda.empty_cache()
            dist.barrier(group=get_gloo_group())
            if not self.fsdp_cpu_offload:
                self.model.cuda()
                dist.barrier(group=get_gloo_group())

    def _logprob_forward(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        """No-grad forward. Logits stay in native bf16; the loss path upcasts
        per-response chunks, which avoids a full-vocab fp32 tensor."""
        model_args = self._get_model_inputs_args(batch)
        with precision_forward_context(self.precision_policy):
            return model(**model_args).logits

    def step_runner(self) -> LinearStepRunner:
        return LinearStepRunner(self._forward, self._zero_grad, self._apply_step)

    def ref_context(self):
        return self._active_model("ref")

    def after_rollout(self, rollout_id: int, rollout_data) -> None:
        if self.args.save_debug_train_data is not None:
            train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _forward(self, batch: dict) -> torch.Tensor:
        """The training pass brackets each real forward in ``replay_forward`` so
        activation-checkpoint recompute keeps the backward cursor to itself."""
        model_args = self._get_model_inputs_args(batch)
        replaying = routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD
        replay_stage = routing_replay.stage(routing_replay.REPLAY_FORWARD) if replaying else nullcontext()
        with replay_stage, precision_forward_context(self.precision_policy):
            return self.model(**model_args).logits

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _apply_step(self) -> StepMetrics:
        grad_norm = clip_and_report(self.model.parameters(), self.args.clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()
        return StepMetrics(
            grad_norm=grad_norm,
            extra_metrics={f"lr-pg_{i}": group["lr"] for i, group in enumerate(self.optimizer.param_groups)},
        )

    def _create_ref_model(self, ref_load_path: str | None):
        """Create a separate FSDP2 ref model. ALWAYS uses CPUOffloadPolicy (regardless of the actor's
        offload setting) to save memory. Raises if ``ref_load_path`` is None or not a directory."""
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            ref_model, ref_patch_n = self._build_model_with_attn_bridge(ref_load_path, init_context)
            if ref_patch_n > 0:
                logger.info(
                    f"[Rank {dist.get_rank()}] Applied triton attention patch to ref model ({ref_patch_n} layer(s))"
                )

            apply_model_instance_patches(ref_model, self.hf_config, self.args)
            if self.precision_policy.keep_fp32_master and self.precision_policy.param_dtype is torch.float32:
                ref_model = apply_fp32_master(ref_model, self.precision_policy.sync_dtype_resolver)
            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(
                ref_model,
                mesh=get_parallel_state().get_mesh("fsdp"),
                cpu_offload=True,
                args=self.args,
                param_dtype=self.precision_policy.param_dtype,
                reduce_dtype=self.precision_policy.reduce_dtype,
            )
            ref_model = self._fsdp2_load_full_state_dict(
                ref_model,
                full_state,
                get_parallel_state().get_mesh("fsdp"),
                cpu_offload=True,
            )

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(self, batch: dict) -> dict:
        model_args = {
            "input_ids": batch["tokens"],
            "position_ids": batch["position_ids"],
            "attention_mask": None,
        }

        if batch.get("multimodal_train_inputs"):
            model_args.update(batch["multimodal_train_inputs"])

        return model_args


@torch.no_grad()
def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None, param_dtype=None, reduce_dtype=None):
    """Apply FSDP2 (fully_shard) to the model.

    ``cpu_offload`` offloads params/grads/optimizer to CPU (the optimizer step runs on CPU).
    ``param_dtype``/``reduce_dtype`` are the MixedPrecisionPolicy dtypes; None falls back to the
    args-based default (bf16 / fp32, or fp16 param when args.fp16).

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and next(iter(layer_cls_to_wrap)) is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    if param_dtype is None:
        param_dtype = torch.float16 if args.fp16 else torch.bfloat16
    if reduce_dtype is None:
        reduce_dtype = torch.float32

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # fully_shard each layer first, then the root model
    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    return model
