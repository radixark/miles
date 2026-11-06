import logging
import os
from pathlib import Path
from typing import Any

import torch

from miles.backends.megatron_utils.lora.native import resolve_adapter_targets, validate_native_lora_args
from miles.backends.sglang_utils.arguments import sglang_attention_dp_enabled
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import has_plain_peft_form
from miles.backends.training_utils.checkpoint.adapter.parametrization import read_parametrization
from miles.backends.training_utils.checkpoint.adapter.readers import (
    AdapterSourceKind,
    classify_adapter_source,
    read_source_lora_alpha,
)
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import (
    LORA_TARGET_GROUPS,
    exclude_hf_lora_targets,
    expand_hf_lora_targets,
    expand_packed_hf_lora_targets,
    gdn_in_proj_served_targets,
    get_hf_lora_targets,
    has_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)
from miles.utils.lora.utils import (
    engine_loads_adapter_from_disk,
    exports_hf_snapshots,
    is_lora_enabled,
    matches_lora_target,
    reads_adapter_slots,
    targets_expert_leaves,
)

logger = logging.getLogger(__name__)

_DEFAULT_LORA_ALPHA = 16.0


def validate_lora_args(args):
    if args.lora_type is None:
        args.lora_type = _default_lora_type(args)
    validate_multi_lora_args(args)
    # checked before the early return: without LoRA nothing would write the destination
    assert (
        args.lora_adapter_convert_save is None or args.lora_adapter_path is not None
    ), "--lora-adapter-convert-save converts --lora-adapter-path"
    if not is_lora_enabled(args):
        return
    # trainers resolve relative paths from their own cwd; rollout-only keeps the string so a hub id reaches SGLang
    if args.lora_adapter_path is not None and not args.debug_rollout_only:
        args.lora_adapter_path = os.path.abspath(args.lora_adapter_path)
    args.lora_alpha = _resolve_lora_alpha(args)
    if args.lora_adapter_convert_save is not None:
        # every cell loads the source and its rank 0 would commit into the same destination
        assert not args.indep_dp, "--lora-adapter-convert-save does not support --indep-dp (or train fault tolerance)"
        # the trainer actors write it from their own working directory, which need not be the driver's
        args.lora_adapter_convert_save = os.path.abspath(args.lora_adapter_convert_save)
        # the commit would retire and delete whatever sits there, including the source when converting in place
        assert not os.path.lexists(
            args.lora_adapter_convert_save
        ), f"--lora-adapter-convert-save {args.lora_adapter_convert_save} already exists; pass a new path"
        # rollout-only builds no model, so nothing would load or write the adapter
        assert not args.debug_rollout_only, "--lora-adapter-convert-save needs the trainer; drop --debug-rollout-only"
    if engine_loads_adapter_from_disk(args):
        _validate_rollout_only_adapter(args)
    assert args.train_backend in (
        "megatron",
        "torchtitan",
    ), f"LoRA injection is not implemented for --train-backend {args.train_backend}; use megatron or torchtitan"
    assert args.lora_rank > 0, "LoRA requires a positive --lora-rank, including when loading an adapter"
    hf_config = load_hf_config(args.hf_checkpoint)
    args.hf_lora_targets = _resolve_lora_targets(args, hf_config)
    args.lora_adapter_targets = list(args.hf_lora_targets)
    if args.lora_adapter_path is not None and not args.debug_rollout_only:
        _validate_megatron_bridge_source(args)
    if args.train_backend == "megatron" and args.megatron_to_hf_mode == "bridge":
        from miles.backends.megatron_utils.lora.target_modules import check_fused_lora_selection

        canonical = args.lora_type == "canonical_lora"
        check_fused_lora_selection(args.hf_checkpoint, args.hf_lora_targets, canonical=canonical)
        if canonical:
            # Bridge's canonical adapters export a GDN in_proj by component
            modules = [
                m for m in _hf_modules(hf_config) if any(matches_lora_target(m, t) for t in args.hf_lora_targets)
            ]
            args.lora_adapter_targets = gdn_in_proj_served_targets(
                modules or args.hf_lora_targets, args.hf_lora_targets
            )

    # Training and serving must agree on shared-outer grouped-expert LoRA (expert_dim=1).
    if args.experts_shared_outer_loras and hasattr(
        args, "sglang_experts_shared_outer_loras"
    ):  # config-access-exempt: older SGLang parsers omit the expert-LoRA switch
        args.sglang_experts_shared_outer_loras = True
    assert args.experts_shared_outer_loras == bool(
        getattr(args, "sglang_experts_shared_outer_loras", args.experts_shared_outer_loras)
    ), "experts_shared_outer_loras and sglang_experts_shared_outer_loras must agree"  # config-access-exempt: older SGLang parsers omit the expert-LoRA switch
    if targets_expert_leaves(args.hf_lora_targets):
        logger.warning(
            "MoE-expert LoRA layout: %s (--experts-shared-outer-loras).",
            "shared-outer" if args.experts_shared_outer_loras else "per-expert",
        )


def _resolve_lora_alpha(args) -> float:
    """The source adapter owns alpha when there is one; --lora-alpha may only repeat it."""
    # a hub id is no local source: SGLang, not miles, reads its alpha
    if args.lora_adapter_path is None or not Path(args.lora_adapter_path).is_dir():
        return _DEFAULT_LORA_ALPHA if args.lora_alpha is None else args.lora_alpha
    source = read_source_lora_alpha(Path(args.lora_adapter_path))
    if source is None:
        assert args.lora_alpha is not None, f"{args.lora_adapter_path} records no LoRA alpha; pass --lora-alpha"
        return args.lora_alpha
    assert args.lora_alpha in (
        None,
        source,
    ), f"--lora-alpha {args.lora_alpha} differs from {args.lora_adapter_path}'s {source}; drop the flag to use it"
    return source


def _validate_megatron_bridge_source(args) -> None:
    """A Megatron-Bridge checkpoint's sharded keys exist only in a Bridge model: refused before the model is built."""
    path = Path(args.lora_adapter_path)
    # only a DCP source can be one
    if not (path / ".metadata").is_file():
        return
    kind = classify_adapter_source(path, args.hf_lora_targets, load_optimizer=False)
    assert kind is not AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT or (
        args.train_backend == "megatron" and args.megatron_to_hf_mode == "bridge"
    ), (
        f"{path} is a Megatron-Bridge checkpoint, which only --megatron-to-hf-mode bridge loads; "
        "convert it there first with --lora-adapter-convert-save"
    )


def _validate_rollout_only_adapter(args) -> None:
    """A local source the library reads must carry an HF-named PEFT pair; anything else goes to SGLang as given."""
    path = Path(args.lora_adapter_path)
    hf_config = load_hf_config(args.hf_checkpoint).to_dict()
    # without the family's layout the library reads no source; SGLang gets it as before
    if not path.is_dir() or not has_hf_lora_targets(hf_config):
        return
    targets = get_hf_lora_targets(hf_config)
    try:
        kind = classify_adapter_source(path, targets, load_optimizer=False)
    except ValueError:
        # not a source the library reads
        return
    assert not kind.is_read_into_model(), (
        f"{path} is a {kind.value} checkpoint, which only a trainer can read; "
        "convert it first with --lora-adapter-convert-save"
    )
    assert kind != AdapterSourceKind.EXTERNAL_WEIGHTS, (
        f"{path} uses official Tinker tensor names, which need a conversion that rollout-only does not do; "
        "convert it first with --lora-adapter-convert-save"
    )
    # a sampler export holds the engines' own form whatever the parametrization
    has_export = (path / "adapter_model.safetensors").is_file()
    # any other such adapter dir never gets a PEFT export, converted or not
    assert (
        has_export or kind != AdapterSourceKind.ADAPTER_DIR or has_plain_peft_form(read_parametrization(path), targets)
    ), (
        f"{path} has no plain-PEFT form (fused-module components with their own lora_A), so rollout-only cannot "
        "serve it to the engines"
    )
    assert (
        has_export
    ), f"{path} has no PEFT export for the engines to load; write one first with --lora-adapter-convert-save"


def _default_lora_type(args) -> str:
    """The one owner of --lora-type's default. Native gives every projection its spec declares its own lora_A
    and Tinker serves the PEFT / cookbook layout; Bridge's LoRA and torchtitan share one lora_A per fused
    projection."""
    if args.train_backend == "megatron" and args.megatron_to_hf_mode == "raw":
        return "canonical_lora"
    if args.entry == "serve":
        return "canonical_lora"
    return "lora"


def validate_lora_args_after_megatron(args):
    """Checks that need Megatron's resolved args or the resolved eval-snapshot setting: Bridge LoRA's HF-export
    layout, and the targets native (raw-mode) LoRA's spec serves."""
    # save() writes a LoRA model to its adapter dir, never into the in-memory checkpoint a heal or rollback restores
    if is_lora_enabled(args) and not args.multi_lora and not args.debug_rollout_only:
        assert not args.indep_dp and args.non_persistent_ckpt_type != "local", (
            "LoRA does not support train fault tolerance (--indep-dp / in-memory checkpoints): a healed or rolled-back "
            "cell would lose the adapter; remove 'train' from --ft-components and resume from a saved iter_N/adapter "
            "with --lora-adapter-path"
        )
        # without --lora-adapter-path the run would start from the base model and overwrite those adapters
        assert args.lora_adapter_path is not None or not (args.load and any(Path(args.load).glob("iter_*/adapter"))), (
            f"--load {args.load} holds LoRA adapter checkpoints (iter_*/adapter): resume one with "
            "--lora-adapter-path, or point --load elsewhere"
        )
        # rollout-only builds no optimizer and saves nothing
        _validate_adapter_state_args(args)
    # after miles_validate_args: the actor's predicate reads eval_uses_snapshots, resolved there
    if is_lora_enabled(args):
        _assert_adapter_layout_defined(args)
    if not (is_lora_enabled(args) and args.megatron_to_hf_mode == "raw"):
        return
    hf_config = load_hf_config(args.hf_checkpoint)
    hf_modules = _hf_modules(hf_config)
    args.lora_adapter_targets = _native_serving_targets(args, hf_config, args.hf_lora_targets, hf_modules)
    validate_native_lora_args(args, hf_modules)


def _validate_adapter_state_args(args) -> None:
    """adapter_state/ holds Adam's fp32 master, fp32 moments and step, gathered whole per DP group over Gloo."""
    if args.save is None and args.lora_adapter_path is None:
        return
    assert not args.debug_disable_optimizer or (args.save is None and args.lora_adapter_convert_save is None), (
        "--save / --lora-adapter-convert-save store the adapter's fp32 optimizer masters, which "
        "--debug-disable-optimizer does not build; drop one of them"
    )
    if args.debug_disable_optimizer:
        # --lora-adapter-path then loads the adapter's weights only, as into a model without an optimizer
        return
    assert args.optimizer == "adam", f"--save / --lora-adapter-path store Adam state; got --optimizer {args.optimizer}"
    assert args.exp_avg_dtype == args.exp_avg_sq_dtype == torch.float32, "the adapter checkpoint stores fp32 moments"
    assert args.main_params_dtype == torch.float32, "the adapter checkpoint stores fp32 masters"
    assert (
        args.use_distributed_optimizer
    ), "the adapter checkpoint gathers optimizer state from the distributed optimizer"
    assert args.use_gloo_process_groups, "the adapter checkpoint gathers optimizer state over Gloo"
    assert not args.stream_optimizer_state_to_disk, (
        "--save / --lora-adapter-path gather the fp32 masters and moments from the distributed optimizer, but "
        "--stream-optimizer-state-to-disk keeps the moments in its own store and defers the masters"
    )
    # TE FusedAdam keeps the master as an int16 remainder unless the optimizer runs on the host
    assert (
        not args.use_precision_aware_optimizer or args.optimizer_cpu_offload
    ), "--use-precision-aware-optimizer needs --optimizer-cpu-offload to save or load a LoRA adapter"
    if args.lora_adapter_path is not None:
        # loading writes masters and re-casts them, which an overlapped param all-gather would race
        assert not args.overlap_param_gather, "--lora-adapter-path cannot be combined with --overlap-param-gather"
        # the load picks each DP group's leader on the full DP ranks, but the optimizer scatters per instance
        assert (
            args.num_distributed_optimizer_instances == 1
        ), "--lora-adapter-path cannot be combined with --num-distributed-optimizer-instances > 1"


def _assert_adapter_layout_defined(args) -> None:
    """The adapter slots need the family's HF LoRA target layout; runs that build none train any family."""
    # a rollout-only run has no trainer; SGLang reads its adapter
    if args.debug_rollout_only:
        return
    if reads_adapter_slots(args, args.save):
        hf_config = load_hf_config(args.hf_checkpoint)
        assert has_hf_lora_targets(hf_config.to_dict()), (
            f"LoRA --save, --save-hf, eval snapshots and --lora-adapter-path need the HF LoRA target layout of "
            f"model_type={hf_config.model_type!r}, which miles does not define (miles/utils/lora/hf_lora_targets.py); "
            "run without them or define the layout"
        )
        # only Bridge exports each expert's side under one unindexed name
        if exports_hf_snapshots(args) and args.megatron_to_hf_mode == "bridge":
            _assert_bridge_hf_exports_keep_every_expert(args, hf_config)


def _assert_bridge_hf_exports_keep_every_expert(args, hf_config) -> None:
    targets = get_hf_lora_targets(hf_config.to_dict())
    adapts_experts = any(
        ".experts." in module and any(matches_lora_target(module, target) for target in args.hf_lora_targets)
        for module in _hf_modules(hf_config)
    )
    # Bridge exports a packed family's per-expert side under one unindexed name for every expert
    packed_shared_outer = targets.packed_expert_gate_up_layout is not None and args.experts_shared_outer_loras
    assert not (packed_shared_outer and adapts_experts), (
        "--save-hf and eval snapshots of a packed-expert family's --experts-shared-outer-loras adapter would keep "
        "only the last expert's per-expert side: drop --experts-shared-outer-loras, or --save-hf and eval snapshots"
    )


def _native_serving_targets(args, hf_config, hf_targets: list[str], hf_modules: list[str]) -> list[str] | str:
    """The targets SGLang serves for ``hf_targets`` under Megatron's native spec, which refuses
    projections it has no adapter for."""
    return resolve_adapter_targets(
        hf_config.to_dict(),
        hf_targets,
        hf_modules=hf_modules,
        lora_type=args.lora_type,
        experts_shared_outer_loras=args.experts_shared_outer_loras,
    )


def _hf_modules(hf_config) -> list[str]:
    return sorted(name.removesuffix(".weight") for name in HfWeightMapping.from_config(hf_config).parameter_names)


def _resolve_lora_targets(args, hf_config) -> list[str]:
    """Return the final HF targets without changing the CLI selectors in `args`."""
    hf_modules = _hf_modules(hf_config)
    targets = parse_lora_targets(args.target_modules)
    exclusions = parse_lora_targets(args.exclude_modules) or []
    explicit_targets = []
    if targets is not None:
        expanded = expand_packed_hf_lora_targets(targets, hf_modules)
        explicit_targets = [
            target for target in dict.fromkeys(targets + expanded) if target not in (*LORA_TARGET_GROUPS, "all-linear")
        ]
        targets = expanded
    if exclusions:
        conflicts = {
            module
            for module in hf_modules + explicit_targets + exclusions
            if any(matches_lora_target(module, target) for target in explicit_targets)
            and any(matches_lora_target(module, exclude) for exclude in exclusions)
        }
        assert not conflicts, f"Explicit LoRA targets overlap --exclude-modules: {sorted(conflicts)}"
    if targets is None and args.multi_lora:
        targets = list(LORA_TARGET_GROUPS)
    targets = resolve_hf_lora_targets(
        hf_config.to_dict(), target_modules=targets, whole_fused=args.lora_type == "lora"
    )
    targets = exclude_hf_lora_targets(targets, exclusions)

    if all(any(matches_lora_target(module, target) for module in hf_modules) for target in targets + exclusions):
        hf_targets = targets
        if exclusions:
            selected = [
                module for module in hf_modules if any(matches_lora_target(module, target) for target in targets)
            ]
            hf_targets = exclude_hf_lora_targets(selected, exclusions)
    elif args.megatron_to_hf_mode == "bridge":
        # Only legacy Megatron selectors need Bridge before trainer creation.
        from miles.backends.megatron_utils.lora.target_modules import normalize_lora_targets_to_hf

        hf_targets = normalize_lora_targets_to_hf(
            args.hf_checkpoint,
            targets,
            canonical=args.lora_type == "canonical_lora",
            exclude_modules=exclusions,
            explicit_targets=explicit_targets,
        )
    else:
        layout = get_hf_lora_targets(hf_config.to_dict())
        hf_targets = exclude_hf_lora_targets(expand_hf_lora_targets(targets, layout), exclusions)
    return hf_targets


def validate_multi_lora_args(args: Any) -> None:
    """Set ``args.multi_lora``, then validate the trainer-side constraints of
    the slot machinery. A no-op for normal runs."""
    args.multi_lora = args.multi_lora_n_adapters > 0
    if not args.multi_lora:
        return

    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert (
        args.lora_adapter_path is None
    ), "Multi-LoRA loads adapters into its slots through the Tinker gateway; --lora-adapter-path is single LoRA only"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
    assert (
        not args.allgather_cp
    ), "Multi-LoRA does not support --allgather-cp: per-adapter token spans assume the zigzag CP layout."
    # Adapter routing is only recompute-safe without pipelining; enforce at launch.
    assert args.pipeline_model_parallel_size == 1, (
        "Multi-LoRA requires --pipeline-model-parallel-size 1: a pipelined schedule would "
        "recompute activations against a later micro-batch's adapter routing."
    )
    # Per-slot token spans assume sequence-major contiguous sample packing, which only 'thd' provides.
    assert args.qkv_format == "thd", (
        "Multi-LoRA requires --qkv-format thd: per-adapter token spans assume the "
        f"micro-batch packs samples contiguously, which bshd does not (got {args.qkv_format!r})."
    )
    assert "muon" not in str(args.optimizer).lower(), (
        "Multi-LoRA does not support Muon: per-adapter decoupled stepping is only "
        "implemented for Adam-family per-slot optimizers"
    )
    assert not args.colocate, "Multi-LoRA requires separate training and sampling GPUs to retain accumulated gradients"
    assert (
        not args.indep_dp and "train" not in args.ft_components
    ), "Multi-LoRA does not support independent-DP training; remove 'train' from --ft-components"
    assert not args.offload_train, (
        "Multi-LoRA retains per-adapter gradient accumulation in GPU buffers between "
        "train calls; --offload-train would destroy it. Disable offload for multi-LoRA."
    )
    assert not args.enable_witness, (
        "Multi-LoRA runs without the distributed optimizer (per-slot LayerWise "
        "optimizers); the witness module assumes use_distributed_optimizer"
    )
    assert args.sglang_tokenizer_worker_num == 1, (
        "Multi-LoRA requires --sglang-tokenizer-worker-num 1: dynamic adapter loading "
        "requires a single tokenizer-side LoRA registry."
    )
    assert not sglang_attention_dp_enabled(
        attn_dp_size=args.sglang_attn_dp_size,
        enable_dp_attention=args.sglang_enable_dp_attention,
        dp_size=args.sglang_dp_size,
    ), (
        "Multi-LoRA cannot run on SGLang attention DP: SGLang serves only pinned adapters there, and a pinned "
        "adapter is never evicted, so the engine could not swap adapters through its LoRA slots. "
        "Drop --sglang-attn-dp-size / --sglang-enable-dp-attention for multi-LoRA."
    )
    assert not args.calculate_per_token_loss, (
        "Multi-LoRA normalizes each sample by its adapter batch "
        "(sample-mean); per-token loss normalization would make adapter batch weights "
        "depend on batch contents. Drop --calculate-per-token-loss."
    )
    assert (args.optimizer or "adam").lower() == "adam", (
        "Multi-LoRA requires --optimizer adam: the per-slot SlotOptimizer only "
        f"implements Adam semantics; got --optimizer {args.optimizer}"
    )
    assert args.megatron_to_hf_mode == "bridge", (
        "Multi-LoRA runs on Megatron-Bridge adapters; pass --megatron-to-hf-mode bridge "
        f"(got {args.megatron_to_hf_mode!r})"
    )
    # train.py's save() writes single LoRA's adapter dir; multi-LoRA slots are saved through the Tinker gateway
    assert (
        args.entry != "train" or args.save is None
    ), "Multi-LoRA saves its adapters through the Tinker gateway; --save is single LoRA only"
