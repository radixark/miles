"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora import is_lora_enabled, lora_rollout_enabled  # noqa: F401  (re-exported)
from miles_plugins.lora.hf_adapter import MEGATRON_MLA_TO_HF, convert_target_modules_to_hf  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

_DEFAULT_LORA_PROVIDER = "miles_plugins.lora"
_NATIVE_LORA_PROVIDER_PATHS = (
    None,
    "miles_plugins.lora",
    "miles_plugins.lora.lora",
)

# ---------------------------------------------------------------------------
# Unified HF <-> Megatron module name mappings
# ---------------------------------------------------------------------------

# Standard LoRA: merged Q/K/V and merged up/gate
_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
    # GDN (Qwen3.5/Qwen3-Next): both slices live in the single fused megatron in_proj
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
}

_STANDARD_LORA_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate
_CANONICAL_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
}

_CANONICAL_LORA_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

_HF_MODULE_NAMES = frozenset(_STANDARD_LORA_HF_TO_MEGATRON)

# DeepSeek / Kimi MLA (HF names on checkpoint; Megatron uses linear_* from Megatron-Bridge mappings).
_MLA_HF_TO_MEGATRON = {hf: megatron for megatron, hf in MEGATRON_MLA_TO_HF.items()}

# Empty: dropping a module here makes sglang silently skip its shipped adapter tensors.
_SGLANG_UNSUPPORTED_HF_TARGETS = frozenset()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


def uses_builtin_native_lora_provider(args: Namespace) -> bool:
    """Whether this run uses the built-in native provider contract."""
    if getattr(args, "megatron_to_hf_mode", "raw") == "bridge":
        return False
    return getattr(args, "lora_provider_path", None) in _NATIVE_LORA_PROVIDER_PATHS


def reduce_marked_lora_grads(model: Sequence[torch.nn.Module]) -> None:
    """Compatibility delegate to the native plugin's distributed gradient reducer."""
    from miles_plugins.lora.distributed import reduce_marked_lora_grads as reduce_native_lora_grads

    reduce_native_lora_grads(model)


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def is_lora_weight_name(name: str) -> bool:
    """Check if a weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


def _adapter_shard_name(tp_rank: int, pp_rank: int, ep_rank: int, *, ep_sharded: bool) -> str:
    """Filename identifying this rank's adapter shard.

    ``ep_sharded`` keys the name by EP rank, since bridge/custom providers may
    hold routed-expert adapter state; native state is EP-invariant
    (``native_adapter_shard_name``).
    """
    from miles_plugins.lora.checkpointing import native_adapter_shard_name

    name = native_adapter_shard_name(tp_rank, pp_rank)
    if ep_sharded and ep_rank > 0:
        name = name.removesuffix(".pt") + f"_ep{ep_rank}.pt"
    return name


def _adapter_shards_are_ep_sharded(model: Sequence[torch.nn.Module]) -> bool:
    """Derive the shard policy from the modules that were actually attached."""
    from miles_plugins.lora.checkpointing import has_native_adapters

    return not has_native_adapters(model)


def _non_native_adapter_load_plan(model, state_dict):
    """Reproduce the pre-refactor Bridge/custom name-matching contract.

    Non-native checkpoints keep their historical unqualified keys: duplicate
    VPP names receive the same surviving tensor, while missing/extra names are
    ignored. Validation is deferred into a plan only so all ranks can agree
    before any copy; the native codec remains the sole owner of strict,
    chunk-qualified checkpoint semantics.
    """
    from miles_plugins.lora.checkpointing import AdapterLoadPlan

    assignments = []
    shape_mismatches = []
    for chunk in model:
        for name, parameter in chunk.named_parameters():
            if not _is_adapter_param_name(name) or name not in state_dict:
                continue
            tensor = state_dict[name]
            if not isinstance(tensor, torch.Tensor):
                shape_mismatches.append(f"{name}: checkpoint value is {type(tensor).__name__}, expected a tensor")
            elif tuple(tensor.shape) != tuple(parameter.shape):
                shape_mismatches.append(
                    f"{name}: checkpoint {tuple(tensor.shape)} != parameter {tuple(parameter.shape)}"
                )
            else:
                assignments.append((name, parameter, tensor))
    return AdapterLoadPlan(assignments, [], [], shape_mismatches)


def _is_canonical_shard_writer(shard_name: str) -> bool:
    """True on exactly one rank per distinct shard filename.

    Elects the lowest global rank holding each filename, so every name some rank
    asks for on resume gets written. Collective: all ranks must call it.

    DP-rank-0 gating is not enough for EP-keyed names: under TP2/EP4 on 8 GPUs
    DP-rank-0 is {0, 1}, so only EP 0-1 were written. With no EP component in
    the name this election collapses to DP-rank-0.
    """
    if not dist.is_initialized():
        return True
    names: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(names, shard_name)
    return names.index(shard_name) == dist.get_rank()


_param_grad_buffer_patched = False


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer to use disable_param_buffers_cpu_backup=True.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  During LoRA training, base weights are
    frozen (requires_grad=False) so DDP only creates buffers for adapter params.

    This patch ensures those buffers are allocated in the "param_buffer" region
    (enable_cpu_backup=False), making them invisible to pause(tag="default") —
    eliminating the need for resume()/pause() around update_weights.

    The patch is idempotent and only takes effect once.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return
    _param_grad_buffer_patched = True

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        # Megatron reads these flags from ddp_config (its first ctor argument).
        ddp_config = kwargs.get("ddp_config", args[0] if args else None)
        ddp_config.disable_param_buffers_cpu_backup = True
        ddp_config.disable_grad_buffers_cpu_backup = True
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (disable cpu backup)")


# ---------------------------------------------------------------------------
# Module name conversion
# ---------------------------------------------------------------------------


def _get_lora_class_name(lora_type: type | object | None) -> str:
    """Resolve LoRA type to its class name string."""
    if lora_type is None:
        return "CanonicalLoRA"
    if isinstance(lora_type, type):
        return lora_type.__name__
    return type(lora_type).__name__


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    lora_type: type | object | None = None,
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.

    HF:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Megatron (LoRA):          linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj,
                              linear_fc1_up, linear_fc1_gate, linear_fc2

    Special values: "all", "all-linear", "all_linear" -> all standard linear modules.
    If input is already in Megatron format, returns as-is.
    """
    class_name = _get_lora_class_name(lora_type)
    is_canonical = class_name == "CanonicalLoRA"

    all_modules = _CANONICAL_LORA_ALL_MODULES if is_canonical else _STANDARD_LORA_ALL_MODULES
    hf_to_megatron = _CANONICAL_LORA_HF_TO_MEGATRON if is_canonical else _STANDARD_LORA_HF_TO_MEGATRON

    # Handle special "all-linear" variants
    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    if isinstance(hf_modules, tuple):
        hf_modules = list(hf_modules)

    # Check if already in Megatron format (standard / canonical / Kimi MLA linear_*).
    if all(m not in _HF_MODULE_NAMES and m not in _MLA_HF_TO_MEGATRON for m in hf_modules if "*" not in m):
        return list(hf_modules)

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        if module in _MLA_HF_TO_MEGATRON:
            megatron_name = _MLA_HF_TO_MEGATRON[module]
        else:
            megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)

    return megatron_modules


def target_modules_hf_for_sglang_rollout(args: Namespace) -> list[str]:
    """HF target_modules for SGLang LoRA init/sync (minus _SGLANG_UNSUPPORTED_HF_TARGETS, currently empty)."""
    raw = list(args.target_modules) if args.target_modules else []
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    checkpoint_readable = not hf_checkpoint or os.path.exists(os.path.join(hf_checkpoint, "config.json"))
    if uses_builtin_native_lora_provider(args) and checkpoint_readable:
        from miles_plugins.lora.sglang_adapter import sglang_target_modules

        hf = sglang_target_modules(args)
    else:
        hf = convert_target_modules_to_hf(raw)
    out = [m for m in hf if m not in _SGLANG_UNSUPPORTED_HF_TARGETS]
    dropped = set(hf) - set(out)
    if dropped:
        logger.warning(
            "target_modules_hf_for_sglang_rollout: omitting %s for SGLang (unsupported by default "
            "get_hidden_dim); Megatron should not train LoRA on these if rollout sync is required.",
            sorted(dropped),
        )
    return out


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def create_lora_instance(args: Namespace):
    """Create a LoRA or CanonicalLoRA instance based on args.

    ``--exclude-modules`` is not forwarded: ``parse_lora_target_modules`` already
    subtracted those names from ``--target-modules``.

    Unsupported:

    - Bridge excludes with targets: asserts at build (``peft/module_matcher.py``).

    Returns:
        A LoRA/CanonicalLoRA dataclass instance ready to be applied to a model.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_type_name = getattr(args, "lora_type", "lora").lower()

    if lora_type_name == "canonical_lora":
        lora_cls = CanonicalLoRA
    else:
        lora_cls = LoRA

    target_modules = convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls)

    lora_kwargs = dict(
        target_modules=target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )
    # shared-outer grouped-expert LoRA (SGLang PR #21466); per-expert is the default
    if getattr(args, "experts_shared_outer_loras", False):
        assert lora_cls is LoRA, "--experts-shared-outer-loras requires the standard LoRA adapter type"
        lora_kwargs["experts_shared_outer_loras"] = True

    lora = lora_cls(**lora_kwargs)

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules} "
        f"(--exclude-modules {getattr(args, 'exclude_modules', None)!r} already applied by the parser)"
    )
    return lora


def resolve_lora_provider(args: Namespace):
    """Return the module implementing the native-LoRA provider protocol.

    ``--lora-provider-path`` selects a model-specific implementation (a dotted
    module path); the default is the ``miles_plugins.lora`` plugin.
    """
    import importlib

    path = getattr(args, "lora_provider_path", None) or _DEFAULT_LORA_PROVIDER
    module = importlib.import_module(path)
    for entry_point in ("wrap_model_provider_with_lora", "load_lora_adapter_hf", "export_lora_hf_named"):
        assert hasattr(module, entry_point), f"--lora-provider-path {path} must define {entry_point}()"
    return module


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def pp_assemble_full_adapter(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Assemble the complete adapter on every PP rank (the exporter gathers TP/EP, not PP)."""
    import math

    pp_group = get_parallel_state().pp.group
    pp_size = dist.get_world_size(group=pp_group)
    if pp_size == 1:
        return hf_named_tensors
    pp_rank = dist.get_rank(group=pp_group)
    global_ranks = dist.get_process_group_ranks(pp_group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp_size
    dist.all_gather_object(all_meta, local_meta, group=pp_group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src_pp, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src_pp == pp_rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src_pp], group=pp_group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Save LoRA adapter checkpoint to disk.

    Saves in two formats:
    1. **HF PEFT format** (``adapter_model.safetensors`` + ``adapter_config.json``) for
       external tool compatibility, and for reloading through ``--lora-adapter-path``.
       Bridge mode exports via Megatron-Bridge's ``export_adapter_weights``; raw mode
       via the native provider, whose adapters the bridge exporter cannot see. Both
       handle fused QKV / gate-up splitting and TP gathering. Tensors are cloned before
       writing because that splitting aliases them -- a fused ``linear_fc1`` has one
       ``lora_A`` that exports under both ``gate_proj`` and ``up_proj``, and its ``B``
       becomes two row views -- and ``safetensors`` refuses to write shared storage.
    2. **Megatron-native format** for checkpoint resume without name/weight
       conversion. The native provider writes one MCore distributed checkpoint
       (``torch_dist/``) with standard sharded-state-dict keys, holding adapter
       weights and optimizer/scheduler state together; it reshards on load
       across TP/PP/DP layout changes. Bridge/custom providers retain their
       per-rank ``adapter_megatron_tp{tp}_pp{pp}_ep{ep}.pt`` shard format plus
       per-rank training-state files, which require the exact saved layout.

    Base model weights are frozen and never change, so they are not saved.

    This function is collective: **all ranks must call it** — the HF export
    performs TP all-gathers, and the native dist-checkpoint save is a
    collective in which every main replica writes its shard. The HF PEFT files
    are written by one rank; legacy bridge/custom shards by one rank per
    ``(tp, pp, ep)`` coordinate.
    """
    import json

    from megatron.bridge import AutoBridge
    from safetensors.torch import save_file

    from miles.utils import megatron_bridge_utils

    save_path = Path(save_dir)
    parallel_state = get_parallel_state()
    is_dp_cp_rank_0 = parallel_state.effective_dp.rank == 0 and parallel_state.cp.rank == 0
    tp_rank = parallel_state.tp.rank
    pp_rank = parallel_state.pp.rank
    ep_rank = parallel_state.ep.rank
    bridge_mode = getattr(args, "megatron_to_hf_mode", "raw") == "bridge"

    save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    ep_sharded_provider = _adapter_shards_are_ep_sharded(model)
    if ep_sharded_provider:
        shard_name = _adapter_shard_name(tp_rank, pp_rank, ep_rank, ep_sharded=True)
        if _is_canonical_shard_writer(shard_name):
            adapter_state = {}
            for model_chunk in model:
                for name, parameter in model_chunk.named_parameters():
                    if _is_adapter_param_name(name):
                        adapter_state[name] = parameter.detach().cpu()
            native_path = save_path / shard_name
            torch.save(adapter_state, native_path)
            logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")
    else:
        from miles_plugins.lora.checkpointing import NATIVE_DIST_CKPT_DIRNAME, save_native_adapter_dist_checkpoint

        save_native_adapter_dist_checkpoint(
            model,
            save_path / NATIVE_DIST_CKPT_DIRNAME,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )

    lora_state_dict: dict[str, torch.Tensor] = {}
    if bridge_mode:
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        with megatron_bridge_utils.patch_megatron_model(model):
            for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                model,
                cpu=True,
                show_progress=False,
            ):
                lora_state_dict[hf_name] = weight
    else:
        for hf_name, weight in resolve_lora_provider(args).export_lora_hf_named(model):
            lora_state_dict[hf_name] = weight.cpu()

    if parallel_state.pp.size > 1:
        assembled = pp_assemble_full_adapter([(name, w.cuda()) for name, w in lora_state_dict.items()])
        lora_state_dict = {name: w.cpu() for name, w in assembled}

    if is_dp_cp_rank_0 and tp_rank == 0 and pp_rank == 0:
        save_file(
            {name: weight.detach().contiguous().clone() for name, weight in lora_state_dict.items()},
            save_path / "adapter_model.safetensors",
        )

        if bridge_mode:
            target_modules_hf = (
                convert_target_modules_to_hf(list(args.target_modules))
                if args.target_modules
                else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        else:
            from miles_plugins.lora.hf_adapter import target_modules_from_hf_names

            target_modules_hf = target_modules_from_hf_names(lora_state_dict)
        config = {
            "peft_type": "LORA",
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": target_modules_hf,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        os.sync()
        logger.info(f"Saved HF PEFT adapter to {save_path} with {len(lora_state_dict)} tensors")

    if optimizer is not None and ep_sharded_provider:
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(
            {
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            },
            save_path / f"training_state_rank{rank}.pt",
        )
        logger.info(f"Saved optimizer/scheduler state to {save_path}")

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    """Load LoRA adapter weights from a saved checkpoint into the model.

    For the native provider, an MCore distributed checkpoint (``torch_dist/``)
    is preferred: one collective load restores adapter weights plus
    optimizer/scheduler state and reshards automatically if the TP/PP/DP layout
    changed since the save. Incompatible contents (e.g. a different
    ``--target-modules`` set) raise on every rank alike. Otherwise the legacy
    per-rank ``.pt`` shard format is tried, which requires the exact saved
    layout. HF PEFT adapters are not loaded here (``--lora-adapter-path``
    handles that format through the provider).

    In the legacy path every rank preflights its complete adapter shard
    (existence, exact target names, and shapes), then collectively agrees
    before any model parameter is mutated. When ``optimizer`` is provided,
    training state (optimizer + LR scheduler) has a second consensus point so
    iterations and LR schedules stay in lockstep. Either way the function is
    collective — when ``torch.distributed`` is initialized, all ranks must
    call it.

    Args:
        model: List of DDP-wrapped model chunks with LoRA layers already applied.
        adapter_path: Path to the adapter checkpoint directory.
        optimizer: If provided, restore optimizer state for training resume.
        opt_param_scheduler: If provided, restore LR scheduler state.

    Returns:
        ``(loaded, iteration)`` — *loaded* is True if adapter weights were
        successfully loaded; *iteration* is the saved iteration number (or None
        if no training state was found).
    """
    adapter_dir = Path(adapter_path)
    for chunk in model:
        chunk._miles_lora_native_checkpoint_loaded = False

    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank
    ep_rank = get_parallel_state().ep.rank

    ep_sharded_provider = _adapter_shards_are_ep_sharded(model)
    native_adapters = not ep_sharded_provider

    if native_adapters:
        from miles_plugins.lora.checkpointing import (
            NATIVE_DIST_CKPT_DIRNAME,
            is_native_adapter_dist_checkpoint,
            load_native_adapter_dist_checkpoint,
        )

        dist_ckpt_dir = adapter_dir / NATIVE_DIST_CKPT_DIRNAME
        if _all_ranks_see_dist_checkpoint(is_native_adapter_dist_checkpoint(dist_ckpt_dir), dist_ckpt_dir):
            iteration = load_native_adapter_dist_checkpoint(
                model, dist_ckpt_dir, optimizer=optimizer, opt_param_scheduler=opt_param_scheduler
            )
            for chunk in model:
                chunk._miles_lora_native_checkpoint_loaded = True
            return True, iteration

    from miles_plugins.lora.checkpointing import native_adapter_load_plan

    native_path = adapter_dir / _adapter_shard_name(tp_rank, pp_rank, ep_rank, ep_sharded=ep_sharded_provider)
    shard_found = native_path.exists()
    plan = None
    load_error: Exception | None = None
    if shard_found:
        try:
            state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
            if native_adapters:
                plan = native_adapter_load_plan(model, state_dict)
            else:
                plan = _non_native_adapter_load_plan(model, state_dict)
        except Exception as error:  # Keep every rank alive until the agreement point.
            load_error = error

    local_adapter_ok = shard_found and load_error is None and plan is not None and plan.compatible
    all_adapters_ok = _all_ranks_can_restore_training_state(local_adapter_ok)
    if not all_adapters_ok:
        if not adapter_dir.exists():
            logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        elif not shard_found:
            logger.warning("Native LoRA adapter shard is missing: %s", native_path)
        elif load_error is not None:
            logger.warning("Could not preflight native LoRA adapter shard %s: %s", native_path, load_error)
        elif plan is not None:
            if plan.unexpected:
                logger.warning(
                    "Native adapter shard has %d tensors absent from the current exact target/chunk set: %s",
                    len(plan.unexpected),
                    plan.unexpected[:8],
                )
            if plan.missing:
                logger.warning(
                    "Native adapter shard is missing %d current adapter parameters: %s",
                    len(plan.missing),
                    plan.missing[:8],
                )
            if plan.shape_mismatches:
                logger.warning("Native adapter shard has incompatible tensor shapes: %s", plan.shape_mismatches[:8])
        logger.warning(
            "Skipping adapter and optimizer/scheduler restore before mutation: at least one rank "
            "reported a missing or incompatible native adapter shard."
        )

        hf_path = next(
            (
                adapter_dir / n
                for n in ("adapter_model.safetensors", "adapter_model.bin")
                if (adapter_dir / n).exists()
            ),
            None,
        )
        if hf_path is not None and not shard_found:
            logger.warning(
                f"Found HF PEFT adapter at {hf_path} but direct HF PEFT loading into "
                f"Megatron is not yet supported. Please save using Megatron-native format "
                f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
            )
        return False, None

    assert plan is not None
    training_state = None
    training_state_error: Exception | None = None
    restore_training_state = False
    if optimizer is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        state_path = adapter_dir / f"training_state_rank{rank}.pt"
        training_state, training_state_error = _read_training_state(state_path)
        local_training_ok = training_state_error is None and training_state is not None
        local_iteration = training_state.get("iteration") if training_state is not None else None
        restore_training_state = _all_ranks_agree_on_training_state(local_training_ok, local_iteration)

    loaded = plan.apply()
    for chunk in model:
        chunk._miles_lora_native_checkpoint_loaded = True
    logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")

    if optimizer is None:
        iteration = None
    elif restore_training_state:
        assert training_state is not None
        iteration = _apply_training_state(training_state, optimizer, opt_param_scheduler)
    else:
        iteration = None
        if training_state_error is not None:
            logger.warning("Could not preflight rank-local LoRA training state: %s", training_state_error)
        logger.warning(
            "Skipping optimizer/scheduler restore: at least one rank reported a missing/corrupt "
            "training-state file or a different iteration; all ranks resume with fresh training "
            "state for consistency."
        )
        reload_model_params = getattr(optimizer, "reload_model_params", None)
        if callable(reload_model_params):
            reload_model_params()
    return True, iteration


def _all_ranks_see_dist_checkpoint(local_probe: bool, dist_ckpt_dir: Path) -> bool:
    """Agree across ranks which load path to take before entering a collective.

    The dist-vs-legacy choice comes from a rank-local filesystem probe; if
    shared-FS visibility skews (e.g. attribute-cache lag right after a save),
    ranks would enter mismatched collectives and hang. A partial view is an
    environment fault, so it raises instead.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_probe
    probes: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(probes, local_probe)
    if any(probes) and not all(probes):
        raise RuntimeError(
            f"ranks disagree on whether {dist_ckpt_dir} is a dist checkpoint "
            f"(visible on {sum(bool(p) for p in probes)}/{len(probes)} ranks); "
            f"check shared-filesystem consistency before resuming."
        )
    return all(probes)


def _all_ranks_can_restore_training_state(local_ok: bool) -> bool:
    """Agree across ranks whether to restore optimizer/scheduler state.

    Collective: every rank must call it. Restoring on some ranks only would
    silently desync the resume iteration and LR schedule.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_ok
    decisions: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(decisions, local_ok)
    return all(decisions)


def _all_ranks_agree_on_training_state(local_ok: bool, iteration: int | None) -> bool:
    """Require every rank's training state to parse and name one iteration."""
    local = (local_ok, iteration)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_ok
    states: list[tuple[bool, int | None] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    valid = [state for state in states if state is not None]
    return len(valid) == len(states) and all(ok for ok, _ in valid) and len({step for _, step in valid}) == 1


def _read_training_state(state_path: Path) -> tuple[dict[str, Any] | None, Exception | None]:
    """Parse and minimally validate one rank's optimizer checkpoint."""
    if not state_path.exists():
        return None, FileNotFoundError(f"training-state file is missing: {state_path}")
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"training-state root is {type(state).__name__}, expected dict")
        if "optimizer" not in state:
            raise KeyError("training-state file has no 'optimizer' entry")
        iteration = state.get("iteration")
        if iteration is not None and not isinstance(iteration, int):
            raise TypeError(f"training-state iteration is {type(iteration).__name__}, expected int or None")
        return state, None
    except Exception as error:
        return None, error


def _apply_training_state(
    training_state: dict[str, Any],
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    """Apply a training-state dictionary that passed collective preflight."""
    if optimizer is None:
        return None

    optimizer.load_state_dict(training_state["optimizer"])
    logger.info("Restored optimizer state from LoRA checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from LoRA checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming LoRA training from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# LoRA config dict for weight sync to SGLang
# ---------------------------------------------------------------------------


def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    target_modules_hf: Any = (
        target_modules_hf_for_sglang_rollout(args)
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    if target_modules_hf == ["all"]:
        target_modules_hf = "all-linear"
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules_hf,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
