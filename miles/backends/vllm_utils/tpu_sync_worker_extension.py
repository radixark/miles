"""vLLM worker extension for receiving trainer weights over tpu-sync (Raiden).

Loaded into each vLLM ``mp`` TPU worker via:
``--worker-extension-cls miles.backends.vllm_utils.tpu_sync_worker_extension.VllmTpuSyncWorkerExtension``

Staging buffers are allocated in TP-sharded bfloat16 layout on the worker's device. On
``install_tpu_sync_weights()``, buffers are moved H2D via ``ws.h2d()``, fused into vLLM's
packed projections (``qkv_proj``, ``gate_up_proj``), transposed when ``vllm-torchtpu`` has
marked the target module with ``_tpu_weight_flipped=True``, and copied into live model
parameters (handling tied ``embed_tokens`` -> ``lm_head`` when ``lm_head`` is omitted).
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

try:
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
except ImportError:
    # vllm-torchtpu owns this name; the literal only serves processes (CPU tests) without it installed
    WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"

_ROW_PARALLEL_SUFFIXES: tuple[str, ...] = (
    ".o_proj.weight",
    ".down_proj.weight",
)
_KV_PROJ_SUFFIXES: tuple[str, ...] = (
    ".k_proj.weight",
    ".v_proj.weight",
)
# VocabParallelEmbedding / ParallelLMHead shard the vocab padded to a multiple of 64, so an even split of the raw
# HF rows lands on other boundaries; these are received whole and sliced with the layer's own shard indices
_VOCAB_PARALLEL_NAMES: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "lm_head.weight",
)
_TIED_LM_HEAD = "lm_head.weight"
_EMBED_TOKENS = "model.embed_tokens.weight"

# Maps packed vLLM projection names to the ordered HF sub-projections concatenated along dim 0.
_FUSED_PROJECTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("gate_up_proj", ("gate_proj", "up_proj")),
)


@dataclass(frozen=True)
class ReceiverSharding:
    """TP sharding specification for one receiver staging tensor registered with tpu-sync."""

    local_shape: list[int]
    mesh_shape: list[int]
    sharding_spec: list[str]
    global_shard_indices: list[int] | None


def compute_receiver_sharding(
    name: str,
    full_shape: Sequence[int],
    *,
    tp_rank: int,
    tp_size: int,
    num_kv_heads: int | None = None,
) -> ReceiverSharding:
    """Compute receiver TP sharding for an HF parameter name and global shape.

    Follows tpu-sync invariants:
    - 1D (and 0D) tensors are always registered whole (``mesh_shape=[1]``).
    - Vocab-parallel embeddings / LM heads are registered whole and sliced on install.
    - ``.o_proj.weight`` / ``.down_proj.weight`` shard dim 1 (``["", "tp"]``).
    - GQA ``.k_proj.weight`` / ``.v_proj.weight`` with ``tp_size > num_kv_heads`` shard across
      ``num_kv_heads`` with ``global_shard_indices=[tp_rank // (tp_size // num_kv_heads)]``.
    - Other 2D+ tensors shard dim 0 (``["tp", ""]``) when evenly divisible by ``tp_size``.
    """
    shape = [int(d) for d in full_shape]
    ndim = len(shape)

    if ndim <= 1 or tp_size <= 1 or name in _VOCAB_PARALLEL_NAMES:
        return ReceiverSharding(
            local_shape=list(shape),
            mesh_shape=[1] * ndim,
            sharding_spec=[""] * ndim,
            global_shard_indices=[0] if ndim > 0 else None,
        )

    if ndim == 2 and name.endswith(_ROW_PARALLEL_SUFFIXES) and shape[1] % tp_size == 0:
        return ReceiverSharding(
            local_shape=[shape[0], shape[1] // tp_size],
            mesh_shape=[1, tp_size],
            sharding_spec=["", "tp"],
            global_shard_indices=[tp_rank],
        )

    if (
        ndim == 2
        and name.endswith(_KV_PROJ_SUFFIXES)
        and num_kv_heads is not None
        and num_kv_heads > 0
        and tp_size > num_kv_heads
        and tp_size % num_kv_heads == 0
        and shape[0] % num_kv_heads == 0
    ):
        kv_shard_idx = tp_rank // (tp_size // num_kv_heads)
        return ReceiverSharding(
            local_shape=[shape[0] // num_kv_heads, shape[1]],
            mesh_shape=[num_kv_heads, 1],
            sharding_spec=["tp", ""],
            global_shard_indices=[kv_shard_idx],
        )

    if shape[0] % tp_size == 0:
        return ReceiverSharding(
            local_shape=[shape[0] // tp_size, *shape[1:]],
            mesh_shape=[tp_size] + [1] * (ndim - 1),
            sharding_spec=["tp"] + [""] * (ndim - 1),
            global_shard_indices=[tp_rank],
        )

    return ReceiverSharding(
        local_shape=list(shape),
        mesh_shape=[1] * ndim,
        sharding_spec=[""] * ndim,
        global_shard_indices=[0],
    )


def _slice_1d_for_tp(
    tensor: torch.Tensor,
    *,
    proj_name: str,
    tp_rank: int,
    tp_size: int,
    num_kv_heads: int | None,
) -> torch.Tensor:
    """Slice a full-replicated 1D bias tensor to this TP rank's local shard when needed."""
    if tp_size <= 1 or tensor.ndim != 1:
        return tensor
    if (
        proj_name in ("k_proj", "v_proj")
        and num_kv_heads is not None
        and num_kv_heads > 0
        and tp_size > num_kv_heads
        and tp_size % num_kv_heads == 0
        and tensor.shape[0] % num_kv_heads == 0
    ):
        shard_size = tensor.shape[0] // num_kv_heads
        shard_idx = tp_rank // (tp_size // num_kv_heads)
        return tensor[shard_idx * shard_size : (shard_idx + 1) * shard_size]
    if tensor.shape[0] % tp_size == 0:
        shard_size = tensor.shape[0] // tp_size
        return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size]
    return tensor


def _is_flipped(module: torch.nn.Module | None, param: torch.nn.Parameter | None = None) -> bool:
    """Whether vllm-torchtpu stores this linear's 2D weight transposed as (in, out)."""
    # vllm-torchtpu sets the flag on the layer (quantization/unquantized.py); the parameter is checked too
    return any(
        bool(getattr(obj, WEIGHT_FLIPPED_ATTR, False))  # config-access-exempt: vllm-torchtpu's flip marker
        for obj in (module, param)
        if obj is not None
    )


def _adapt_for_target(
    tensor: torch.Tensor, parent_module: torch.nn.Module | None, param: torch.nn.Parameter | None = None
) -> torch.Tensor:
    """Transpose a 2D HF (out, in) weight to (in, out) if vllm-torchtpu flipped the target linear."""
    if tensor.ndim == 2 and _is_flipped(parent_module, param):
        return tensor.t().contiguous()
    return tensor


def _copy_into_param(target: torch.nn.Parameter, source: torch.Tensor, *, param_name: str) -> None:
    if target.shape != source.shape:
        raise ValueError(
            f"Shape mismatch installing {param_name!r}: received {tuple(source.shape)} vs model {tuple(target.shape)}"
        )
    target.data.copy_(source.to(dtype=target.dtype, device=target.device))


def _install_vocab_parallel(
    target: torch.nn.Parameter, source: torch.Tensor, module: torch.nn.Module | None, *, param_name: str
) -> None:
    """Copy this rank's rows of a whole HF [vocab, hidden] tensor into a vocab-parallel layer's padded shard."""
    flipped = _is_flipped(module, target)
    target_rows = target.shape[1] if flipped else target.shape[0]
    if target_rows == source.shape[0]:
        rows = source
    else:
        shard_indices = getattr(
            module, "shard_indices", None
        )  # config-access-exempt: vLLM VocabParallelEmbedding exposes its org vocab range as shard_indices
        if shard_indices is None:
            raise ValueError(
                f"{param_name!r} holds {target_rows} vocab rows but received {source.shape[0]}, and its layer "
                f"exposes no shard_indices to say which rows this rank owns"
            )
        rows = source[shard_indices.org_vocab_start_index : shard_indices.org_vocab_end_index]
        assert rows.shape[0] <= target_rows, (
            f"{param_name!r}: this rank owns {rows.shape[0]} vocab rows but its shard has {target_rows}; "
            f"the vocab padding differs from what vLLM allocated"
        )
    rows = rows.to(dtype=target.dtype, device=target.device)
    # rows past the real vocab are vLLM's padding, which no token id indexes; they keep their values
    if flipped:
        target.data[:, : rows.shape[0]].copy_(rows.t())
    else:
        target.data[: rows.shape[0]].copy_(rows)


def _find_fused_install_groups(
    staging_buffers: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.nn.Parameter],
) -> tuple[list[tuple[str, str, tuple[str, ...], list[str]]], set[str]]:
    """Identify HF staging keys that fuse into vLLM packed modules (qkv_proj, gate_up_proj)."""
    groups: list[tuple[str, str, tuple[str, ...], list[str]]] = []
    consumed: set[str] = set()

    for packed_name, sub_names in _FUSED_PROJECTION_GROUPS:
        lead_marker = f".{sub_names[0]}."
        for key in staging_buffers:
            if lead_marker not in key:
                continue
            prefix, suffix = key.rsplit(lead_marker, 1)
            target_name = f"{prefix}.{packed_name}.{suffix}"
            if target_name not in named_params:
                continue
            member_keys = [f"{prefix}.{sub}.{suffix}" for sub in sub_names]
            missing = [member_key for member_key in member_keys if member_key not in staging_buffers]
            if missing:
                raise ValueError(f"{target_name!r} fuses {member_keys}, but the trainer sent none of {missing}")
            groups.append((target_name, f"{prefix}.{packed_name}", sub_names, member_keys))
            consumed.update(member_keys)

    return groups, consumed


def _assert_every_buffer_has_a_target(
    staging_buffers: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.nn.Parameter],
    consumed_keys: set[str],
) -> None:
    """Refuse a sync that would leave received weights uninstalled (naming drift, an unfused layout, ...)."""
    unmatched = sorted(name for name in staging_buffers if name not in consumed_keys and name not in named_params)
    # vLLM drops a tied lm_head from named_parameters(); embed_tokens carries those weights
    if _TIED_LM_HEAD in unmatched and _EMBED_TOKENS in named_params:
        unmatched.remove(_TIED_LM_HEAD)
    if unmatched:
        raise ValueError(
            f"tpu-sync received weights with no matching vLLM parameter, so they would never update: {unmatched}"
        )


def _install_fused_group(
    named_params: Mapping[str, torch.nn.Parameter],
    named_modules: Mapping[str, torch.nn.Module],
    staging_buffers: Mapping[str, torch.Tensor],
    group: tuple[str, str, tuple[str, ...], list[str]],
    *,
    tp_rank: int,
    tp_size: int,
    num_kv_heads: int | None,
) -> None:
    target_name, parent_name, sub_names, member_keys = group
    target_param = named_params[target_name]
    parts = [staging_buffers[k] for k in member_keys]
    if parts[0].ndim == 1 and sum(p.shape[0] for p in parts) != target_param.shape[0]:
        parts = [
            _slice_1d_for_tp(p, proj_name=sub, tp_rank=tp_rank, tp_size=tp_size, num_kv_heads=num_kv_heads)
            for p, sub in zip(parts, sub_names, strict=True)
        ]
    adapted = _adapt_for_target(torch.cat(parts, dim=0), named_modules.get(parent_name), target_param)
    _copy_into_param(target_param, adapted, param_name=target_name)


def _install_single(
    name: str,
    buf: torch.Tensor,
    target_param: torch.nn.Parameter,
    named_modules: Mapping[str, torch.nn.Module],
    *,
    tp_rank: int,
    tp_size: int,
    num_kv_heads: int | None,
) -> None:
    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
    module = named_modules.get(parent_name)
    if name in _VOCAB_PARALLEL_NAMES:
        _install_vocab_parallel(target_param, buf, module, param_name=name)
        return
    tensor = buf
    if tensor.ndim == 1 and tensor.shape[0] != target_param.shape[0] and tp_size > 1:
        proj_name = parent_name.rsplit(".", 1)[-1]
        tensor = _slice_1d_for_tp(
            tensor, proj_name=proj_name, tp_rank=tp_rank, tp_size=tp_size, num_kv_heads=num_kv_heads
        )
    _copy_into_param(target_param, _adapt_for_target(tensor, module, target_param), param_name=name)


@torch.no_grad()
def install_weights_into_model(
    model: torch.nn.Module,
    staging_buffers: Mapping[str, torch.Tensor],
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    num_kv_heads: int | None = None,
) -> int:
    """Copy staged HF-named weights into a vLLM model, fusing QKV/gate-up and handling flips/ties."""
    named_params = dict(model.named_parameters())
    named_modules = dict(model.named_modules())
    fused_groups, consumed_keys = _find_fused_install_groups(staging_buffers, named_params)
    _assert_every_buffer_has_a_target(staging_buffers, named_params, consumed_keys)
    shard = dict(tp_rank=tp_rank, tp_size=tp_size, num_kv_heads=num_kv_heads)

    for group in fused_groups:
        _install_fused_group(named_params, named_modules, staging_buffers, group, **shard)

    singles = [name for name in staging_buffers if name not in consumed_keys and name in named_params]
    for name in singles:
        _install_single(name, staging_buffers[name], named_params[name], named_modules, **shard)
    updated = len(fused_groups) + len(singles)

    # Tied embeddings kept as a separate lm_head tensor: the trainer omits lm_head, so embed_tokens fills it.
    if _TIED_LM_HEAD in named_params and _TIED_LM_HEAD not in staging_buffers and _EMBED_TOKENS in staging_buffers:
        lm_head_param = named_params[_TIED_LM_HEAD]
        embed_param = named_params.get(_EMBED_TOKENS)
        if embed_param is None or lm_head_param.data_ptr() != embed_param.data_ptr():
            _install_vocab_parallel(
                lm_head_param, staging_buffers[_EMBED_TOKENS], named_modules.get("lm_head"), param_name=_TIED_LM_HEAD
            )
            updated += 1

    if torch.accelerator.is_available():
        torch.accelerator.synchronize()

    return updated


def compute_first_linear_weight_checksum(model: torch.nn.Module) -> float:
    """L2 norm of the first decoder projection weight (debug checksum for E2E verification).

    A projection rather than the first 2D parameter, which is the embedding: it is the part of the model a sync that
    misses the decoder layers would leave unchanged.
    """
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    params = list(model.named_parameters())
    for name, param in params:
        if param.ndim == 2 and name.endswith("proj.weight"):
            return float(param.detach().float().norm().item())
    for _, param in params:
        if param.ndim == 2:
            return float(param.detach().float().norm().item())
    return 0.0


def _resolve_bind_ip(controller_address: str) -> str:
    host = controller_address.rsplit(":", 1)[0].strip("[]")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((host, 80))
            ip = sock.getsockname()[0]
            if ip:
                return ip
    except OSError:
        pass
    return os.environ.get("MILES_HOST_IP") or socket.gethostbyname(socket.gethostname())


def _build_receiver_variable_proto(
    index: int,
    name: str,
    full_shape: list[int],
    buf: torch.Tensor,
    sharding: ReceiverSharding,
):
    # Deferred: tpu_sync is a TPU-only dependency absent on CUDA/CPU hosts.
    from tpu_sync.rpc import raiden_service_pb2

    kwargs: dict[str, Any] = {
        "name": name,
        "shape": full_shape,
        "layout": list(range(len(full_shape) - 1, -1, -1)),
        "item_size": buf.element_size(),
        "layer_idx": index,
        "mesh_shape": sharding.mesh_shape,
        "sharding_spec": sharding.sharding_spec,
    }
    if sharding.global_shard_indices is not None:
        kwargs["global_shard_indices"] = sharding.global_shard_indices
    return raiden_service_pb2.VariableMetadataProto(**kwargs)


def _get_tp_rank_and_size() -> tuple[int, int]:
    # Deferred: vllm is an optional rollout engine dependency.
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())


def _get_num_kv_heads(worker: Any, model: torch.nn.Module) -> int | None:
    vllm_config = getattr(
        worker, "vllm_config", None
    )  # config-access-exempt: vLLM worker mixin reads external VllmConfig
    model_config = getattr(
        vllm_config, "model_config", None
    )  # config-access-exempt: vLLM worker mixin reads external ModelConfig
    hf_config = getattr(
        model_config, "hf_config", None
    )  # config-access-exempt: vLLM worker mixin reads external HF config
    if hf_config is None:
        hf_config = getattr(
            model, "config", None
        )  # config-access-exempt: model may expose external HF config directly
    if hf_config is None:
        return None
    get_text_config = getattr(
        hf_config, "get_text_config", None
    )  # config-access-exempt: transformers multimodal configs expose get_text_config()
    if callable(get_text_config):
        hf_config = get_text_config()
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", None
    )  # config-access-exempt: external HF config field varies by architecture
    if num_kv_heads is None:
        num_kv_heads = getattr(
            hf_config, "num_attention_heads", None
        )  # config-access-exempt: MHA models omit num_key_value_heads
    return int(num_kv_heads) if num_kv_heads is not None else None


def _resolve_staging_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    if torch.accelerator.is_available():
        from miles.platforms import current_platform

        return current_platform.current_device()
    return torch.device("cpu")


class VllmTpuSyncWorkerExtension:
    """vLLM worker extension mixin receiving TP-sharded weights via tpu-sync."""

    def start_weight_update(self) -> None:
        """No-op hook so vLLM's POST /start_weight_update succeeds on TPUWorker."""

    def finish_weight_update(self) -> None:
        """No-op hook so vLLM's POST /finish_weight_update succeeds on TPUWorker."""

    def reset_encoder_cache(self) -> None:
        """Hook EngineCore calls on every worker when /pause clears caches; TPUWorker does not define it.

        Without it each worker raises, and MultiprocExecutor leaves the other workers' error replies queued, so every
        later collective RPC (including the tpu-sync install) dequeues a stale error until the engine restarts.
        """
        reset = getattr(
            self.model_runner, "reset_encoder_cache", None
        )  # config-access-exempt: only multimodal model runners keep an encoder cache
        if callable(reset):
            reset()

    def init_tpu_sync_on_worker(
        self,
        controller_address: str,
        job_name: str,
        weight_shapes: dict[str, list[int]] | str,
        parallelism: int | str = 8,
    ) -> dict[str, Any]:
        """Allocate TP-sharded bf16 staging buffers, start WeightSynchronizer, and register with Raiden."""
        if isinstance(weight_shapes, str):
            weight_shapes = json.loads(weight_shapes)
        parallelism_int = int(parallelism)

        model = self.model_runner.model
        tp_rank, tp_size = _get_tp_rank_and_size()
        num_kv_heads = _get_num_kv_heads(self, model)
        device = _resolve_staging_device(model)

        self._shutdown_tpu_sync_ws()

        staging_buffers: dict[str, torch.Tensor] = {}
        variable_protos: list[Any] = []
        for idx, (name, raw_shape) in enumerate(weight_shapes.items()):
            full_shape = [int(d) for d in raw_shape]
            sharding = compute_receiver_sharding(
                name,
                full_shape,
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
            )
            buf = torch.empty(sharding.local_shape, dtype=torch.bfloat16, device=device)
            staging_buffers[name] = buf
            variable_protos.append(_build_receiver_variable_proto(idx, name, full_shape, buf, sharding))

        bind_ip = _resolve_bind_ip(controller_address)
        # Deferred: tpu_sync is a TPU-only dependency absent on CUDA/CPU hosts.
        from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer
        from tpu_sync.rpc import raiden_controller

        ws = WeightSynchronizer(
            [[buf] for buf in staging_buffers.values()],
            local_port=0,
            listener_port=0,
            parallelism=parallelism_int,
            bind_ip=bind_ip,
            unsafe_skip_buffer_lock=True,
            auto_h2d=False,
        )
        raiden_controller.RaidenControllerClientFacade(controller_address).register_work_unit(
            raiden_controller.RaidenId(job_name=job_name, job_replica_id=str(tp_rank), data_name="weights"),
            [f"{bind_ip}:{ws.local_port}"],
            f"{bind_ip}:{ws.listener_port}",
            mesh_shape=[1, tp_size],
            variables=variable_protos,
            mesh_axes=["fsdp", "tp"],
        )

        self._tpu_sync_ws = ws
        self._tpu_sync_buffers = staging_buffers
        self._tpu_sync_tp_rank = tp_rank
        self._tpu_sync_tp_size = tp_size
        self._tpu_sync_num_kv_heads = num_kv_heads

        return {
            "job_name": job_name,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "num_tensors": len(staging_buffers),
        }

    def install_tpu_sync_weights(self) -> dict[str, Any]:
        """Transfer received DRAM weights H2D and install into live vLLM model parameters."""
        ws = getattr(
            self, "_tpu_sync_ws", None
        )  # config-access-exempt: vLLM worker extension state is lazily initialized
        if ws is None:
            raise RuntimeError("tpu-sync receiver not initialized on worker; call init_tpu_sync_on_worker first")

        ws.h2d()
        updated = install_weights_into_model(
            self.model_runner.model,
            self._tpu_sync_buffers,
            tp_rank=self._tpu_sync_tp_rank,
            tp_size=self._tpu_sync_tp_size,
            num_kv_heads=self._tpu_sync_num_kv_heads,
        )
        return {"updated": updated, "tp_rank": self._tpu_sync_tp_rank}

    def get_weight_checksum(self) -> float:
        """Return float L2 norm of the first linear weight on this worker."""
        return compute_first_linear_weight_checksum(self.model_runner.model)

    def _shutdown_tpu_sync_ws(self) -> None:
        ws = getattr(
            self, "_tpu_sync_ws", None
        )  # config-access-exempt: vLLM worker extension state is lazily initialized
        if ws is None:
            return
        self._tpu_sync_ws = None
        shutdown = getattr(ws, "shutdown", None)  # config-access-exempt: external WeightSynchronizer cleanup hook
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.warning("WeightSynchronizer.shutdown() failed on vLLM worker", exc_info=True)
