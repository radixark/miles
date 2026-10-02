"""Hybrid NCCL M2N reshard plus broadcast weight updates."""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx
import ray
import torch
import torch.distributed as dist

from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.megatron_utils.parallel import get_expert_data_parallel_rank_and_size
from miles.backends.megatron_utils.sglang import per_block_cast_to_fp8
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.broadcast import (
    UpdateWeightFromDistributed,
    connect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)
from miles.backends.training_utils.weight_update.protocols.nccl_m2n_manifest import (
    _build_manifest,
    _dtype_from_name,
    _fp8_manifest_quantization,
    _fp8_scale_shape,
    _local_source_spec,
    _manifest_digest,
    _split_manifest_by_pp,
    _tensor_bytes,
)
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton

logger = logging.getLogger(__name__)


def _new_m2n_group_name() -> str:
    return f"miles-m2n-{uuid.uuid4().hex}"


def _validated_engine_gpu_counts(
    args: Namespace,
    engine_count: int,
    engine_gpu_counts: Sequence[int] | None,
) -> list[int]:
    expected_size = int(args.rollout_num_gpus_per_engine)
    total_gpus = int(args.rollout_num_gpus)
    if expected_size <= 0 or total_gpus <= 0 or total_gpus % expected_size:
        raise ValueError(
            "NCCL M2N requires positive rollout GPU counts with the total "
            f"divisible by the per-engine count, got total={total_gpus}, per_engine={expected_size}"
        )
    expected_engines = total_gpus // expected_size
    if engine_count != expected_engines:
        raise ValueError(
            f"NCCL M2N expected {expected_engines} rollout engine handles for "
            f"{total_gpus} GPUs at {expected_size} GPUs per engine, got {engine_count}"
        )

    counts = [expected_size] * engine_count if engine_gpu_counts is None else list(engine_gpu_counts)
    if len(counts) != engine_count:
        raise ValueError(
            "NCCL M2N requires one GPU count per rollout engine handle, "
            f"got {engine_count} handles and {len(counts)} counts"
        )
    if counts != [expected_size] * engine_count:
        raise ValueError(f"NCCL M2N requires homogeneous {expected_size}-GPU rollout engines, got {counts}")
    return counts


def _is_unreachable_engine_error(error: Exception) -> bool:
    return isinstance(error, httpx.TransportError)


def _quantize_block_fp8(weight: torch.Tensor, *, scale_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.dtype != torch.bfloat16 or weight.dim() != 3:
        raise ValueError(
            "NCCL M2N FP8 expert sources must be 3-D BF16 tensors, "
            f"got shape={tuple(weight.shape)} dtype={weight.dtype}"
        )
    if scale_format not in ("canonical", "ue8m0_unpacked"):
        raise ValueError(f"Unsupported NCCL M2N scale format {scale_format!r}")
    if scale_format == "ue8m0_unpacked" and per_block_cast_to_fp8 is None:
        raise RuntimeError("NCCL M2N UE8M0 transfers require the trainer's power-of-two FP8 quantizer")
    scale_shape = _fp8_scale_shape(weight.shape, "FP8 expert source")
    flat = weight.contiguous().view(-1, weight.shape[-1])
    # Match broadcast's canonical quantizer selection; UE8M0 always requires
    # power-of-two scales, but leaves inference-layout packing to rollout.
    if scale_format == "ue8m0_unpacked" or (
        os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES") == "0" and per_block_cast_to_fp8 is not None
    ):
        qweight, scale = per_block_cast_to_fp8(flat)
    else:
        qweight, scale = blockwise_cast_to_fp8_triton(flat, [128, 128])
    qweight = qweight.view_as(weight).contiguous()
    scale = scale.view(scale_shape).to(torch.float32).contiguous()
    if qweight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"Block-FP8 quantizer returned unexpected dtype {qweight.dtype}")
    return qweight, scale


def _process_group_options() -> Any:
    options = dist.ProcessGroupNCCL.Options()
    options.config.blocking = 1
    return options


def _nccl_m2n() -> Any:
    try:
        from nccl import m2n
    except Exception as exc:
        raise RuntimeError(
            "NCCL M2N was selected, but its nccl.m2n package or native library " "is unavailable"
        ) from exc
    return m2n


def _warm_and_borrow_nccl_comm(pg: dist.ProcessGroup, device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(f"NCCL M2N requires CUDA, got {device}")
    torch.cuda.set_device(device)
    dist.all_reduce(torch.zeros(1, device=device), group=pg)
    torch.cuda.synchronize(device)
    comm_ptr = int(pg._get_backend(device)._comm_ptr())
    if not comm_ptr:
        raise RuntimeError("ProcessGroupNCCL returned a null communicator pointer")
    return comm_ptr


def _check_engine_results(results: Sequence[Any], operation: str) -> None:
    for result in results:
        if result is None:
            continue
        success = result.get("success") if isinstance(result, Mapping) else getattr(result, "success", None)
        message = result.get("message", "") if isinstance(result, Mapping) else getattr(result, "message", "")
        if success is not True:
            raise RuntimeError(
                f"SGLang {operation} returned an unsuccessful or unsupported " f"response {result!r}: {message}"
            )


def _collect_errors(error: str | None) -> list[str]:
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error, group=get_gloo_group())
    return [item for item in errors if item]


class UpdateWeightFromNcclM2N(UpdateWeightFromDistributed):
    """Reshard FFNs on one M2N communicator per PP stage; broadcast the rest.

    The global manifest negotiates coverage only. Wire manifests rebase each
    stage's source ranks and give rollout a separate native staging namespace.
    """

    supports_lora = False

    def __init__(self, args: Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL M2N requires CUDA in the trainer process")
        _nccl_m2n()
        super().__init__(args)
        self._source_device = torch.device("cuda", torch.cuda.current_device())
        self._m2n_group_name: str | None = None
        self._m2n_pg: dist.ProcessGroup | None = None
        self._m2n_comm_ptr: int | None = None
        self._m2n_manifest: dict[str, Any] | None = None
        self._m2n_stage_manifests: dict[int, dict[str, Any]] = {}
        self._m2n_group_names: dict[int, str] = {}
        self._m2n_comm_rank: int | None = None
        self._m2n_local_tensors: dict[str, torch.Tensor] = {}
        self._m2n_fp8_pair_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._residual_pp_rank = get_parallel_state().pp.rank
        self._residual_group_started = False
        self._deferred_update_error: str | None = None

    def configure_model(self, iterator: HfWeightIteratorBase) -> None:
        self._iterator = iterator
        self.model = iterator.model
        self._m2n_quantization = _fp8_manifest_quantization(iterator.quantization_config)
        if self._m2n_quantization is not None:
            # FP8 M2N entries are routed experts. Share broadcast's backend
            # selection instead of selecting a format from helper availability.
            scale_format = quantizer_fp8._get_scale_format(
                self.args, self._m2n_quantization["weight_block_size"], is_expert=True
            )
            self._m2n_quantization["scale_format"] = "ue8m0_unpacked" if scale_format == "ue8m0" else "canonical"
        global_names = [name for name, _ in named_params_and_buffers(self.args, self.model)]
        local_names = (
            [name for name, _ in named_params_and_buffers(self.args, self.model, convert_to_global_name=False)]
            if self.args.megatron_to_hf_mode == "bridge"
            else global_names
        )
        self._weight_source_names = dict(zip(global_names, local_names, strict=True))

    def before_base_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        self._deferred_update_error = None
        error = None
        try:
            # Offload may rebind parameter storage between updates. Always read
            # the same fresh source mapping as the residual HF iterator.
            self._m2n_local_tensors = {
                name: weights[self._weight_source_names[name]] for name in self._m2n_local_tensors
            }
        except Exception as exc:
            error = f"trainer rank {dist.get_rank()} weight source: {type(exc).__name__}: {exc}"
        failures = _collect_errors(error)
        if failures:
            raise RuntimeError("NCCL M2N source preparation failed: " + " | ".join(failures))
        self._update_bulk_weights()

    def run_engine_session(self, operation: Callable[[], None]) -> None:
        error = None
        try:
            super().run_engine_session(operation)
        except Exception as exc:
            error = f"rollout session: {type(exc).__name__}: {exc}"
        failures = _collect_errors(error)
        if failures:
            raise RuntimeError("NCCL M2N rollout session failed: " + " | ".join(failures))

    @property
    def _is_source(self) -> bool:
        return self.is_sender

    def _trainer_payload(self) -> dict[str, Any]:
        ps = get_parallel_state()
        expert_dp_rank, expert_dp_size = get_expert_data_parallel_rank_and_size()
        local_tensors: dict[str, torch.Tensor] = {}
        specs: list[dict[str, Any]] = []
        update_units: list[list[str]] = []
        for name, tensor in named_params_and_buffers(self.args, self.model):
            update_units.append([name])
            spec = _local_source_spec(name, tensor)
            if spec is not None:
                local_tensors[name] = tensor
                specs.append(spec)
        self._m2n_local_tensors = local_tensors
        return {
            "topology": {
                "world_rank": dist.get_rank(),
                "tp_rank": ps.tp.rank,
                "tp_size": ps.tp.size,
                "pp_rank": ps.pp.rank,
                "pp_size": ps.pp.size,
                "cp_rank": ps.cp.rank,
                "cp_size": ps.cp.size,
                "dense_dp_rank": ps.intra_dp.rank,
                "dense_dp_size": ps.intra_dp.size,
                "ep_rank": ps.ep.rank,
                "ep_size": ps.ep.size,
                "etp_rank": ps.etp.rank,
                "etp_size": ps.etp.size,
                "expert_dp_rank": expert_dp_rank,
                "expert_dp_size": expert_dp_size,
                "independent_dp_rank": ps.indep_dp.rank,
                "independent_dp_size": ps.indep_dp.size,
            },
            "specs": specs,
            "update_units": update_units,
        }

    def _negotiate_manifest(self, engine_gpu_counts: Sequence[int]) -> dict[str, Any]:
        try:
            if (
                self._m2n_quantization is not None
                and self._m2n_quantization["scale_format"] == "ue8m0_unpacked"
                and per_block_cast_to_fp8 is None
            ):
                raise RuntimeError("NCCL M2N FP8 requires the UE8M0 quantizer on every trainer rank")
            record = {"payload": self._trainer_payload(), "quantization": self._m2n_quantization, "error": None}
        except Exception as exc:
            record = {
                "payload": None,
                "error": (f"trainer rank {dist.get_rank()}: " f"{type(exc).__name__}: {exc}"),
            }
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, record, group=get_gloo_group())
        objects: list[dict[str, Any] | None] = [None]
        if dist.get_rank() == 0:
            failures = [item["error"] for item in gathered if item is not None and item["error"] is not None]
            if any(
                item is not None and item["error"] is None and item.get("quantization") != self._m2n_quantization
                for item in gathered
            ):
                failures.append("Trainer ranks selected different NCCL M2N FP8 formats")
            if failures:
                objects[0] = {
                    "manifest": None,
                    "error": " | ".join(failures),
                }
            else:
                payloads = [item["payload"] for item in gathered if item is not None and item["payload"] is not None]
                try:
                    manifest = _build_manifest(
                        payloads,
                        engine_gpu_counts,
                        quantization_config=self._m2n_quantization,
                        destination_ep_size=self.args.sglang_ep_size,
                    )
                    objects[0] = {
                        "manifest": manifest,
                        "error": None,
                    }
                except Exception as exc:
                    objects[0] = {
                        "manifest": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        dist.broadcast_object_list(objects, src=0, group=get_gloo_group())
        result = objects[0]
        if result is None:
            raise RuntimeError("Trainer rank zero did not produce an NCCL M2N manifest")
        if result["error"] is not None:
            raise RuntimeError(f"NCCL M2N manifest negotiation failed: {result['error']}")
        manifest = result["manifest"]
        if manifest is None:
            raise RuntimeError("NCCL M2N manifest negotiation returned no manifest")
        if manifest.get("manifest_hash") != _manifest_digest(manifest):
            raise RuntimeError("NCCL M2N manifest hash validation failed")
        self._m2n_manifest = manifest
        return manifest

    def _teardown_local_m2n(self) -> None:
        if getattr(self, "_m2n_comm_ptr", None) is not None:
            torch.cuda.synchronize()
            _nccl_m2n().finalize()
            self._m2n_comm_ptr = None
        if getattr(self, "_m2n_pg", None) is not None:
            dist.destroy_process_group(self._m2n_pg)
            self._m2n_pg = None
        self._m2n_comm_rank = None

    def _disconnect_existing(
        self,
        replacement_engines: Sequence[SGLangApiClient] | None = None,
    ) -> None:
        old_engines = self.rollout_engines
        if old_engines is None:
            return

        def record_destroy_error(
            errors: list[str],
            engine: SGLangApiClient,
            exc: Exception,
        ) -> None:
            retired = replacement_engines is not None and engine not in replacement_engines
            if retired and _is_unreachable_engine_error(exc):
                logger.info(
                    "Ignoring teardown from an unreachable retired rollout "
                    "engine; it must restart before reuse: %s",
                    exc,
                )
            else:
                errors.append(f"{type(exc).__name__}: {exc}")

        def launch_remote_destroy(
            group_name: str,
            *,
            current_rank: bool = False,
        ) -> tuple[list[tuple[SGLangApiClient, Any]], list[str]]:
            if not current_rank and dist.get_rank() != 0:
                return [], []
            refs: list[tuple[SGLangApiClient, Any]] = []
            errors: list[str] = []
            for engine in old_engines:
                try:
                    refs.append(
                        (
                            engine,
                            async_utils.submit(engine.destroy_weights_update_group(group_name, strict=True)),
                        )
                    )
                except Exception as exc:
                    record_destroy_error(errors, engine, exc)
            return refs, errors

        def finish_remote_destroy(
            refs: Sequence[tuple[SGLangApiClient, Any]],
            errors: list[str],
            operation: str,
        ) -> str | None:
            for engine, ref in refs:
                try:
                    _check_engine_results([ref.result()], operation)
                except Exception as exc:
                    record_destroy_error(errors, engine, exc)
            return " | ".join(errors) or None

        def raise_collective(error: str | None, message: str) -> None:
            failures = _collect_errors(error)
            if failures:
                raise RuntimeError(f"{message}: {' | '.join(failures)}")

        # Dispatch every stage before local teardown: rollout must release all
        # native M2N caches while all of its PP communicators are still alive.
        m2n_refs, m2n_errors = [], []
        for group_name in self._m2n_group_names.values():
            refs, errors = launch_remote_destroy(group_name)
            m2n_refs.extend(refs)
            m2n_errors.extend(errors)
        local_m2n_error: str | None = None
        try:
            self._teardown_local_m2n()
        except Exception as exc:
            local_m2n_error = f"trainer rank {dist.get_rank()} local M2N teardown: " f"{type(exc).__name__}: {exc}"
        m2n_error = finish_remote_destroy(m2n_refs, m2n_errors, "M2N teardown")
        if local_m2n_error is not None:
            m2n_error = local_m2n_error if m2n_error is None else f"{local_m2n_error} | {m2n_error}"
        raise_collective(
            m2n_error,
            "Failed to destroy NCCL M2N PP connections",
        )

        pp_size = getattr(
            getattr(self, "args", None),
            "pipeline_model_parallel_size",
            1,
        )
        for pp_rank in range(pp_size):
            residual_error: str | None = None
            if (
                self._is_source
                and getattr(self, "_residual_pp_rank", 0) == pp_rank
                and self._m2n_manifest is not None
                and (getattr(self, "_residual_group_started", False) or self._model_update_groups is not None)
            ):
                residual_refs, residual_errors = launch_remote_destroy(
                    self.group_name,
                    current_rank=True,
                )
                if self._model_update_groups is not None:
                    try:
                        dist.destroy_process_group(self._model_update_groups)
                        self._model_update_groups = None
                    except Exception as exc:
                        residual_error = (
                            f"trainer rank {dist.get_rank()} local residual teardown: " f"{type(exc).__name__}: {exc}"
                        )
                remote_error = finish_remote_destroy(
                    residual_refs,
                    residual_errors,
                    "residual broadcast teardown",
                )
                if remote_error is not None:
                    residual_error = remote_error if residual_error is None else f"{residual_error} | {remote_error}"
                if residual_error is None:
                    self._residual_group_started = False

            raise_collective(
                residual_error,
                "Failed to destroy the previous residual broadcast connection",
            )

        self._m2n_manifest = None
        self._m2n_group_name = None
        self._m2n_group_names.clear()
        self._m2n_stage_manifests = {}

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        if self.args.sglang_speculative_algorithm and selector != "target":
            raise ValueError(
                "NCCL M2N supports speculation only with a frozen draft and target-only weight updates; "
                "disable trainer MTP layers."
            )
        del engine_gpu_offsets
        engine_gpu_counts = _validated_engine_gpu_counts(
            self.args,
            len(rollout_engines),
            engine_gpu_counts,
        )
        self._disconnect_existing(rollout_engines)

        self.rollout_engines = rollout_engines
        self._selector = selector
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        self._engine_gpu_counts = list(engine_gpu_counts)
        self._residual_pp_rank = 0 if placement.gather_pp else parallel_state.pp.rank
        self.group_name = f"miles-pp_{self._residual_pp_rank}"
        manifest = self._negotiate_manifest(engine_gpu_counts)
        routed_names = {name for unit in manifest["routed_update_units"] for name in unit}
        self._m2n_local_tensors = {
            name: tensor for name, tensor in self._m2n_local_tensors.items() if name in routed_names
        }
        # Direct conversion can skip these native parameters before TP/EP
        # gathers. Bridge still gathers, then filters complete HF units.
        if hasattr(self._iterator, "excluded_native_names"):
            self._iterator.excluded_native_names = routed_names
        hf_names = set()
        for entry in manifest["entries"]:
            if entry["family"] == "routed_expert":
                prefix, suffix = entry["name"].split(".experts.", 1)
                hf_names.update(f"{prefix}.experts.{expert}.{suffix}" for expert in range(entry["global_shape"][0]))
            else:
                hf_names.add(entry["name"])
        self._iterator.excluded_hf_names = hf_names

        self._m2n_stage_manifests = _split_manifest_by_pp(manifest)
        for pp_rank, stage_manifest in self._m2n_stage_manifests.items():
            connection: list[dict[str, Any] | None] = [None]
            rendezvous_world_rank = stage_manifest["source_world_ranks"][0]
            if dist.get_rank() == rendezvous_world_rank:
                master_address = ray._private.services.get_node_ip_address()
                with socket.socket() as sock:
                    sock.bind(("", 0))
                    master_port = sock.getsockname()[1]
                connection[0] = {
                    "master_address": master_address,
                    "master_port": master_port,
                    "group_name": f"{_new_m2n_group_name()}-pp{pp_rank}",
                }
            dist.broadcast_object_list(connection, src=rendezvous_world_rank, group=get_gloo_group())
            if connection[0] is None:
                raise RuntimeError(f"Trainer rank {rendezvous_world_rank} did not publish PP={pp_rank} M2N metadata")
            group_name = connection[0]["group_name"]
            # Record before setup so partially initialized remote groups can be
            # cleaned up if a later stage or receiver fails initialization.
            self._m2n_group_names[pp_rank] = group_name
            refs = []
            local_error: str | None = None
            if dist.get_rank() == 0:
                try:
                    rank_cursor = len(stage_manifest["source_world_ranks"])
                    for engine, count in zip(rollout_engines, engine_gpu_counts, strict=True):
                        refs.append(
                            async_utils.submit(
                                engine.init_weights_update_group(
                                    connection[0]["master_address"],
                                    connection[0]["master_port"],
                                    rank_cursor,
                                    stage_manifest["communicator_world_size"],
                                    group_name,
                                    backend="nccl",
                                    m2n_manifest=stage_manifest,
                                )
                            )
                        )
                        rank_cursor += count
                except Exception as exc:
                    local_error = f"PP={pp_rank} rollout initialization dispatch: {type(exc).__name__}: {exc}"

            comm_rank = stage_manifest["trainer_world_to_comm_rank"].get(str(dist.get_rank()))
            if comm_rank is not None:
                try:
                    self._m2n_group_name = group_name
                    self._m2n_comm_rank = comm_rank
                    device = torch.device("cuda", torch.cuda.current_device())
                    self._m2n_pg = init_process_group(
                        backend="nccl",
                        init_method=f"tcp://{connection[0]['master_address']}:{connection[0]['master_port']}",
                        world_size=stage_manifest["communicator_world_size"],
                        rank=comm_rank,
                        group_name=group_name,
                        pg_options=_process_group_options(),
                    )
                    self._m2n_comm_ptr = _warm_and_borrow_nccl_comm(self._m2n_pg, device)
                except Exception as exc:
                    local_error = f"trainer rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            if dist.get_rank() == 0:
                try:
                    _check_engine_results(async_utils.wait_futures(refs), f"PP={pp_rank} M2N initialization")
                except Exception as exc:
                    remote_error = f"SGLang PP={pp_rank} initialization: {type(exc).__name__}: {exc}"
                    local_error = f"{local_error}; {remote_error}" if local_error else remote_error

            failures = _collect_errors(local_error)
            if failures:
                try:
                    self._disconnect_existing()
                except Exception as exc:
                    logger.warning("Failed to clean up partial M2N PP connections: %s", exc)
                raise RuntimeError(f"NCCL M2N PP={pp_rank} connection setup failed: " + " | ".join(failures))
            if dist.get_rank() == 0:
                logger.info(
                    "NCCL M2N PP=%d group=%s world_size=%d source_world_ranks=%s entries=%d",
                    pp_rank,
                    group_name,
                    stage_manifest["communicator_world_size"],
                    stage_manifest["source_world_ranks"],
                    len(stage_manifest["entries"]),
                )

        local_pp_rank = self._residual_pp_rank
        for pp_rank in range(self.args.pipeline_model_parallel_size):
            residual_error: str | None = None
            if self._is_source and local_pp_rank == pp_rank:
                try:
                    self._residual_group_started = True
                    self._model_update_groups = connect_rollout_engines_from_distributed(
                        self.args,
                        self.group_name,
                        rollout_engines,
                        engine_gpu_counts=engine_gpu_counts,
                    )
                except Exception as exc:
                    residual_error = (
                        f"trainer rank {dist.get_rank()} residual broadcast setup: " f"{type(exc).__name__}: {exc}"
                    )
            residual_failures = _collect_errors(residual_error)
            if residual_failures:
                self._disconnect_existing()
                raise RuntimeError("NCCL M2N residual connection setup failed: " + " | ".join(residual_failures))

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        if self._deferred_update_error is not None:
            bucket.clear()
            return
        broadcast_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in bucket)
        lock_acquired = False
        try:
            self._engine_lock.__enter__()
            lock_acquired = True
            futures = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.rollout_engines,
                bucket,
                selector=self._selector,
            )
            _check_engine_results(async_utils.wait_futures(futures), "residual broadcast")
        except Exception as exc:
            # Keep draining the iterator's TP/EP collectives on every rank.
            self._deferred_update_error = (
                f"trainer rank {dist.get_rank()} residual update: {type(exc).__name__}: {exc}"
            )
        finally:
            if lock_acquired:
                try:
                    self._engine_lock.__exit__(None, None, None)
                except Exception as exc:
                    self._deferred_update_error = (
                        f"{self._deferred_update_error or ''}; residual lock release: {type(exc).__name__}: {exc}"
                    )
            bucket.clear()
        if dist.get_rank() == 0 and self._deferred_update_error is None:
            self.update_weight_metrics["m2n_broadcast_bytes"] += float(broadcast_bytes)

    def after_base_weights(self) -> None:
        failures = _collect_errors(self._deferred_update_error)
        if failures:
            raise RuntimeError("NCCL M2N residual update failed: " + " | ".join(failures))

    def _source_tensor(self, entry: Mapping[str, Any], *, scale_format: str = "canonical") -> torch.Tensor:
        if self._m2n_comm_rank is None:
            raise RuntimeError("Current trainer rank is not in the NCCL M2N communicator")
        source = entry["source"]
        names = source["names_by_rank"].get(str(self._m2n_comm_rank))
        if not names:
            raise RuntimeError(
                f"Manifest entry {entry['name']} has no source recipe for communicator rank " f"{self._m2n_comm_rank}"
            )
        recipe = source["recipe"]
        pair_id = entry.get("pair_id")
        tensor_role = entry.get("tensor_role")
        if pair_id is not None:
            if (
                not isinstance(pair_id, str)
                or tensor_role not in ("weight", "scale")
                or not recipe.startswith("expert_")
            ):
                raise RuntimeError(f"Invalid NCCL M2N FP8 source recipe for {entry['name']}")
            cache = self._m2n_fp8_pair_cache
            if pair_id not in cache:
                base_recipe = recipe.removesuffix("_scale") if tensor_role == "scale" else recipe
                tensors = [
                    self._m2n_local_tensors[name].data.to(self._source_device, non_blocking=True) for name in names
                ]
                logical = self._logical_source_tensor(tensors, base_recipe)
                qweight, scale = _quantize_block_fp8(logical, scale_format=scale_format)
                cache[pair_id] = {"weight": qweight, "scale": scale}
            result = cache[pair_id][tensor_role]
            expected_shape = tuple(source["local_shape"])
            expected_dtype = _dtype_from_name(entry["dtype"])
            if tuple(result.shape) != expected_shape or result.dtype != expected_dtype:
                raise RuntimeError(
                    f"NCCL M2N FP8 source {entry['name']} produced "
                    f"shape={tuple(result.shape)} dtype={result.dtype}; expected "
                    f"shape={expected_shape} dtype={expected_dtype}"
                )
            return result

        tensors = [self._m2n_local_tensors[name].data.to(self._source_device, non_blocking=True) for name in names]
        return self._logical_source_tensor(tensors, recipe)

    @staticmethod
    def _logical_source_tensor(
        tensors: Sequence[torch.Tensor],
        recipe: str,
    ) -> torch.Tensor:
        if recipe.startswith("dense_fc1_"):
            index = int(recipe.rsplit("_", 1)[1])
            return tensors[0].chunk(2, dim=0)[index].contiguous()
        if recipe == "dense_fc2":
            return tensors[0].contiguous()
        if recipe.startswith("expert_fc1_"):
            index = int(recipe.rsplit("_", 1)[1])
            return torch.stack([tensor.chunk(2, dim=0)[index] for tensor in tensors])
        if recipe == "expert_fc2":
            return torch.stack(tensors)
        raise RuntimeError(f"Unknown NCCL M2N source recipe {recipe!r}")

    def _run_m2n_batch(self, manifest: Mapping[str, Any] | None = None) -> None:
        if (
            self._m2n_manifest is None
            or self._m2n_pg is None
            or self._m2n_comm_ptr is None
            or self._m2n_comm_rank is None
        ):
            return
        m2n = _nccl_m2n()
        manifest = self._m2n_manifest if manifest is None else manifest
        stream = torch.cuda.current_stream()

        def placements(descriptors: Sequence[Mapping[str, Any]]) -> list[Any]:
            result = []
            for descriptor in descriptors:
                if descriptor["type"] == "replicate":
                    result.append(m2n.Replicate())
                elif descriptor["type"] == "shard":
                    result.append(m2n.Shard(descriptor["dim"]))
                else:
                    raise RuntimeError(f"Unknown NCCL M2N placement {descriptor!r}")
            return result

        try:
            previous_source_mesh = None
            for entry in manifest["entries"]:
                source_descriptor = entry["source"]
                destination_descriptor = entry["destination"]
                if previous_source_mesh is not None and source_descriptor["mesh"] != previous_source_mesh:
                    # Match the receiver's handoff when dense/expert ownership
                    # changes inside this PP stage's communicator.
                    dist.barrier(group=self._m2n_pg)
                previous_source_mesh = source_descriptor["mesh"]
                source = None
                if any(self._m2n_comm_rank in row for row in source_descriptor["mesh"]):
                    source = self._source_tensor(
                        entry, scale_format=manifest.get("quantization", {}).get("scale_format", "canonical")
                    )
                dtype = _dtype_from_name(entry["dtype"])
                m2n.reshard(
                    source,
                    None,
                    self._m2n_comm_ptr,
                    stream,
                    src_mesh=source_descriptor["mesh"],
                    src_placements=placements(source_descriptor["placements"]),
                    src_local_shape=source_descriptor["local_shape"],
                    src_dtype=dtype,
                    dst_mesh=destination_descriptor["mesh"],
                    dst_placements=placements(destination_descriptor["placements"]),
                    dst_local_shape=destination_descriptor["local_shape"],
                    dst_dtype=dtype,
                )
                stream.synchronize()
                tensor_role = entry.get("tensor_role")
                if tensor_role is not None:
                    pair = self._m2n_fp8_pair_cache.get(entry["pair_id"])
                    if pair is not None:
                        pair.pop(tensor_role, None)
                        if not pair:
                            self._m2n_fp8_pair_cache.pop(entry["pair_id"])
        finally:
            self._m2n_fp8_pair_cache.clear()

    def _update_m2n_stage(self, pp_rank: int, manifest: Mapping[str, Any]) -> None:
        self._update_m2n_stages({pp_rank: manifest})

    def _update_m2n_stages(self, stages: Mapping[int, Mapping[str, Any]]) -> None:
        """Run one bounded wave; every trainer joins only its own PP group."""
        group_names = [self._m2n_group_names[pp_rank] for pp_rank in stages]
        entries = [entry for manifest in stages.values() for entry in manifest["entries"]]
        stage_label = ",".join(str(pp_rank) for pp_rank in stages)
        # A single scheduler request must drive every receiver in this wave.
        # Independent HTTP requests serialize inside SGLang's update lock.
        batch_kwargs = {"m2n_group_names": group_names} if len(group_names) > 1 else {}
        refs = []
        startup_error: str | None = None
        if dist.get_rank() == 0:
            try:
                for engine in self.rollout_engines:
                    refs.append(
                        async_utils.submit(
                            engine.update_weights_from_distributed(
                                names=[entry["name"] for entry in entries],
                                dtypes=[_dtype_from_name(entry["dtype"]) for entry in entries],
                                shapes=[entry["global_shape"] for entry in entries],
                                group_name=group_names[0],
                                load_format="nccl_m2n",
                                selector=self._selector,
                                **batch_kwargs,
                            )
                        )
                    )
            except Exception as exc:
                startup_error = f"trainer rank 0 PP={stage_label} M2N startup: {type(exc).__name__}: {exc}"
        startup_failures = _collect_errors(startup_error)
        if startup_failures:
            raise RuntimeError("NCCL M2N update startup failed: " + " | ".join(startup_failures))

        local_error: str | None = None
        # All selected stages start together, without a trainer-wide collective
        # between them. A trainer belongs to exactly one stage communicator.
        for pp_rank, manifest in stages.items():
            if self._m2n_group_name == self._m2n_group_names[pp_rank]:
                try:
                    self._run_m2n_batch(manifest)
                except Exception as exc:
                    local_error = f"trainer rank {dist.get_rank()} M2N transfer: {type(exc).__name__}: {exc}"
                break
        if dist.get_rank() == 0:
            try:
                _check_engine_results(async_utils.wait_futures(refs), f"PP={stage_label} M2N weight update")
            except Exception as exc:
                remote_error = f"SGLang M2N transfer: {type(exc).__name__}: {exc}"
                local_error = f"{local_error}; {remote_error}" if local_error else remote_error
        failures = _collect_errors(local_error)
        if failures:
            raise RuntimeError(f"NCCL M2N PP={stage_label} weight update failed: " + " | ".join(failures))

    def _update_bulk_weights(self) -> bool:
        if self._m2n_manifest is None or not self._m2n_group_names:
            raise RuntimeError("NCCL M2N updater is not connected")
        concurrency = getattr(self.args, "m2n_pp_concurrency", 2)
        if not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("--m2n-pp-concurrency must be a positive integer")
        lock_acquired = False
        local_error: str | None = None
        if dist.get_rank() == 0:
            try:
                self._engine_lock.__enter__()
                lock_acquired = True
            except Exception as exc:
                local_error = f"rollout-engine lock acquire: {type(exc).__name__}: {exc}"
        startup_failures = _collect_errors(local_error)
        try:
            if startup_failures:
                raise RuntimeError("NCCL M2N update startup failed: " + " | ".join(startup_failures))
            stages = sorted(self._m2n_stage_manifests.items())
            for start in range(0, len(stages), concurrency):
                wave = dict(stages[start : start + concurrency])
                started = time.perf_counter()
                if len(wave) == 1:
                    pp_rank, manifest = next(iter(wave.items()))
                    self._update_m2n_stage(pp_rank, manifest)
                else:
                    self._update_m2n_stages(wave)
                if dist.get_rank() == 0:
                    logger.info(
                        "NCCL M2N PP wave=%s concurrency=%d elapsed=%.3fs",
                        list(wave),
                        concurrency,
                        time.perf_counter() - started,
                    )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        finally:
            if dist.get_rank() == 0 and lock_acquired:
                try:
                    self._engine_lock.__exit__(None, None, None)
                except Exception as exc:
                    release_error = f"rollout-engine lock release: {type(exc).__name__}: {exc}"
                    local_error = f"{local_error}; {release_error}" if local_error else release_error
        failures = _collect_errors(local_error)
        if failures:
            raise RuntimeError("NCCL M2N weight update failed: " + " | ".join(failures))

        if dist.get_rank() == 0:
            self.update_weight_metrics = {
                "m2n_staging_bytes": float(
                    sum(
                        _tensor_bytes(entry["global_shape"], entry["dtype"]) for entry in self._m2n_manifest["entries"]
                    )
                ),
                "m2n_broadcast_bytes": 0.0,
            }
        return True
