import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.megatron_utils.lora_utils import (
    build_lora_sync_config,
    is_lora_weight_name,
    lora_base_cpu_backup_enabled,
    lora_rollout_base_retained,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer, monkey_patch_torch_reductions
from .common import _check_weight_sync_results, begin_weight_update, end_weight_update
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


def _should_skip_lora_base_sync(
    *,
    is_lora: bool,
    retains_rollout_base: bool,
    check_weight_update_equal: bool,
    lora_base_synced: bool,
) -> bool:
    return is_lora and retains_rollout_base and (not check_weight_update_equal or lora_base_synced)


def _validate_zero_lora_delta(hf_named_tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
    lora_b_tensors = [(name, tensor) for name, tensor in hf_named_tensors if ".lora_B." in name]
    if not lora_b_tensors:
        raise RuntimeError("Initial Kimi K3 LoRA chunk contains no lora_B tensors")

    for name, tensor in lora_b_tensors:
        nonzero = torch.count_nonzero(tensor).item()
        if nonzero:
            raise RuntimeError(
                f"Initial Kimi K3 LoRA delta is not zero: {name} has {nonzero} "
                f"nonzero values (max_abs={tensor.abs().max().item():.9g})"
            )


class _LoraVersionChangeValidator:
    def __init__(self) -> None:
        self._baseline: dict[str, str] = {}
        self._seen: set[str] = set()
        self._changed = 0

    def validate(
        self,
        weight_version: int,
        checksums: Mapping[str, str],
        *,
        is_first_chunk: bool,
        is_last_chunk: bool,
    ) -> None:
        if is_first_chunk:
            self._seen = set()
            self._changed = 0

        duplicate = self._seen & checksums.keys()
        if duplicate:
            raise RuntimeError(f"LoRA version {weight_version} contains duplicate tensors: {sorted(duplicate)[:5]}")
        self._seen.update(checksums)

        if weight_version == 1:
            self._baseline.update(checksums)
        else:
            unexpected = checksums.keys() - self._baseline.keys()
            if unexpected:
                raise RuntimeError(
                    f"LoRA version {weight_version} contains unexpected tensors: {sorted(unexpected)[:5]}"
                )
            self._changed += sum(self._baseline[name] != checksum for name, checksum in checksums.items())

        if not is_last_chunk:
            return

        if weight_version == 1:
            if not self._baseline:
                raise RuntimeError("Initial LoRA version contains no tensors")
            logger.info("Recorded initial LoRA checksum baseline: %d tensors", len(self._baseline))
            return

        missing = self._baseline.keys() - self._seen
        if missing:
            raise RuntimeError(f"LoRA version {weight_version} is missing tensors: {sorted(missing)[:5]}")
        if self._changed == 0:
            raise RuntimeError(f"LoRA version {weight_version} did not change any exported tensor")
        logger.info(
            "LoRA version %d update verified: %d/%d exported tensors changed",
            weight_version,
            self._changed,
            len(self._baseline),
        )


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict->GPU) -> broadcast PP/EP(GPU NCCL) -> gather TP(GPU NCCL) -> convert HF(GPU) -> send.
    Colocated: GPU->CPU serialize -> gather_object(Gloo CPU) -> Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.is_lora = is_lora
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=self.is_lora,
        )
        if self.is_lora:
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False
            self._lora_base_synced = False
            self._lora_version_change_validator = _LoraVersionChangeValidator()

        # Create IPC gather groups for complete colocated engines. A partial tail
        # of trainer ranks can be reserved as placeholder GPU slots.
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        world_size = dist.get_world_size()
        engine_size = self.args.rollout_num_gpus_per_engine
        for start_rank in range(0, world_size - engine_size + 1, engine_size):
            end_rank = start_rank + engine_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and get_parallel_state().tp.rank == 0
                and get_parallel_state().pp.rank == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, g, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear ``update_weight_metrics``. Empty under colocate today; kept symmetric
        with the distributed updaters so the actor can drain unconditionally."""
        out = self.__dict__.pop("update_weight_metrics", {})
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()

        # LoRA never mutates the base. With either path that retains it on the
        # rollout side (distributed keeps it on GPU; colocate + cpu_backup keeps
        # a host mirror across pause/resume), we can skip the base sync entirely
        # and the surrounding restore_weights_before_load / post_process_quantization
        # calls that would otherwise prep / re-quantize fresh base bytes.
        skip_base_sync = _should_skip_lora_base_sync(
            is_lora=self.is_lora,
            retains_rollout_base=(
                self.use_distribute or lora_base_cpu_backup_enabled(self.args) or lora_rollout_base_retained(self.args)
            ),
            check_weight_update_equal=getattr(self.args, "check_weight_update_equal", False),
            lora_base_synced=self._lora_base_synced if self.is_lora else False,
        )

        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if not skip_base_sync:
                begin_weight_update(self.rollout_engines)
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        if not skip_base_sync:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors

            # SGLang restores packed base weights for the duration of a base
            # update. Close that session before Bridge starts its TP/EP LoRA
            # gathers; keeping both live exceeds colocated full-model memory.
            if rank == 0:
                end_weight_update(self.rollout_engines)
            dist.barrier(group=get_gloo_group())
            if self.is_lora:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        if self.is_lora:
            chunks = iter(self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights, weight_type="lora"))
            try:
                hf_named_tensors = next(chunks)
            except StopIteration:
                raise RuntimeError(
                    "LoRA weight sync failed: the weight iterator produced zero chunks. No adapter weights were sent to the rollout engine. This usually means the Megatron-Bridge or SGLang version is incompatible."
                ) from None

            is_first_chunk = True
            sent_chunks = 0
            for next_hf_named_tensors in chunks:
                refs, long_lived_tensors = self._send_lora_params(
                    hf_named_tensors,
                    is_first_chunk=is_first_chunk,
                    is_last_chunk=False,
                )
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=True)
                del long_lived_tensors
                sent_chunks += 1
                hf_named_tensors = next_hf_named_tensors
                is_first_chunk = False

            refs, long_lived_tensors = self._send_lora_params(
                hf_named_tensors,
                is_first_chunk=is_first_chunk,
                is_last_chunk=True,
            )
            results = ray.get(refs)
            _check_weight_sync_results(results, is_lora=True)
            del long_lived_tensors
            sent_chunks += 1

            if rank == 0:
                logger.info(
                    "LoRA weight version %d sent in %d chunks",
                    self.weight_version,
                    sent_chunks,
                )

            if not self._lora_base_synced:
                self._lora_base_synced = True

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        return refs or [], long_lived_tensors

    def _send_lora_params(
        self,
        hf_named_tensors,
        *,
        is_first_chunk: bool = True,
        is_last_chunk: bool = True,
    ) -> tuple[list[ObjectRef], Any]:
        if not any(is_lora_weight_name(n) for n, _ in hf_named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights (no lora_A/lora_B names found). Check weight iterator configuration."
            )
        if self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")
        else:
            if (
                self.weight_version == 1
                and "kimi_k3" in self.model_name.lower()
                and getattr(self.args, "lora_adapter_path", None) is None
                and getattr(self.args, "check_lora_weight_equal", False)
            ):
                _validate_zero_lora_delta(hf_named_tensors)
            refs, long_lived_tensors = _send_to_colocated_engine(
                hf_named_tensors=hf_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
                lora_config=self._lora_config,
                lora_name=LORA_ADAPTER_NAME,
                lora_loaded=self._lora_loaded,
                lora_is_first_chunk=is_first_chunk,
                lora_is_last_chunk=is_last_chunk,
                check_lora_weight_equal=getattr(self.args, "check_lora_weight_equal", False),
                lora_version_validator=self._lora_version_change_validator.validate,
            )
            if is_last_chunk:
                self._lora_loaded = True
            return refs or [], long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
    lora_config: dict | None = None,
    lora_name: str | None = None,
    lora_loaded: bool = False,
    lora_is_first_chunk: bool = True,
    lora_is_last_chunk: bool = True,
    check_lora_weight_equal: bool = False,
    lora_version_validator: Callable[..., None] | None = None,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_lora = lora_config is not None
    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []
    monkey_patch_torch_reductions()

    if is_lora:
        assert weight_version is not None, "LoRA tensor sync requires a weight version"
        # Every bridge rank has the complete adapter after its internal TP/EP
        # collectives. Keep one CUDA-IPC payload per rank so a multi-node
        # SGLang engine can open handles from its node-local trainer process.
        # Do not flatten here: K3's full adapter is about 24 GB, and torch.cat
        # would temporarily duplicate it on every GPU.
        tensor_dict = dict(hf_named_tensors)
        assert len(tensor_dict) == len(hf_named_tensors), "LoRA adapter contains duplicate HF tensor names"
        long_live_tensors.append(tensor_dict)
        serialized_tensors = [MultiprocessingSerializer.serialize(tensor_dict, output_str=True)]
    elif getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    if not is_lora:
        serialized_tensors = []
        for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": flattened_tensor_bucket.get_metadata(),
            }
            long_live_tensors.append(flattened_tensor_data)
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if is_gather_src:
        if is_lora:
            if lora_is_first_chunk and lora_loaded:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))

            # (Yusheng) to-do-1: update lora weights from tensors should support multiple dtypes (bf16, fp8, fp16, fp32)
            # currently, we only support 1 type. If there are multiple dtypes, we need to serialize the tensors for each dtype.
            # Thus, we need to apply the same way as `ipc_engine.update_weights_from_tensor` in future
            # (Yusheng) to-do-2: need to add ci test acc here - now it will pass but fail to update lora weights

            assert all(
                len(rank_payloads) == 1 for rank_payloads in serialized_named_tensors
            ), "LoRA tensor sync requires one payload per engine rank"

            expected_checksums = None
            if check_lora_weight_equal:
                import hashlib

                expected_checksums = {
                    name: hashlib.sha256(
                        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for name, tensor in hf_named_tensors
                }
                if lora_version_validator is not None:
                    lora_version_validator(
                        weight_version,
                        expected_checksums,
                        is_first_chunk=lora_is_first_chunk,
                        is_last_chunk=lora_is_last_chunk,
                    )

            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_tensors=[rank_payloads[0] for rank_payloads in serialized_named_tensors],
                    is_first_chunk=lora_is_first_chunk,
                    is_last_chunk=lora_is_last_chunk,
                    expected_checksums=expected_checksums,
                )
            )
            if lora_is_last_chunk:
                refs.append(ipc_engine.update_weight_version.remote(weight_version=str(weight_version)))

            probe_name, probe_tensor = next(
                ((name, tensor) for name, tensor in hf_named_tensors if ".lora_B." in name),
                hf_named_tensors[0],
            )
            probe = probe_tensor.reshape(-1)[:4096].float()
            logger.info(
                "LoRA sync version=%s source_rank=%s first=%s last=%s probe_tensor=%s probe_l2=%.9g",
                weight_version,
                dist.get_rank(),
                lora_is_first_chunk,
                lora_is_last_chunk,
                probe_name,
                torch.linalg.vector_norm(probe).item(),
            )

        else:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                }
                refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
