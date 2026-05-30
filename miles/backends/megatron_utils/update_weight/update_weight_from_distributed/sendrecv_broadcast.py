from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

from ..common import post_process_weights
from .broadcast_utils import (
    acquire_rollout_engine_lock,
    connect_rollout_relay_from_distributed,
    disconnect_rollout_relay_from_distributed,
    update_weights_from_distributed_send_recv,
)
from .mixin import DistBucketedWeightUpdateMixin


@dataclass(frozen=True)
class _FanoutGroupInfo:
    master_address: str
    ports: str
    group_name: str


@ray.remote(num_cpus=0)
def _update_relay_and_resume_engines(
    rollout_engines: Sequence[ActorHandle],
    rollout_engine_lock: ActorHandle,
    relay_engine: ActorHandle,
    peer_engines: Sequence[ActorHandle],
    relay_gpu_count: int,
    next_fanout_port: int,
    fanout_group_info: _FanoutGroupInfo | None,
    fanout_group_name: str,
    weight_version: int,
    relay_receive_refs: Sequence[ray.ObjectRef],
) -> dict[str, object]:
    _raise_sglang_rpc_failures(
        ray.get(list(relay_receive_refs)),
        "receive relay weights from NCCL",
    )
    post_process_weights(
        rollout_engines=[relay_engine],
        restore_weights_before_load=False,
        post_process_quantization=False,
        post_load_weights=False,
    )
    next_fanout_port, fanout_group_info = _sync_relay_weights_to_peer_instances(
        rollout_engine_lock=rollout_engine_lock,
        relay_engine=relay_engine,
        peer_engines=peer_engines,
        relay_gpu_count=relay_gpu_count,
        next_fanout_port=next_fanout_port,
        fanout_group_info=fanout_group_info,
        fanout_group_name=fanout_group_name,
    )
    post_process_weights(
        rollout_engines=rollout_engines,
        restore_weights_before_load=False,
        post_process_quantization=True,
        post_load_weights=True,
    )
    ray.get([engine.update_weight_version.remote(weight_version=str(weight_version)) for engine in rollout_engines])
    ray.get([engine.continue_generation.remote() for engine in rollout_engines])
    return {
        "next_fanout_port": next_fanout_port,
        "fanout_group_info": fanout_group_info,
    }


@ray.remote(num_cpus=0)
class _SendRecvBroadcastCoordinator:
    def __init__(self) -> None:
        self._relay_receive_refs_by_version: dict[int, list[ray.ObjectRef]] = {}
        self._completed_pp_sources_by_version: dict[int, set[int]] = {}
        self._pending_fanout_ref: ray.ObjectRef | None = None
        self._terminal_error: str | None = None
        self._fanout_group_info: _FanoutGroupInfo | None = None

    def configure(
        self,
        *,
        expected_sources: int,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        relay_engine: ActorHandle,
        peer_engines: Sequence[ActorHandle],
        relay_gpu_count: int,
        next_fanout_port: int,
        fanout_group_name: str,
    ) -> None:
        if self._fanout_group_info is not None:
            _destroy_relay_fanout_group(
                relay_engine=self._relay_engine,
                peer_engines=self._peer_engines,
                fanout_group_info=self._fanout_group_info,
            )
            self._fanout_group_info = None

        self._expected_sources = expected_sources
        self._rollout_engines = rollout_engines
        self._rollout_engine_lock = rollout_engine_lock
        self._relay_engine = relay_engine
        self._peer_engines = peer_engines
        self._relay_gpu_count = relay_gpu_count
        self._next_fanout_port = next_fanout_port
        self._fanout_group_name = fanout_group_name
        self._fanout_group_info = None

    def finish_pp_stage_and_maybe_fanout(
        self,
        *,
        weight_version: int,
        pp_rank: int,
        relay_receive_refs: Sequence[ray.ObjectRef],
    ) -> dict[str, int | bool]:
        self._relay_receive_refs_by_version.setdefault(weight_version, []).extend(relay_receive_refs)
        completed_pp_sources = self._completed_pp_sources_by_version.setdefault(weight_version, set())
        completed_pp_sources.add(pp_rank)
        if len(completed_pp_sources) != self._expected_sources:
            return {"scheduled": False, "transfers": 0}

        relay_receive_refs = self._relay_receive_refs_by_version.pop(weight_version, [])
        self._completed_pp_sources_by_version.pop(weight_version, None)
        self._pending_fanout_ref = _update_relay_and_resume_engines.remote(
            rollout_engines=self._rollout_engines,
            rollout_engine_lock=self._rollout_engine_lock,
            relay_engine=self._relay_engine,
            peer_engines=self._peer_engines,
            relay_gpu_count=self._relay_gpu_count,
            next_fanout_port=self._next_fanout_port,
            fanout_group_info=self._fanout_group_info,
            fanout_group_name=self._fanout_group_name,
            weight_version=weight_version,
            relay_receive_refs=relay_receive_refs,
        )
        return {"scheduled": True, "transfers": len(relay_receive_refs)}

    def wait_pending_fanout(self) -> dict[str, object]:
        if self._terminal_error is not None:
            raise RuntimeError("sendrecv_broadcast background update failed: " f"{self._terminal_error}")
        if self._pending_fanout_ref is None:
            return {
                "next_fanout_port": self._next_fanout_port,
                "fanout_group_info": self._fanout_group_info,
            }
        try:
            result = ray.get(self._pending_fanout_ref)
        except Exception as exc:
            self._pending_fanout_ref = None
            self._terminal_error = repr(exc)
            raise
        self._next_fanout_port = result["next_fanout_port"]
        self._fanout_group_info = result["fanout_group_info"]
        self._pending_fanout_ref = None
        return result


def _sync_relay_weights_to_peer_instances(
    *,
    rollout_engine_lock: ActorHandle,
    relay_engine: ActorHandle,
    peer_engines: Sequence[ActorHandle],
    relay_gpu_count: int,
    next_fanout_port: int,
    fanout_group_info: _FanoutGroupInfo | None,
    fanout_group_name: str,
) -> tuple[int, _FanoutGroupInfo | None]:
    if not peer_engines:
        return next_fanout_port, None

    acquire_rollout_engine_lock(rollout_engine_lock)
    try:
        if fanout_group_info is None:
            master_address, first_port = ray.get(
                relay_engine._get_current_node_ip_and_free_port.remote(
                    start_port=next_fanout_port,
                    consecutive=relay_gpu_count,
                )
            )
            next_fanout_port = first_port + relay_gpu_count + 1
            ports = ",".join(str(first_port + rank) for rank in range(relay_gpu_count))
            group_name = fanout_group_name
            fanout_group_info = _FanoutGroupInfo(
                master_address=master_address,
                ports=ports,
                group_name=group_name,
            )
            world_size = len(peer_engines) + 1

            init_refs = [
                relay_engine.init_weights_send_group_for_remote_instance.remote(
                    master_address=master_address,
                    ports=ports,
                    group_rank=0,
                    world_size=world_size,
                    group_name=group_name,
                    backend="nccl",
                )
            ]
            init_refs.extend(
                peer_engine.init_weights_send_group_for_remote_instance.remote(
                    master_address=master_address,
                    ports=ports,
                    group_rank=peer_rank,
                    world_size=world_size,
                    group_name=group_name,
                    backend="nccl",
                )
                for peer_rank, peer_engine in enumerate(peer_engines, start=1)
            )
            _raise_sglang_rpc_failures(
                ray.get(init_refs),
                f"initialize relay fanout group {group_name}",
            )

        send_refs = [
            relay_engine.send_weights_to_remote_instance.remote(
                master_address=fanout_group_info.master_address,
                ports=fanout_group_info.ports,
                group_name=fanout_group_info.group_name,
            )
        ]
        send_refs.extend(
            peer_engine.send_weights_to_remote_instance.remote(
                master_address=fanout_group_info.master_address,
                ports=fanout_group_info.ports,
                group_name=fanout_group_info.group_name,
            )
            for peer_engine in peer_engines
        )
        _raise_sglang_rpc_failures(
            ray.get(send_refs),
            f"broadcast relay weights through {fanout_group_info.group_name}",
        )
    finally:
        ray.get(rollout_engine_lock.release.remote())

    return next_fanout_port, fanout_group_info


def _destroy_relay_fanout_group(
    *,
    relay_engine: ActorHandle,
    peer_engines: Sequence[ActorHandle],
    fanout_group_info: _FanoutGroupInfo,
) -> None:
    destroy_refs = [
        relay_engine.destroy_weights_send_group_for_remote_instance.remote(
            group_name=fanout_group_info.group_name,
        )
    ]
    destroy_refs.extend(
        peer_engine.destroy_weights_send_group_for_remote_instance.remote(
            group_name=fanout_group_info.group_name,
        )
        for peer_engine in peer_engines
    )
    _raise_sglang_rpc_failures(
        ray.get(destroy_refs),
        f"destroy relay fanout group {fanout_group_info.group_name}",
    )


def _raise_sglang_rpc_failures(responses: Sequence[dict | None], action: str) -> None:
    for response in responses:
        if response is None:
            continue
        if not isinstance(response, dict) or response.get("success") is not True:
            message = response.get("message", response) if isinstance(response, dict) else response
            raise RuntimeError(f"Failed to {action}: {message}")


class UpdateWeightSendRecvBroadcast(DistBucketedWeightUpdateMixin):
    """
    Send each training PP stage's canonical buckets to one rollout relay, then
    let the relay synchronize the loaded model to peer rollout instances.
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
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._next_fanout_port = 20000
        self._relay_update_group = None
        self._relay_receive_refs: list[ray.ObjectRef] = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if len(engine_gpu_counts) != len(rollout_engines):
            raise ValueError(
                "sendrecv_broadcast requires one GPU count per rollout engine, "
                f"but got {len(engine_gpu_counts)} counts for {len(rollout_engines)} engines."
            )
        relay_gpu_count = engine_gpu_counts[0]
        if any(gpu_count != relay_gpu_count for gpu_count in engine_gpu_counts):
            raise ValueError(
                "sendrecv_broadcast requires homogeneous rollout engine GPU counts "
                f"for relay fanout, but got {list(engine_gpu_counts)}."
            )

        self._relay_engine = rollout_engines[0]
        self._peer_engines = list(rollout_engines[1:])
        self._relay_gpu_count = relay_gpu_count
        self._pp_rank = get_parallel_state().pp.rank
        self._connect_coordinator()

        if self._is_source:
            self._group_name = f"miles-sendrecv-broadcast-train-pp_{self._pp_rank}"
            if self._relay_update_group is not None:
                disconnect_rollout_relay_from_distributed(
                    self._group_name,
                    self._relay_update_group,
                    self._relay_engine,
                )
            self._relay_update_group = connect_rollout_relay_from_distributed(
                self._group_name,
                self._relay_engine,
            )

    def _connect_coordinator(self) -> None:
        coordinator_name = f"miles-sendrecv-broadcast-coordinator-{ray.get_runtime_context().get_job_id()}"
        fanout_group_name = f"miles-sendrecv-broadcast-fanout-{ray.get_runtime_context().get_job_id()}"
        if dist.get_rank() == 0:
            try:
                self._coordinator = ray.get_actor(coordinator_name)
            except ValueError:
                self._coordinator = _SendRecvBroadcastCoordinator.options(name=coordinator_name).remote()
            ray.get(
                self._coordinator.configure.remote(
                    expected_sources=get_parallel_state().pp.size,
                    rollout_engines=self.rollout_engines,
                    rollout_engine_lock=self.rollout_engine_lock,
                    relay_engine=self._relay_engine,
                    peer_engines=self._peer_engines,
                    relay_gpu_count=self._relay_gpu_count,
                    next_fanout_port=self._next_fanout_port,
                    fanout_group_name=fanout_group_name,
                )
            )
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() != 0:
            self._coordinator = ray.get_actor(coordinator_name)

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        acquire_rollout_engine_lock(self.rollout_engine_lock)
        try:
            relay_receive_ref = update_weights_from_distributed_send_recv(
                self._group_name,
                self._relay_update_group,
                self._relay_engine,
                converted_named_tensors,
            )
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        self._relay_receive_refs.append(relay_receive_ref)
        converted_named_tensors.clear()

        if pbar:
            pbar.update(1)

    def _finalize_and_resume_engines(self, post_load_weights: bool = False) -> None:
        if self._is_source:
            ray.get(
                self._coordinator.finish_pp_stage_and_maybe_fanout.remote(
                    weight_version=self.weight_version,
                    pp_rank=self._pp_rank,
                    relay_receive_refs=self._relay_receive_refs,
                )
            )
            self._relay_receive_refs = []

    def get_coordinator(self) -> ActorHandle | None:
        return getattr(self, "_coordinator", None)

    def wait_pending_fanout(self) -> None:
        gloo_group = get_gloo_group()
        error_message = None
        if dist.get_rank() == 0 and hasattr(self, "_coordinator"):
            try:
                result = ray.get(self._coordinator.wait_pending_fanout.remote())
                self._next_fanout_port = result["next_fanout_port"]
            except Exception as exc:
                error_message = repr(exc)
        errors = [None] * dist.get_world_size(group=gloo_group)
        dist.all_gather_object(errors, error_message, group=gloo_group)
        if any(errors):
            raise RuntimeError(
                "sendrecv_broadcast background update failed: " f"{next(error for error in errors if error)}"
            )
