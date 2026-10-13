import base64
import os
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    parse_weight_locations,
    query_remote_shards,
)
from miles.backends.training_utils.weight_update.protocols.transports.rollout_engine_write_threads import (
    RolloutEngineWriteThreads,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)

# a write thread sleeps this long between checks of a running transfer, leaving the GIL to the loads
_TRANSFER_POLL_SECONDS = 0.0005


class NixlTransport:
    """Writes tensors from this trainer process's registered memory into rollout engines over NIXL.

    Used like `MooncakeTransport`. Each runner of a rollout engine rank publishes a NIXL agent; `connect` loads
    those agents' metadata into this process's agent, and a write is one NIXL WRITE per shard, failed if it does not
    finish within `write_timeout`.
    """

    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self, write_timeout: float) -> None:
        self._agent = _create_nixl_agent()
        self._write_timeout = write_timeout
        self._write_threads = RolloutEngineWriteThreads(thread_name_prefix="nixl-write")
        self._registrations_by_address: dict[int, Any] = {}
        self._gpu_id_by_remote_agent: dict[str, int] = {}

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        assignments: Sequence[RolloutEngineRankAssignment],
        runner_roles: Sequence[str],
    ) -> dict[str, dict[int, list[RemoteShard]]]:
        """Return assigned shards by runner role and rollout rank, with their agents loaded."""
        self._write_threads.restart()
        for remote_agent in self._gpu_id_by_remote_agent:
            self._agent.remove_remote_agent(remote_agent)
        self._gpu_id_by_remote_agent = {}
        return query_remote_shards(rollout_engines, assignments, runner_roles, "nixl", self._connect_remote_shard)

    def register_memory(self, tensor: torch.Tensor) -> None:
        self._registrations_by_address[tensor.data_ptr()] = self._agent.register_memory(tensor)

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        self._agent.deregister_memory(self._registrations_by_address.pop(tensor.data_ptr()))

    def write(self, remote_shards: Sequence[RemoteShard], tensors_by_name: dict[str, torch.Tensor]) -> list[Future]:
        """Writes each tensor into the weight of the same name on every shard; returns one future per shard.

        The tensors must lie in registered memory and stay unchanged until their futures are done.
        """
        return self._write_threads.write_each_shard(remote_shards, tensors_by_name, self._write_shard)

    def _connect_remote_shard(
        self, transfer_engine_info: dict, rollout_engine_ind: int, rollout_engine_rank: int, runner_role: str
    ) -> RemoteShard:
        remote_agent = self._agent.add_remote_agent(base64.b64decode(transfer_engine_info["agent_metadata"]))
        self._gpu_id_by_remote_agent[remote_agent] = transfer_engine_info["gpu_id"]
        return RemoteShard(
            rollout_engine_ind=rollout_engine_ind,
            rollout_engine_rank=rollout_engine_rank,
            runner_role=runner_role,
            transport_peer=remote_agent,
            weight_locations_by_name=parse_weight_locations(transfer_engine_info),
        )

    def _write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        names = list(tensors_by_name)
        gpu_id = self._gpu_id_by_remote_agent[remote_shard.transport_peer]
        transfer = self._agent.initialize_xfer(
            "WRITE",
            self._agent.get_xfer_descs([tensors_by_name[name] for name in names]),
            self._agent.get_xfer_descs(
                [
                    (remote_shard.weight_locations_by_name[name].address, tensors_by_name[name].nbytes, gpu_id)
                    for name in names
                ],
                "VRAM",
            ),
            remote_shard.transport_peer,
        )
        try:
            state = self._agent.transfer(transfer)
            deadline = time.monotonic() + self._write_timeout
            while state == "PROC" and time.monotonic() < deadline:
                time.sleep(_TRANSFER_POLL_SECONDS)
                state = self._agent.check_xfer_state(transfer)
        finally:
            self._agent.release_xfer_handle(transfer)
        if state != "DONE":
            raise RuntimeError(
                f"the NIXL WRITE of {len(names)} weights is {state}"
                + (f" after {self._write_timeout}s" if state == "PROC" else "")
            )


def _create_nixl_agent() -> Any:
    # not in miles' requirements
    from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    # the plugin the rollout engines' agents use
    backend = os.environ.get("SGLANG_REMOTE_INSTANCE_NIXL_BACKEND", "UCX")
    agent = nixl_agent(
        f"miles-p2p-{uuid.uuid4()}",
        # the write threads of all rollout engines share the agent
        nixl_agent_config(backends=[backend], sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT),
    )
    if backend not in agent.get_plugin_list():
        raise ValueError(
            f"NIXL has no {backend} plugin (SGLANG_REMOTE_INSTANCE_NIXL_BACKEND); installed: {agent.get_plugin_list()}"
        )
    return agent
