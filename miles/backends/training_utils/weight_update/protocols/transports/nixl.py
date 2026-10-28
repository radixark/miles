import base64
import os
import time
import uuid
from typing import Any

import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import RemoteShard

# a write thread sleeps this long between checks of a running transfer, leaving the GIL to the loads
_TRANSFER_POLL_SECONDS = 0.0005


class NixlTransport:
    """The NIXL `NativeTransport` of `P2PTransport`: one NIXL agent for this trainer process.

    Each runner of a rollout engine rank publishes a NIXL agent, which `connect_peer` loads into this one. A write is
    one NIXL WRITE per shard, failed if it does not finish within `write_timeout`.
    """

    name = "nixl"
    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self, write_timeout: float) -> None:
        self._agent = _create_nixl_agent()
        self._write_timeout = write_timeout
        self._gpu_id_by_remote_agent: dict[str, int] = {}
        # a write that did not finish does not prove NIXL stopped reading its source
        self._has_failed_writes = False

    def connect_peer(self, transfer_engine_info: dict) -> str:
        remote_agent = self._agent.add_remote_agent(base64.b64decode(transfer_engine_info["agent_metadata"]))
        self._gpu_id_by_remote_agent[remote_agent] = transfer_engine_info["gpu_id"]
        return remote_agent

    def remove_peers(self) -> None:
        for remote_agent in self._gpu_id_by_remote_agent:
            self._agent.remove_remote_agent(remote_agent)
        self._gpu_id_by_remote_agent = {}

    def register_tensor(self, tensor: torch.Tensor) -> Any:
        return self._agent.register_memory(tensor)

    def unregister_tensor(self, registration: Any) -> None:
        if self._has_failed_writes:
            raise RuntimeError(
                f"Cannot release registered memory after a {self.name} write with unknown native completion"
            )
        self._agent.deregister_memory(registration)

    def write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
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
        state = None
        try:
            state = self._agent.transfer(transfer)
            deadline = time.monotonic() + self._write_timeout
            while state == "PROC" and time.monotonic() < deadline:
                time.sleep(_TRANSFER_POLL_SECONDS)
                state = self._agent.check_xfer_state(transfer)
        finally:
            # write threads share this flag: only ever set it, so one thread's success never clears another's failure
            if state != "DONE":
                self._has_failed_writes = True
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
