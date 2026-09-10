from concurrent.futures import Future
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.backends.training_utils.weight_update.protocols.p2p_checksums import P2PChecksumShard


@pytest.fixture
def checksum_shards() -> list[P2PChecksumShard]:
    return [
        P2PChecksumShard("cell-a", 0, "session-a0", frozenset({"w", "b"}), {"w": "a0-w"}),
        P2PChecksumShard("cell-a", 0, "session-a0", frozenset({"w", "b"}), {"b": "a0-b"}),
        P2PChecksumShard("cell-a", 1, "session-a1", frozenset({"w", "b"}), {"w": "a1-w", "b": "a1-b"}),
        P2PChecksumShard("cell-b", 0, "session-b0", frozenset({"w"}), {"w": "b0-w"}),
    ]


class _ChecksumReplica:
    def __init__(self, buffer: torch.Tensor, engine_rank: int) -> None:
        self._buffer = buffer
        self._engine_rank = engine_rank

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        self._buffer.fill_(97 + self._engine_rank)


class _TransferRecorder:
    cell_id = "cell-a"

    def __init__(self, buffer: torch.Tensor) -> None:
        self._buffer = buffer
        self.sent: list[tuple[int, bytes]] = []

    def submit_write(self, engine_rank: int, names: list[str], weight_memory_registry: dict) -> Future:
        self.sent.append((engine_rank, self._buffer.numpy().tobytes()))
        future: Future = Future()
        future.set_result(None)
        return future

    def wait_for_write(self, future: Future) -> None:
        future.result()


@pytest.fixture
def checksum_protocol(p2p_protocol: ModuleType) -> tuple[object, _TransferRecorder]:
    protocol = p2p_protocol.UpdateWeightP2P.__new__(p2p_protocol.UpdateWeightP2P)
    protocol.args = SimpleNamespace(save_inference_engine_weight_checksum=True)
    protocol.is_sender = True
    protocol._model_registered = True
    buffer = torch.zeros(1, dtype=torch.uint8)
    recorder = _TransferRecorder(buffer)
    protocol._shared_params_dict = {"w": buffer}
    protocol._shared_param_mapper = object()
    protocol._weight_memory_registry = {"w": (buffer.data_ptr(), 1, 1)}
    protocol._model_param_stager = SimpleNamespace(
        get_transfer_ready_params=lambda tensors, **kwargs: (["w"], tensors)
    )
    protocol._transfer_engine_meta_list = [
        p2p_protocol.TransferEngineMeta(rank, _ChecksumReplica(buffer, rank), [recorder]) for rank in range(2)
    ]
    protocol._checksum_shards = {
        ("cell-a", rank): P2PChecksumShard("cell-a", rank, f"session-{rank}", frozenset({"w"})) for rank in range(2)
    }
    protocol._engine_gpu_counts = {"cell-a": 2}
    protocol.inference_cell_health = InferenceCellHealth(["cell-a"])
    protocol.expected_base_weight_checksums_by_cell = None
    return protocol, recorder
