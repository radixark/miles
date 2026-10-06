import ctypes
import dataclasses
import importlib
import sys
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any
import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks_for_data_replica,
)

_FAILURE_BOUND = 10.0
_WEIGHT_NUMEL = 4
_BUCKET_VALUES = {
    "hf.w": [1.0, 2.0, 3.0, 4.0],
    "hf.q": [5.0, 6.0],
    "hf.k": [7.0, 8.0],
}


@dataclasses.dataclass
class _FakeServerArgs:
    rl_quant_profile: str | None = None
    quantization: str | None = None


@dataclasses.dataclass
class _FakeRankParallelismConfig:
    tp_rank: int
    global_rank: int

    @classmethod
    def from_dict(cls, parallelism_info: dict) -> "_FakeRankParallelismConfig":
        return cls(**parallelism_info)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class _FakeMapping:
    sglang_name: str
    num_shards: int
    num_local_experts: int | None = None


_HF_TO_SGLANG = {
    "hf.w": _FakeMapping("w", num_shards=1),
    "hf.q": _FakeMapping("qk", num_shards=2),
    "hf.k": _FakeMapping("qk", num_shards=2),
}


class _FakeParameterMapper:
    def map(self, name: str) -> _FakeMapping:
        return _HF_TO_SGLANG[name]


class _SharedBufferReplica(torch.nn.Module):
    def __init__(self, tp_rank: int, harness: "_P2PSenderHarness") -> None:
        super().__init__()
        self.tp_rank = tp_rank
        self._harness = harness
        self.w = torch.nn.Parameter(torch.zeros(_WEIGHT_NUMEL), requires_grad=False)
        self.qk = torch.nn.Parameter(torch.zeros(_WEIGHT_NUMEL), requires_grad=False)

    def load_weights(self, tensors: list[tuple[str, torch.Tensor]]) -> None:
        by_name = dict(tensors)
        offset = 100.0 * self.tp_rank
        if "hf.w" in by_name:
            self.w.data.copy_(by_name["hf.w"] + offset)
        if "hf.q" in by_name or "hf.k" in by_name:
            self.qk.data.copy_(torch.cat([by_name["hf.q"], by_name["hf.k"]]) + offset)
        self._harness.log.append(("load", self.tp_rank, tuple(by_name)))
        self._harness.loaded_event(self.tp_rank).set()


class _WriteHold:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()


class _FakeTransferEngine:
    def __init__(self, log: list[tuple]) -> None:
        self._log = log
        self.registered: list[tuple[int, int]] = []
        self.register_return_code = 0
        self.writes: list[tuple[str, dict[int, list[float]]]] = []
        self.holds: dict[str, _WriteHold] = {}
        self.failing_sessions: set[str] = set()

    def hold(self, session_id: str) -> _WriteHold:
        return self.holds.setdefault(session_id, _WriteHold())

    def release_all(self) -> None:
        for hold in self.holds.values():
            hold.release.set()

    def register_memory(self, address: int, size: int) -> int:
        self.registered.append((address, size))
        return self.register_return_code

    def batch_transfer_sync_write(
        self, session_id: str, source_ptrs: list[int], target_ptrs: list[int], source_lens: list[int]
    ) -> int:
        if (hold := self.holds.get(session_id)) is not None:
            hold.entered.set()
            if not hold.release.wait(timeout=_FAILURE_BOUND):
                raise TimeoutError(f"the test never released the write to {session_id}")
        payload = {
            target_ptr: torch.frombuffer(bytearray(ctypes.string_at(source_ptr, length)), dtype=torch.float32).tolist()
            for source_ptr, target_ptr, length in zip(source_ptrs, target_ptrs, source_lens, strict=True)
        }
        self._log.append(("write", session_id))
        self.writes.append((session_id, payload))
        return -1 if session_id in self.failing_sessions else 0

    def written_sessions(self) -> list[str]:
        return [session_id for session_id, _payload in self.writes]

    def payload_of(self, session_id: str) -> dict[int, list[float]]:
        (payload,) = [payload for written, payload in self.writes if written == session_id]
        return payload


class _FakeRolloutApi:
    def __init__(
        self,
        cell_id: str,
        gpu_count: int,
        generation: int = 1,
        unreachable: bool = False,
    ) -> None:
        self.cell_id = cell_id
        self.gpu_count = gpu_count
        self.generation = generation
        self.unreachable = unreachable
        self.calls: list[str] = []

    def session_id(self, rank: int) -> str:
        return f"{self.cell_id}-g{self.generation}-r{rank}"

    def target_address(self, rank: int, name: str) -> int:
        return hash((self.session_id(rank), name)) & 0xFFFFFFFF

    async def get_remote_instance_transfer_engine_info(self, rank: int) -> tuple[str, dict]:
        self.calls.append("get_remote_instance_transfer_engine_info")
        if self.unreachable:
            raise RuntimeError(f"{self.cell_id} is gone")
        weights = {name: (self.target_address(rank, name), _WEIGHT_NUMEL, 4) for name in ("w", "qk")}
        return self.session_id(rank), weights

    async def get_parallelism_info(self, rank: int) -> dict:
        self.calls.append("get_parallelism_info")
        return {"tp_rank": rank, "global_rank": 10 * self.generation + rank}

    async def get_server_info(self) -> dict:
        self.calls.append("get_server_info")
        return {"rl_quant_profile": None}


class _ObservedExecutor(ThreadPoolExecutor):
    def __init__(self, waiting_threads: set[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._waiting_threads = waiting_threads

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        future = super().submit(fn, *args, **kwargs)
        wait_for_result = future.result

        def _observed_result(timeout: float | None = None) -> Any:
            self._waiting_threads.add(threading.get_ident())
            try:
                return wait_for_result(timeout=timeout)
            finally:
                self._waiting_threads.discard(threading.get_ident())

        future.result = _observed_result
        return future


class _ProtocolCall:
    def __init__(self, harness: "_P2PSenderHarness", target: Callable[[], None]) -> None:
        self._harness = harness
        self._target = target
        self.error: BaseException | None = None
        self.log_at_return: list[tuple] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._target()
        except BaseException as error:
            self.error = error
        self.log_at_return = list(self._harness.log)

    @property
    def returned(self) -> bool:
        return not self._thread.is_alive()

    def is_draining(self) -> bool:
        return self._thread.ident in self._harness.waiting_threads

    def wait_until_draining_or_returned(self, extra: Callable[[], bool] = lambda: False) -> str:
        deadline = time.monotonic() + _FAILURE_BOUND
        while time.monotonic() < deadline:
            if self.returned:
                return "returned"
            if self.is_draining():
                return "draining"
            if extra():
                return "extra"
            time.sleep(0.001)
        raise AssertionError("the protocol call neither drained nor returned")

    def join(self) -> None:
        self._thread.join(timeout=_FAILURE_BOUND)
        assert not self._thread.is_alive(), "the protocol call is still running"
        if self.error is not None:
            raise self.error


class _P2PSenderHarness:
    def __init__(self, p2p_protocol: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        self._p2p_protocol = p2p_protocol
        self.log: list[tuple] = []
        self.transfer_engine = _FakeTransferEngine(self.log)
        self.transfer_engines_created = 0
        self.replicas_created: list[_SharedBufferReplica] = []
        self._loaded_events: dict[int, threading.Event] = {}
        self._calls: list[_ProtocolCall] = []
        self.waiting_threads: set[int] = set()
        self.data_replica_rank = 0
        self.data_replica_size = 1

        mooncake_transport = sys.modules[p2p_protocol.MooncakeTransport.__module__]
        monkeypatch.setattr(mooncake_transport, "_create_transfer_engine", self._create_transfer_engine)
        monkeypatch.setattr(
            mooncake_transport,
            "ThreadPoolExecutor",
            lambda **kwargs: _ObservedExecutor(self.waiting_threads, **kwargs),
        )
        monkeypatch.setattr(
            p2p_protocol,
            "assign_rollout_engine_ranks",
            lambda parallel_state, placement, engine_gpu_counts: assign_rollout_engine_ranks_for_data_replica(
                data_replica_rank=self.data_replica_rank,
                data_replica_size=self.data_replica_size,
                engine_gpu_counts=engine_gpu_counts,
            ),
        )
        monkeypatch.setattr(p2p_protocol, "ParallelismContext", lambda parallelism_config: nullcontext())
        monkeypatch.setattr(p2p_protocol, "get_gloo_group", lambda: None)
        monkeypatch.setattr(p2p_protocol, "dist", SimpleNamespace(get_rank=lambda group=None: 0))
        model_replica = sys.modules[p2p_protocol.query_rollout_engine_rank_configs.__module__]
        monkeypatch.setattr(model_replica, "RankParallelismConfig", _FakeRankParallelismConfig)
        monkeypatch.setattr(model_replica, "ServerArgs", _FakeServerArgs)
        monkeypatch.setattr(model_replica, "_build_cpu_replica", self._build_cpu_replica)
        monkeypatch.setattr(
            model_replica, "ParameterMapper", SimpleNamespace(from_model=lambda model: _FakeParameterMapper())
        )
        # CPU CI has no pinned memory
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)

    def loaded_event(self, tp_rank: int) -> threading.Event:
        return self._loaded_events.setdefault(tp_rank, threading.Event())

    def make_protocol(self, *, data_replica_rank: int = 0, data_replica_size: int = 1) -> Any:
        self.data_replica_rank = data_replica_rank
        self.data_replica_size = data_replica_size
        args = Namespace(
            hf_checkpoint="/model",
            p2p_transfer_timeout=_FAILURE_BOUND,
            p2p_transfer_num_workers=4,
            update_weight_engine_request_timeout=_FAILURE_BOUND,
            sglang_pp_size=1,
        )
        return self._p2p_protocol.UpdateWeightP2P(args)

    def connect(self, protocol: Any, apis: list[_FakeRolloutApi]) -> None:
        (gpu_count,) = {api.gpu_count for api in apis} or {1}
        protocol.args.rollout_num_gpus_per_engine = gpu_count
        protocol.args.rollout_num_gpus = gpu_count * len(apis)
        protocol.connect(
            rollout_engines=apis,
            engine_gpu_counts=[api.gpu_count for api in apis],
            engine_gpu_offsets=None,
            parallel_state=None,
            placement=None,
            selector="",
        )

    def call_in_thread(self, target: Callable[[], None]) -> _ProtocolCall:
        call = _ProtocolCall(self, target)
        self._calls.append(call)
        return call

    def close(self) -> None:
        self.transfer_engine.release_all()
        for call in self._calls:
            call.join()

    def _create_transfer_engine(self) -> _FakeTransferEngine:
        self.transfer_engines_created += 1
        return self.transfer_engine

    def _build_cpu_replica(self, config: Any, model_path: str) -> _SharedBufferReplica:
        replica = _SharedBufferReplica(tp_rank=config.parallelism.tp_rank, harness=self)
        self.replicas_created.append(replica)
        return replica


_P2P_PROTOCOL_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p"


@contextmanager
def _stubbed_missing_external_sdks(module_attributes: dict[str, dict[str, object]]) -> Iterator[None]:
    created_modules: list[str] = []
    created_attributes: list[tuple[ModuleType, str]] = []

    for module_name, attributes in module_attributes.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name in sys.modules:
                continue
            try:
                importlib.import_module(name)
                continue
            except ImportError:
                pass
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            created_modules.append(name)
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                setattr(parent, parts[depth - 1], module)
                created_attributes.append((parent, parts[depth - 1]))
        for attribute, value in attributes.items():
            module = sys.modules[module_name]
            if not hasattr(module, attribute):
                setattr(module, attribute, value)
                created_attributes.append((module, attribute))

    try:
        yield
    finally:
        for parent, attribute in reversed(created_attributes):
            delattr(parent, attribute)
        for name in reversed(created_modules):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def p2p_protocol() -> ModuleType:
    with _stubbed_missing_external_sdks(
        {
            "mooncake.engine": {"TransferEngine": object},
            "sglang.srt.server_args": {"ServerArgs": object},
            "sglang.srt.configs.device_config": {"DeviceConfig": object},
            "sglang.srt.configs.load_config": {"LoadConfig": object},
            "sglang.srt.configs.model_config": {"ModelConfig": object},
            "sglang.srt.distributed.parallel_state": {
                "ParallelismContext": object,
                "RankParallelismConfig": object,
            },
            "sglang.srt.layers.moe": {"initialize_moe_config": lambda *args, **kwargs: None},
            "sglang.srt.layers.quantization.fp4_utils": {"initialize_fp4_gemm_config": lambda *args, **kwargs: None},
            "sglang.srt.layers.quantization.fp8_utils": {"initialize_fp8_gemm_config": lambda *args, **kwargs: None},
            "sglang.srt.model_loader": {"get_model": lambda *args, **kwargs: None},
            "sglang.srt.model_loader.parameter_mapper": {"ParameterMapper": object},
        }
    ):
        return importlib.import_module(_P2P_PROTOCOL_MODULE)


@pytest.fixture(scope="module")
def model_replica_module(p2p_protocol: ModuleType) -> ModuleType:
    return sys.modules[p2p_protocol.query_rollout_engine_rank_configs.__module__]


@pytest.fixture
def p2p_sender(p2p_protocol: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Iterator[_P2PSenderHarness]:
    harness = _P2PSenderHarness(p2p_protocol, monkeypatch)
    yield harness
    harness.close()


@pytest.fixture
def make_rollout_api() -> Callable[..., _FakeRolloutApi]:
    return _FakeRolloutApi


@pytest.fixture
def make_bucket() -> Callable[..., list[tuple[str, torch.Tensor]]]:
    def _make(*names: str) -> list[tuple[str, torch.Tensor]]:
        return [(name, torch.tensor(_BUCKET_VALUES[name])) for name in names]

    return _make
