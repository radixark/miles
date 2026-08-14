import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
    _send_to_colocated_engine,
)
from miles.utils import async_utils

_TENSOR_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"

_ENGINE_GPU_COUNTS = [2, 2, 2, 2]
_SPARSE_GPU_OFFSETS = [0, 2, 6, 8]
_BASE_LIFECYCLE_STAGES = (
    "pause_generation",
    "flush_cache",
    "begin_weight_update",
    "update_weights_from_tensor",
    "end_weight_update",
    "continue_generation",
)
_LORA_LIFECYCLE_STAGES = (
    "pause_generation",
    "flush_cache",
    "begin_weight_update",
    "update_weights_from_tensor",
    "load_lora_adapter_from_tensors",
    "end_weight_update",
    "continue_generation",
)


class _LifecycleRequest:
    def __init__(self, engine_index: int, method: str, error: Exception | None) -> None:
        self.engine_index = engine_index
        self.method = method
        self.error = error


class _LifecycleFuture:
    def __init__(self, probe: "_LifecycleProbe", request: _LifecycleRequest) -> None:
        self._probe = probe
        self._request = request

    def result(self) -> dict:
        return self._probe.await_request(self._request)


class _LifecycleProbe:
    def __init__(
        self,
        expected_stages: tuple[str, ...],
        num_engines: int,
        failing_request: tuple[int, str] | None = None,
    ) -> None:
        self.expected_stages = expected_stages
        self.num_engines = num_engines
        self.failing_request = failing_request
        self.requested: list[tuple[int, str]] = []
        self.awaited: list[tuple[int, str]] = []
        self._stage_index = 0

    def request(self, engine_index: int, method: str) -> _LifecycleRequest:
        if method != self.expected_stages[self._stage_index]:
            previous_stage = self.expected_stages[self._stage_index]
            assert self.requested.count((engine_index, previous_stage)) == 1
            assert len([item for item in self.awaited if item[1] == previous_stage]) == self.num_engines
            self._stage_index += 1

        assert method == self.expected_stages[self._stage_index]
        request_key = (engine_index, method)
        assert request_key not in self.requested
        self.requested.append(request_key)
        error = RuntimeError("engine is unreachable") if request_key == self.failing_request else None
        return _LifecycleRequest(engine_index=engine_index, method=method, error=error)

    def submit(self, request: _LifecycleRequest) -> _LifecycleFuture:
        return _LifecycleFuture(probe=self, request=request)

    def await_request(self, request: _LifecycleRequest) -> dict:
        request_key = (request.engine_index, request.method)
        assert len([item for item in self.requested if item[1] == request.method]) == self.num_engines
        assert request_key not in self.awaited
        self.awaited.append(request_key)
        if request.error is not None:
            raise request.error
        return {"success": True}


class _LifecycleEngine:
    def __init__(self, engine_index: int, probe: _LifecycleProbe) -> None:
        self._engine_index = engine_index
        self._probe = probe

    def __getattr__(self, name: str):
        def method(**kwargs):
            return self._probe.request(self._engine_index, name)

        return method


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = False

    def __init__(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        self._named_tensors = named_tensors

    def get_flattened_tensor(self) -> str:
        return self._named_tensors[0][0]

    def get_metadata(self) -> None:
        return None


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(data: dict, output_str: bool = False) -> str:
        return f"serialized:{data['flattened_tensor']}"


class _SerialProbeEngine:
    def __init__(self) -> None:
        self.received: list[str] = []
        self.active = 0
        self.max_active = 0

    async def update_weights_from_tensor(self, **kwargs) -> dict:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.received.append(kwargs["serialized_named_tensors"][0])
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"success": True}


class _LoraEngine:
    def __init__(self, unload_error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._unload_error = unload_error

    async def unload_lora_adapter(self, lora_name: str) -> dict:
        self.calls.append(("unload_lora_adapter", {"lora_name": lora_name}))
        if self._unload_error is not None:
            raise self._unload_error
        return {"success": True}

    async def load_lora_adapter_from_tensors(self, **kwargs) -> dict:
        self.calls.append(("load_lora_adapter_from_tensors", kwargs))
        return {"success": True}


def _gather_object_onto_source(obj, object_gather_list=None, dst=None, group=None) -> None:
    object_gather_list[0] = obj


class TestConnectRolloutEngines:
    @staticmethod
    def _make_updater() -> UpdateWeightFromTensor:
        updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        updater.args = Namespace(
            rollout_num_gpus_per_engine=2,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        updater._model_update_groups = None
        updater._ipc_gather_group = MagicMock(name="gather_group_from_init")
        updater._ipc_gather_src = 0
        return updater

    @pytest.mark.parametrize(
        ("rank", "expected_engine_index", "expected_gather_src"),
        [(0, 0, 0), (6, 2, 6)],
        ids=["first_engine", "engine_after_sparse_gap"],
    )
    def test_connect_splits_clients_and_maps_sparse_gpu_offsets(
        self, rank: int, expected_engine_index: int, expected_gather_src: int
    ) -> None:
        """Only remote engines use NCCL, while ranks on either side of a sparse gap map to their covered colocated engine."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        updater = self._make_updater()
        updater._ipc_gather_src = expected_gather_src
        parallel_state = SimpleNamespace(
            intra_dp_cp=SimpleNamespace(rank=0),
            tp=SimpleNamespace(rank=0),
            pp=SimpleNamespace(rank=0),
        )
        gather_group_from_init = updater._ipc_gather_group

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.get_parallel_state", return_value=parallel_state),
            patch(f"{_TENSOR_MODULE}.connect_rollout_engines_from_distributed") as connect,
            patch(f"{_TENSOR_MODULE}.disconnect_rollout_engines_from_distributed") as disconnect,
        ):
            dist_mock.get_rank.return_value = rank
            updater.connect_rollout_engines(
                engines,
                engine_gpu_counts=_ENGINE_GPU_COUNTS,
                engine_gpu_offsets=_SPARSE_GPU_OFFSETS,
            )

        assert updater.use_distribute is True
        assert updater.rollout_engines == engines[:3]
        assert updater.distributed_rollout_engines == engines[3:]
        connect.assert_called_once_with(updater.args, "miles", engines[3:], engine_gpu_counts=[2])
        disconnect.assert_not_called()
        assert updater._model_update_groups is connect.return_value
        assert updater._ipc_engine is engines[expected_engine_index]
        assert updater._ipc_gather_group is gather_group_from_init
        assert updater._ipc_gather_src == expected_gather_src
        dist_mock.new_group.assert_not_called()

    def test_connect_clears_ipc_state_for_a_placeholder_rank(self) -> None:
        """A reserved GPU slot that no engine covers drops its stale gather group, or it would enter a collective nobody else joins."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        updater = self._make_updater()
        parallel_state = SimpleNamespace(
            intra_dp_cp=SimpleNamespace(rank=0),
            tp=SimpleNamespace(rank=1),
            pp=SimpleNamespace(rank=0),
        )

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.get_parallel_state", return_value=parallel_state),
            patch(f"{_TENSOR_MODULE}.connect_rollout_engines_from_distributed") as connect,
            patch(f"{_TENSOR_MODULE}.disconnect_rollout_engines_from_distributed"),
        ):
            dist_mock.get_rank.return_value = 4
            updater.connect_rollout_engines(
                engines,
                engine_gpu_counts=_ENGINE_GPU_COUNTS,
                engine_gpu_offsets=_SPARSE_GPU_OFFSETS,
            )

        assert updater._ipc_gather_group is None
        assert updater._ipc_gather_src is None
        assert updater._ipc_engine is None
        connect.assert_not_called()


class TestTensorUpdateLifecycle:
    @staticmethod
    def _make_updater(engines: list[_LifecycleEngine], *, is_lora: bool = False) -> UpdateWeightFromTensor:
        updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        updater.args = Namespace(pause_generation_mode="retract")
        updater.is_lora = is_lora
        updater.use_distribute = False
        updater.weight_version = 0
        updater.rollout_engines = engines
        updater.weights_getter = lambda: {}
        updater._hf_weight_iterator = SimpleNamespace(
            get_hf_weight_chunks=lambda weights, weight_type: [
                [
                    (
                        "model.layers.0.mlp.gate_proj.lora_A.weight" if weight_type == "lora" else "w",
                        torch.zeros(2),
                    )
                ]
            ]
        )
        updater._send_base_params = lambda hf_named_tensors: (
            [async_utils.submit(engine.update_weights_from_tensor()) for engine in engines],
            None,
        )
        if is_lora:
            updater._send_lora_params = lambda hf_named_tensors: (
                [async_utils.submit(engine.load_lora_adapter_from_tensors()) for engine in engines],
                None,
            )
            updater._lora_base_synced = False
        return updater

    @staticmethod
    def _run(updater: UpdateWeightFromTensor, probe: _LifecycleProbe) -> None:
        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_TENSOR_MODULE}._pp_assemble_full_adapter", side_effect=lambda tensors: tensors),
            patch(f"{_TENSOR_MODULE}.async_utils.submit", side_effect=probe.submit),
            patch(f"{_TENSOR_MODULE}.torch.cuda.ipc_collect"),
            patch(f"{_TENSOR_MODULE}.torch.cuda.empty_cache"),
        ):
            dist_mock.get_rank.return_value = 0
            UpdateWeightFromTensor.update_weights(updater)

    def test_tensor_update_lifecycle_drives_every_client_through_pause_transfer_and_resume(self) -> None:
        """Every phase waits for every engine before the next phase can start."""
        probe = _LifecycleProbe(expected_stages=_BASE_LIFECYCLE_STAGES, num_engines=2)
        engines = [_LifecycleEngine(engine_index=i, probe=probe) for i in range(2)]

        self._run(updater=self._make_updater(engines), probe=probe)

        expected = [(engine_index, method) for method in _BASE_LIFECYCLE_STAGES for engine_index in range(2)]
        assert probe.requested == expected
        assert probe.awaited == expected

    def test_tensor_update_lifecycle_awaits_lora_transfer_before_close_and_resume(self) -> None:
        """The LoRA transfer result from every engine settles before the session closes and generation resumes."""
        probe = _LifecycleProbe(expected_stages=_LORA_LIFECYCLE_STAGES, num_engines=2)
        engines = [_LifecycleEngine(engine_index=i, probe=probe) for i in range(2)]

        self._run(updater=self._make_updater(engines, is_lora=True), probe=probe)

        expected = [(engine_index, method) for method in _LORA_LIFECYCLE_STAGES for engine_index in range(2)]
        assert probe.requested == expected
        assert probe.awaited == expected

    @pytest.mark.parametrize(
        ("failing_method", "completed_stages"),
        [
            (
                "update_weights_from_tensor",
                _BASE_LIFECYCLE_STAGES[:4],
            ),
            (
                "end_weight_update",
                _BASE_LIFECYCLE_STAGES[:5],
            ),
        ],
    )
    def test_tensor_update_lifecycle_does_not_advance_or_resume_after_api_failure(
        self, failing_method: str, completed_stages: tuple[str, ...]
    ) -> None:
        """One client's failure stops the next phase only after the other client's result has also settled."""
        probe = _LifecycleProbe(
            expected_stages=_BASE_LIFECYCLE_STAGES,
            num_engines=2,
            failing_request=(0, failing_method),
        )
        engines = [_LifecycleEngine(engine_index=i, probe=probe) for i in range(2)]

        with pytest.raises(RuntimeError, match="engine is unreachable"):
            self._run(updater=self._make_updater(engines), probe=probe)

        expected = [(engine_index, method) for method in completed_stages for engine_index in range(2)]
        assert probe.requested == expected
        assert probe.awaited == expected


class TestColocatedBaseSend:
    def test_multiple_colocated_buckets_are_sent_serially_in_source_order(self) -> None:
        """Two dtype buckets reach the engine one at a time and in the order they were built."""
        engine = _SerialProbeEngine()
        named_tensors = [
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, dtype=torch.float32)),
            ("model.layers.0.mlp.up_proj.weight", torch.zeros(2, dtype=torch.float16)),
        ]

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.FlattenedTensorBucket", _FakeFlattenedTensorBucket),
            patch(f"{_TENSOR_MODULE}.MultiprocessingSerializer", _FakeMultiprocessingSerializer),
        ):
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            dist_mock.gather_object.side_effect = _gather_object_onto_source
            futures, _long_lived_tensors = _send_to_colocated_engine(
                named_tensors,
                ipc_engine=engine,
                ipc_gather_src=0,
                ipc_gather_group=MagicMock(name="gather_group"),
                weight_version=5,
            )
            async_utils.wait_futures(futures)

        assert engine.received == [
            "serialized:model.layers.0.mlp.gate_proj.weight",
            "serialized:model.layers.0.mlp.up_proj.weight",
        ]
        assert engine.max_active == 1


class TestColocatedLoraSend:
    def test_colocated_lora_load_continues_when_prior_adapter_unload_fails(self) -> None:
        """An engine that has no adapter to unload yet must not block the replacement adapter from being sent."""
        engine = _LoraEngine(unload_error=RuntimeError("adapter not found"))
        named_tensors = [("model.layers.0.mlp.gate_proj.lora_A.weight", torch.zeros(4, 8))]

        with patch(f"{_TENSOR_MODULE}.dist") as dist_mock:
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            dist_mock.gather_object.side_effect = _gather_object_onto_source
            futures, _long_lived_tensors = _send_to_colocated_engine(
                named_tensors,
                ipc_engine=engine,
                ipc_gather_src=0,
                ipc_gather_group=MagicMock(name="gather_group"),
                weight_version=5,
                lora_config={"peft_type": "LORA", "r": 8},
                lora_name="adapter-a",
            )
            async_utils.wait_futures(futures)

        assert [name for name, _kwargs in engine.calls] == ["unload_lora_adapter", "load_lora_adapter_from_tensors"]
        assert engine.calls[1][1]["lora_name"] == "adapter-a"
        assert engine.calls[1][1]["config_dict"] == {"peft_type": "LORA", "r": 8}
