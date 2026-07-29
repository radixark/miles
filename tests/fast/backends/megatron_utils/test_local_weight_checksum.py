"""Tests for local_weight_checksum module."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils.local_weight_checksum import (
    _compute_weight_checksum_state,
    _transform_tensor_to_hash,
    dump_local_weight_checksums,
)
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import TrainEngineLocalWeightChecksumEvent
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def _make_mock_model_chunk(
    params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor] | None = None
) -> MagicMock:
    """Create a mock DDP model chunk with given named parameters and buffers."""
    chunk = MagicMock()

    param_list = sorted(params.items(), key=lambda x: x[0])
    for _name, tensor in param_list:
        tensor.main_param = tensor

    chunk.named_parameters.return_value = param_list
    chunk.named_buffers.return_value = sorted((buffers or {}).items(), key=lambda x: x[0])
    return chunk


def _make_mock_optimizer_with_state_dict(
    params: list[torch.Tensor],
    states: dict[int, dict[str, torch.Tensor]] | None = None,
) -> MagicMock:
    """Create a mock optimizer that has a torch.optim.Optimizer-like inner optimizer."""
    inner = MagicMock(spec=torch.optim.Adam)
    inner.param_groups = [{"params": params, "lr": 0.001}]

    inner_state: dict[torch.Tensor, dict] = {}
    sd_state: dict[int, dict] = {}
    for i, p in enumerate(params):
        s = (states or {}).get(i, {"step": torch.tensor(1)})
        inner_state[p] = s
        sd_state[i] = s
    inner.state = inner_state
    inner.state_dict.return_value = {
        "state": sd_state,
        "param_groups": [{"params": list(range(len(params))), "lr": 0.001}],
    }

    sub_opt = MagicMock(spec=["optimizer"])
    sub_opt.optimizer = inner

    optimizer = MagicMock(spec=["chained_optimizers"])
    optimizer.chained_optimizers = [sub_opt]
    return optimizer


class TestComputeWeightChecksums:
    def test_param_hashes_keys_match_expected_names(self) -> None:
        params = {
            "module.layers.0.weight": torch.randn(4, 4),
            "module.layers.1.weight": torch.randn(4, 4),
        }
        model = [_make_mock_model_chunk(params=params)]
        optimizer = _make_mock_optimizer_with_state_dict(params=list(params.values()))

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert set(entry.param_hashes.keys()) == {
            "pp0.module.layers.0.weight",
            "pp0.module.layers.1.weight",
        }

    def test_hash_determinism_same_params_same_hash(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        params = {"weight": tensor}
        model = [_make_mock_model_chunk(params=params)]
        optimizer = _make_mock_optimizer_with_state_dict(params=[tensor])

        entry1 = _compute_weight_checksum_state(model=model, optimizer=optimizer)
        entry2 = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert entry1.param_hashes == entry2.param_hashes

    def test_param_value_change_changes_hash(self) -> None:
        tensor_a = torch.tensor([1.0, 2.0, 3.0])
        tensor_b = torch.tensor([1.0, 2.0, 4.0])
        model_a = [_make_mock_model_chunk(params={"weight": tensor_a})]
        model_b = [_make_mock_model_chunk(params={"weight": tensor_b})]
        opt_a = _make_mock_optimizer_with_state_dict(params=[tensor_a])
        opt_b = _make_mock_optimizer_with_state_dict(params=[tensor_b])

        entry_a = _compute_weight_checksum_state(model=model_a, optimizer=opt_a)
        entry_b = _compute_weight_checksum_state(model=model_b, optimizer=opt_b)

        assert entry_a.param_hashes["pp0.weight"] != entry_b.param_hashes["pp0.weight"]

    def test_buffer_hashing_produces_values(self) -> None:
        params = {"weight": torch.randn(4, 4)}
        buffers = {"running_mean": torch.randn(4), "running_var": torch.randn(4)}
        model = [_make_mock_model_chunk(params=params, buffers=buffers)]
        optimizer = _make_mock_optimizer_with_state_dict(params=list(params.values()))

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert "pp0.running_mean" in entry.buffer_hashes
        assert "pp0.running_var" in entry.buffer_hashes
        assert len(entry.buffer_hashes["pp0.running_mean"]) == 64

    def test_optimizer_state_dict_captured(self) -> None:
        weight = torch.randn(4, 4)
        exp_avg = torch.randn(4, 4)

        model = [_make_mock_model_chunk(params={"weight": weight})]
        optimizer = _make_mock_optimizer_with_state_dict(
            params=[weight],
            states={0: {"exp_avg": exp_avg, "step": torch.tensor(5)}},
        )

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert len(entry.optimizer_hashes) == 1
        info = entry.optimizer_hashes[0]
        assert info.param_names[0] == "pp0.weight"
        assert isinstance(info.state_dict["state"][0]["exp_avg"], str)
        assert len(info.state_dict["state"][0]["exp_avg"]) == 64


class TestTransformTensorToHash:
    def test_replaces_tensor_with_hash(self) -> None:
        t = torch.tensor([1.0, 2.0])
        result = _transform_tensor_to_hash(t)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_preserves_non_tensor_values(self) -> None:
        assert _transform_tensor_to_hash(42) == 42
        assert _transform_tensor_to_hash("hello") == "hello"

    def test_recurses_into_dict(self) -> None:
        result = _transform_tensor_to_hash({"a": torch.tensor([1.0]), "b": 2})
        assert isinstance(result["a"], str)
        assert result["b"] == 2

    def test_recurses_into_list(self) -> None:
        result = _transform_tensor_to_hash([torch.tensor([1.0]), 3])
        assert isinstance(result[0], str)
        assert result[1] == 3

    def test_multiple_pp_chunks_produce_distinct_prefixes(self) -> None:
        params_0 = {"embed.weight": torch.randn(4, 4)}
        params_1 = {"head.weight": torch.randn(4, 4)}
        model = [
            _make_mock_model_chunk(params=params_0),
            _make_mock_model_chunk(params=params_1),
        ]
        all_params = list(params_0.values()) + list(params_1.values())
        optimizer = _make_mock_optimizer_with_state_dict(params=all_params)

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert "pp0.embed.weight" in entry.param_hashes
        assert "pp1.head.weight" in entry.param_hashes

    def test_transform_tensor_to_hash_preserves_tuple_type(self) -> None:
        result = _transform_tensor_to_hash((torch.tensor([1.0]), 42))
        assert isinstance(result, tuple)
        assert isinstance(result[0], str)
        assert result[1] == 42


class TestFailFastAssertions:
    def test_assert_event_logger_initialized_when_enabled(self) -> None:
        import miles.utils.audit_utils.event_logger.logger as mod

        original = mod._event_logger
        mod._event_logger = None
        try:
            args = Namespace(save_local_weight_checksum=True)
            model = [_make_mock_model_chunk(params={"w": torch.randn(2, 2)})]
            optimizer = _make_mock_optimizer_with_state_dict(params=[torch.randn(2, 2)])

            with pytest.raises(AssertionError, match="EventLogger is not initialized"):
                dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)
        finally:
            mod._event_logger = original

    def test_assert_empty_params_fails(self) -> None:
        model = [_make_mock_model_chunk(params={})]
        optimizer = MagicMock()
        optimizer.chained_optimizers = []

        with pytest.raises(AssertionError, match="No parameters found"):
            _compute_weight_checksum_state(model=model, optimizer=optimizer)

    def test_assert_no_sub_optimizers_fails(self) -> None:
        model = [_make_mock_model_chunk(params={"w": torch.randn(2, 2)})]
        optimizer = MagicMock()
        optimizer.chained_optimizers = []

        with pytest.raises(AssertionError, match="No sub-optimizers found"):
            _compute_weight_checksum_state(model=model, optimizer=optimizer)

    def test_assert_param_without_main_param_fails(self) -> None:
        from miles.backends.megatron_utils.local_weight_checksum import _build_name_by_tensor_id

        chunk = MagicMock()
        param = torch.randn(2, 2)
        chunk.named_parameters.return_value = [("weight", param)]

        with pytest.raises(AssertionError, match="main_param is None"):
            _build_name_by_tensor_id([chunk])

    def test_assert_unmapped_fp32_param_fails(self) -> None:
        from miles.backends.megatron_utils.local_weight_checksum import _build_param_names_for_optimizer

        unmapped_param = torch.randn(2, 2)
        inner = MagicMock(spec=torch.optim.Adam)
        inner.param_groups = [{"params": [unmapped_param]}]

        with pytest.raises(AssertionError, match="not found in model name mapping"):
            _build_param_names_for_optimizer(inner, name_by_tensor_id={})


class TestDumpLocalWeightChecksums:
    def test_does_nothing_when_disabled(self) -> None:
        args = Namespace(save_local_weight_checksum=False)
        model = [_make_mock_model_chunk(params={})]
        optimizer = MagicMock()
        optimizer.chained_optimizers = []

        dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)

    def test_dumps_when_enabled(self, tmp_path: Path) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=7)
        event_logger = EventLogger(log_dir=tmp_path, source=source)
        set_event_logger(event_logger)
        try:
            weight = torch.randn(2, 2)
            args = Namespace(save_local_weight_checksum=True)
            model = [_make_mock_model_chunk(params={"w": weight})]
            optimizer = _make_mock_optimizer_with_state_dict(params=[weight])

            with event_logger.with_context({"rollout_id": 0, "attempt": 0}):
                dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)

            event_logger.close()

            events = read_events(tmp_path)
            checksum_events = [e for e in events if isinstance(e, TrainEngineLocalWeightChecksumEvent)]
            assert len(checksum_events) == 1
            assert checksum_events[0].source.cell_index == 2
            assert checksum_events[0].source.rank_within_cell == 7
        finally:
            set_event_logger(None)
