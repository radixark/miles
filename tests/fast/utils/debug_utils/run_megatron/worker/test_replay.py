from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.utils.debug_utils.run_megatron.worker.replay import (
    _sp_slice,
    load_replay_data,
    save_replay_data,
    setup_replay_before_model,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs


def _make_script_args(**overrides: object) -> WorkerScriptArgs:
    defaults = dict(
        hf_checkpoint=Path("/fake/hf"),
        token_ids_file=Path("/tmp/tokens.json"),
    )
    defaults.update(overrides)
    return WorkerScriptArgs(**defaults)  # type: ignore[arg-type]


class TestSpSlice:
    def test_slices_evenly(self) -> None:
        tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        result = _sp_slice(tensor, tp_size=2, tp_rank=0)
        assert result.tolist() == [1, 2, 3, 4]

    def test_second_rank(self) -> None:
        tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        result = _sp_slice(tensor, tp_size=2, tp_rank=1)
        assert result.tolist() == [5, 6, 7, 8]

    def test_not_divisible_raises(self) -> None:
        tensor = torch.tensor([1, 2, 3, 4, 5])
        with pytest.raises(AssertionError, match="not divisible"):
            _sp_slice(tensor, tp_size=2, tp_rank=0)

    def test_single_rank_full(self) -> None:
        tensor = torch.tensor([1, 2, 3, 4])
        result = _sp_slice(tensor, tp_size=1, tp_rank=0)
        assert result.tolist() == [1, 2, 3, 4]


class TestSetupReplayBeforeModel:
    def test_dump_enables_record(self) -> None:
        script = _make_script_args(routing_replay_dump_path=Path("/dump"))
        with patch(
            "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
        ) as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is True
            assert mock_mgr.stage == "record"

    def test_load_enables_replay_forward(self) -> None:
        script = _make_script_args(routing_replay_load_path=Path("/load"))
        with patch(
            "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
        ) as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is True
            assert mock_mgr.stage == "replay_forward"

    def test_neither_noop(self) -> None:
        script = _make_script_args()
        with patch(
            "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
        ) as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is False
            assert mock_mgr.stage == "fallthrough"


class TestSaveReplayData:
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_saves_file(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[torch.tensor([1, 2])])
        mock_mgr.replays = [replay]

        script = _make_script_args(routing_replay_dump_path=tmp_path)
        save_replay_data(script, rank=0)

        saved_files = list(tmp_path.glob("*.pt"))
        assert len(saved_files) == 1
        assert "rank0" in saved_files[0].name

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_noop_when_no_path(self, mock_mgr: MagicMock) -> None:
        script = _make_script_args()
        save_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_asserts_rank_zero(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[torch.tensor([1, 2])])
        mock_mgr.replays = [replay]

        script = _make_script_args(routing_replay_dump_path=tmp_path)
        with pytest.raises(AssertionError):
            save_replay_data(script, rank=1)


class TestLoadReplayData:
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_noop_when_no_path(self, mock_mgr: MagicMock) -> None:
        script = _make_script_args()
        load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_no_file_raises(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        script = _make_script_args(routing_replay_load_path=tmp_path)
        with pytest.raises(ValueError, match="Expected exactly 1 replay file"):
            load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_multiple_files_raises(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        (tmp_path / "a_routing_replay.pt").touch()
        (tmp_path / "b_routing_replay.pt").touch()

        script = _make_script_args(routing_replay_load_path=tmp_path)
        with pytest.raises(ValueError, match="Expected exactly 1 replay file"):
            load_replay_data(script, rank=0)

    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_loads_data(
        self,
        mock_mgr: MagicMock,
        mock_ranks: MagicMock,
        tmp_path: Path,
    ) -> None:
        from miles.utils.debug_utils.run_megatron.worker.replay import _ParallelRanks

        mock_ranks.return_value = _ParallelRanks(
            cp_size=1, cp_rank=0, tp_size=1, tp_rank=0
        )
        mock_mgr.filename = "routing_replay.pt"
        mock_mgr.if_sp_region = False

        replay_obj = SimpleNamespace(
            top_indices_list=[], forward_index=0, backward_index=0
        )
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.tensor([10, 20]), torch.tensor([30, 40])]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        script = _make_script_args(routing_replay_load_path=tmp_path)
        load_replay_data(script, rank=0)

        assert len(replay_obj.top_indices_list) == 2
        assert replay_obj.forward_index == 0
        assert replay_obj.backward_index == 0
