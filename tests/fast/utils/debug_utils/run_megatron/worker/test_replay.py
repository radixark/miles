from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.utils.debug_utils.run_megatron.worker.replay import (
    _ParallelRanks,
    _load_replay,
    _sp_slice,
    load_replay_data,
    save_replay_data,
    setup_replay_before_model,
)
from tests.fast.utils.debug_utils.run_megatron.conftest import make_script_args


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
        script = make_script_args(routing_replay_dump_path=Path("/dump"))
        with patch(
            "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
        ) as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is True
            assert mock_mgr.stage == "record"

    def test_load_enables_replay_forward(self) -> None:
        script = make_script_args(routing_replay_load_path=Path("/load"))
        with patch(
            "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
        ) as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is True
            assert mock_mgr.stage == "replay_forward"

    def test_neither_noop(self) -> None:
        script = make_script_args()
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

        script = make_script_args(routing_replay_dump_path=tmp_path)
        save_replay_data(script, rank=0)

        saved_files = list(tmp_path.glob("*.pt"))
        assert len(saved_files) == 1
        assert "rank0" in saved_files[0].name

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_saved_content_roundtrips(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        original = [torch.tensor([10, 20]), torch.tensor([30, 40])]
        replay = SimpleNamespace(top_indices_list=original)
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        save_replay_data(script, rank=0)

        saved_file = list(tmp_path.glob("*.pt"))[0]
        loaded = torch.load(saved_file, weights_only=False)
        assert len(loaded) == 1
        assert len(loaded[0]) == 2
        assert loaded[0][0].tolist() == [10, 20]
        assert loaded[0][1].tolist() == [30, 40]

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_noop_when_no_path(self, mock_mgr: MagicMock) -> None:
        script = make_script_args()
        save_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_asserts_rank_zero(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[torch.tensor([1, 2])])
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        with pytest.raises(AssertionError):
            save_replay_data(script, rank=1)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_empty_entries_asserts(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[])
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        with pytest.raises(AssertionError):
            save_replay_data(script, rank=0)


class TestLoadReplayData:
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_noop_when_no_path(self, mock_mgr: MagicMock) -> None:
        script = make_script_args()
        load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_no_file_raises(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        script = make_script_args(routing_replay_load_path=tmp_path)
        with pytest.raises(ValueError, match="Expected exactly 1 replay file"):
            load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_multiple_files_raises(
        self, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_mgr.filename = "routing_replay.pt"
        (tmp_path / "a_routing_replay.pt").touch()
        (tmp_path / "b_routing_replay.pt").touch()

        script = make_script_args(routing_replay_load_path=tmp_path)
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

        script = make_script_args(routing_replay_load_path=tmp_path)
        load_replay_data(script, rank=0)

        assert len(replay_obj.top_indices_list) == 2
        assert replay_obj.forward_index == 0
        assert replay_obj.backward_index == 0


class TestLoadReplaySpSlicing:
    """Test _load_replay with SP (sequence parallel) slicing."""

    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_sp_slices_tensors(
        self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path
    ) -> None:
        mock_ranks.return_value = _ParallelRanks(
            cp_size=1, cp_rank=0, tp_size=2, tp_rank=0
        )
        mock_mgr.if_sp_region = True

        replay_obj = SimpleNamespace(
            top_indices_list=[], forward_index=0, backward_index=0
        )
        mock_mgr.replays = [replay_obj]

        # 8 elements, tp_size=2, tp_rank=0 → first 4
        saved_data = [[torch.arange(8)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        _load_replay(save_path, rank=0, sequence_parallel=True)

        assert len(replay_obj.top_indices_list) == 1
        assert replay_obj.top_indices_list[0].tolist() == [0, 1, 2, 3]

    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_sp_second_rank(
        self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path
    ) -> None:
        mock_ranks.return_value = _ParallelRanks(
            cp_size=1, cp_rank=0, tp_size=2, tp_rank=1
        )
        mock_mgr.if_sp_region = True

        replay_obj = SimpleNamespace(
            top_indices_list=[], forward_index=0, backward_index=0
        )
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.arange(8)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        _load_replay(save_path, rank=1, sequence_parallel=True)

        assert replay_obj.top_indices_list[0].tolist() == [4, 5, 6, 7]

    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_sp_disabled_no_slice(
        self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path
    ) -> None:
        mock_ranks.return_value = _ParallelRanks(
            cp_size=1, cp_rank=0, tp_size=2, tp_rank=0
        )
        mock_mgr.if_sp_region = True

        replay_obj = SimpleNamespace(
            top_indices_list=[], forward_index=0, backward_index=0
        )
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.arange(8)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        # sequence_parallel=False → no sp slice even if if_sp_region=True
        _load_replay(save_path, rank=0, sequence_parallel=False)

        assert replay_obj.top_indices_list[0].tolist() == list(range(8))


class TestLoadReplayCpSlicing:
    """Test _load_replay with CP (context parallel) zigzag slicing."""

    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
    )
    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    def test_cp_calls_zigzag_slice(
        self, mock_ranks: MagicMock, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_ranks.return_value = _ParallelRanks(
            cp_size=2, cp_rank=0, tp_size=1, tp_rank=0
        )
        mock_mgr.if_sp_region = False

        replay_obj = SimpleNamespace(
            top_indices_list=[], forward_index=0, backward_index=0
        )
        mock_mgr.replays = [replay_obj]

        original_tensor = torch.arange(8)
        saved_data = [[original_tensor]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        mock_zigzag = MagicMock(return_value=torch.tensor([0, 7, 1, 6]))
        mock_cp_utils = MagicMock()
        mock_cp_utils.natural_to_zigzag_slice = mock_zigzag

        import sys
        with patch.dict(sys.modules, {
            "megatron.core.transformer.deepseek_v4_cp_utils": mock_cp_utils,
        }):
            _load_replay(save_path, rank=0, sequence_parallel=False)

        mock_zigzag.assert_called_once()
        call_kwargs = mock_zigzag.call_args[1]
        assert call_kwargs["dim"] == 0
        assert call_kwargs["cp_size"] == 2
        assert call_kwargs["cp_rank"] == 0
        assert replay_obj.top_indices_list[0].tolist() == [0, 7, 1, 6]

    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager"
    )
    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    def test_cp1_no_zigzag(
        self, mock_ranks: MagicMock, mock_mgr: MagicMock, tmp_path: Path
    ) -> None:
        mock_ranks.return_value = _ParallelRanks(
            cp_size=1, cp_rank=0, tp_size=1, tp_rank=0
        )
        mock_mgr.if_sp_region = False

        replay_obj = SimpleNamespace(
            top_indices_list=[], forward_index=0, backward_index=0
        )
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.arange(4)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        _load_replay(save_path, rank=0, sequence_parallel=False)
        assert replay_obj.top_indices_list[0].tolist() == [0, 1, 2, 3]


class TestLoadReplayMismatch:
    @patch(
        "miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks"
    )
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_replay_count_mismatch_raises(
        self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path
    ) -> None:
        mock_ranks.return_value = _ParallelRanks(
            cp_size=1, cp_rank=0, tp_size=1, tp_rank=0
        )
        mock_mgr.if_sp_region = False
        mock_mgr.replays = [
            SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0),
            SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0),
        ]

        # Save only 1 replay but model expects 2
        saved_data = [[torch.tensor([1, 2])]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        with pytest.raises(ValueError, match="has 1 replays but model expects 2"):
            _load_replay(save_path, rank=0, sequence_parallel=False)
