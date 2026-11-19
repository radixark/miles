from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_config

from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens
from miles_plugins.models import hf_attention


class TestBackendSequenceParallelAttention:
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_backend_sequence_parallel_controls_both_sdk_boundaries_around_hf_computation(
        self, monkeypatch: pytest.MonkeyPatch, sequence_parallel: bool
    ) -> None:
        """The actual forward gathers before HF attention and scatters its result."""
        args = make_trainer_config(sequence_parallel=sequence_parallel)
        original = args.model_dump()
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
        packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3]))
        group = object()
        calls = []

        def gather(value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            calls.append(("gather", kwargs))
            torch.testing.assert_close(value, hidden)
            return value + 10

        def compute(value: torch.Tensor, received: Any) -> torch.Tensor:
            calls.append(("hf", received))
            expected = hidden + 10 if sequence_parallel else hidden
            torch.testing.assert_close(value, expected.permute(1, 0, 2))
            return value + 3

        def scatter(value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            calls.append(("scatter", kwargs))
            torch.testing.assert_close(value, hidden + 13)
            return value * 2

        monkeypatch.setattr(hf_attention.mpu, "get_context_parallel_world_size", lambda: 1)
        monkeypatch.setattr(hf_attention.mpu, "get_tensor_model_parallel_group", lambda: group)
        monkeypatch.setattr(hf_attention.tensor_parallel, "gather_from_sequence_parallel_region", gather)
        monkeypatch.setattr(hf_attention.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
        layer = SimpleNamespace(args=args, hybrid_cp=False, hf_forward=compute)

        output, bias = hf_attention.HuggingfaceAttention.forward(
            layer, hidden_states=hidden, attention_mask=None, packed_seq_params=packed
        )

        assert bias is None
        torch.testing.assert_close(output, (hidden + 13) * 2 if sequence_parallel else hidden + 3)
        assert calls == (
            [
                ("gather", {"tensor_parallel_output_grad": False, "group": group}),
                ("hf", packed),
                ("scatter", {"group": group}),
            ]
            if sequence_parallel
            else [("hf", packed)]
        )
        assert args.model_dump() == original

    @pytest.mark.parametrize("native_host_metadata", [False, True])
    def test_hybrid_cp_passes_existing_host_boundaries_to_both_relayouts(self, monkeypatch, native_host_metadata):
        boundaries = (0, 4)
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
        packed = (
            PackedSeqParamsWithHostCuSeqlens(cu_seqlens_q=cu_seqlens, cu_seqlens_host=boundaries)
            if native_host_metadata
            else SimpleNamespace(cu_seqlens_q=cu_seqlens)
        )
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
        group = object()
        calls = []

        def relayout(value, received_cu, received_group, rank, cp_size, *, cu_seqlens_host):
            assert received_cu is cu_seqlens and received_group is group
            assert (rank, cp_size) == (0, 2)
            calls.append(cu_seqlens_host)
            return value

        monkeypatch.setattr(hf_attention.mpu, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(hf_attention.mpu, "get_context_parallel_rank", lambda: 0)
        monkeypatch.setattr(hf_attention.mpu, "get_context_parallel_group", lambda: group)
        monkeypatch.setattr(hf_attention, "zigzag_to_packed_shard", relayout)
        monkeypatch.setattr(hf_attention, "packed_shard_to_zigzag", relayout)
        layer = SimpleNamespace(
            args=SimpleNamespace(backend=SimpleNamespace(sequence_parallel=False), allgather_cp=False),
            hybrid_cp=True,
            hf_forward=lambda value, params: value,
        )
        output, bias = hf_attention.HuggingfaceAttention.forward(
            layer, hidden_states=hidden, attention_mask=None, packed_seq_params=packed
        )
        torch.testing.assert_close(output, hidden)
        assert bias is None
        assert calls == [boundaries if native_host_metadata else None] * 2
