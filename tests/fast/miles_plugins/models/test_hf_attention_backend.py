from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_config

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
