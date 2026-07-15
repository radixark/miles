"""Tests for miles.utils.audit_utils.witness.module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer import ChainedOptimizer

from miles.utils.audit_utils.event_logger.models import WitnessSnapshotParamEvent
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.audit_utils.witness.module import (
    _abs_broadcast_add,
    _AbsBroadcastAdd,
    _DataWitness,
    _record_and_log_witness_param,
    _zero_witness_rows,
    install_witness,
    witness_dump_and_clear_stale,
)


class TestDataWitnessForward:
    def test_forward_does_not_change_hidden_states(self) -> None:
        """Witness output is zero, so hidden_states should be unchanged."""
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[0, 1, 2, 3]])  # [1, 4]
        hidden = torch.randn(4, 1, 8)  # [s, b, h] Megatron SBH layout
        result = witness(ids, hidden)
        assert torch.equal(result, hidden)

    def test_forward_unchanged_after_optimizer_step(self) -> None:
        witness = _DataWitness(buffer_size=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        ids = torch.tensor([[0, 1, 2]])
        hidden = torch.randn(3, 1, 8)
        result = witness(ids, hidden)
        result.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        # After optimizer update, weights are nonzero, but hidden_states still unchanged
        assert not torch.all(witness.witness.weight == 0.0)
        result2 = witness(ids, hidden)
        assert torch.equal(result2, hidden)

    def test_backward_records_gradient_on_witness_weight(self) -> None:
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[2, 5]])
        hidden = torch.randn(2, 1, 4, requires_grad=True)

        result = witness(ids, hidden)
        result.sum().backward()

        grad = witness.witness.weight.grad
        assert grad is not None
        nonzero_rows = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
        assert set(nonzero_rows) == {2, 5}

    def test_no_effect_on_main_model_gradients(self) -> None:
        """Witness should not alter gradients for upstream or downstream model parameters."""
        torch.manual_seed(42)
        embed = nn.Embedding(100, 8)
        linear = nn.Linear(8, 1)

        tokens = torch.tensor([[1, 2, 3, 4]])

        # Step 1: Compute loss without witness
        hidden = embed(tokens).transpose(0, 1).contiguous()  # [s=4, b=1, h=8]
        out_no_witness = linear(hidden).sum()
        out_no_witness.backward()
        grad_embed_no = embed.weight.grad.clone()
        grad_linear_no = linear.weight.grad.clone()

        embed.zero_grad()
        linear.zero_grad()

        # Step 2: Compute loss with witness
        hidden = embed(tokens).transpose(0, 1).contiguous()
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[0, 0, 0, 0]])
        h = witness(ids, hidden)
        out_with_witness = linear(h).sum()
        out_with_witness.backward()

        assert torch.equal(grad_embed_no, embed.weight.grad)
        assert torch.equal(grad_linear_no, linear.weight.grad)


class TestRecordAndLogWitnessParam:
    def test_logs_nonzero_weight_rows(self) -> None:
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[3] = 1.0
        witness.witness.weight.data[7] = 2.0

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.head", stale_ids=[])

            mock_logger.log.assert_called_once()
            # New API: log(event_cls, partial_dict)
            partial = mock_logger.log.call_args[0][1]
            assert set(partial["nonzero_witness_ids"]) == {3, 7}
            assert partial["instance_id"] == "pp0.head"

    def test_record_and_log_event_fields(self) -> None:
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[1] = 0.5
        witness.witness.weight.data[4] = -0.3

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.tail", stale_ids=[])

            mock_logger.log.assert_called_once()
            assert mock_logger.log.call_args[0][0] is WitnessSnapshotParamEvent
            partial = mock_logger.log.call_args[0][1]
            assert partial["instance_id"] == "pp0.tail"
            assert set(partial["nonzero_witness_ids"]) == {1, 4}


# ---------------------------------------------------------------------------
# Fake GPTModel for install_witness / forward integration tests
# ---------------------------------------------------------------------------


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_tensor: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class _FakeGPTModel(nn.Module):
    def __init__(self, *, pre_process: bool = True) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.decoder = _FakeDecoder()
        self.embedding = nn.Embedding(100, 16)

    def forward(self, input_ids: torch.Tensor, witness_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_process:
            # Megatron decoders use sequence-first [s, b, h] layout, which is what
            # _DataWitness expects (it transposes its own output to [s, b, 1]).
            decoder_input = self.embedding(input_ids).transpose(0, 1).contiguous()
        else:
            decoder_input = None

        if hasattr(self, "local_head_witness") and witness_ids is not None:
            if decoder_input is not None:
                decoder_input = self.local_head_witness(witness_ids, decoder_input)
            else:
                self.decoder.input_tensor = self.local_head_witness(witness_ids, self.decoder.input_tensor)

        if decoder_input is None:
            decoder_input = self.decoder.input_tensor

        return self.decoder(hidden_states=decoder_input)


class TestInstallWitness:
    def test_witness_is_submodule(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        assert "local_head_witness" in dict(model.named_modules())
        assert "local_tail_witness" in dict(model.named_modules())

    def test_witness_in_parameters(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        param_names = [name for name, _ in model.named_parameters()]
        assert any("local_head_witness" in name for name in param_names)
        assert any("local_tail_witness" in name for name in param_names)

    def test_forward_adds_zero(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out_no = model(tokens)
        out_with = model(tokens, witness_ids=torch.tensor([[0, 1, 2]]))
        assert torch.equal(out_no, out_with)

    def test_forward_produces_gradient(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out = model(tokens, witness_ids=torch.tensor([[5, 5, 5]]))
        out.sum().backward()
        grad = model.local_head_witness.witness.weight.grad
        assert grad is not None
        assert 5 in grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_no_witness_ids_no_effect(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        out = model(torch.tensor([[1, 2, 3]]))
        assert out is not None

    def test_witness_in_state_dict(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        sd = model.state_dict()
        assert any("local_head_witness" in k for k in sd)
        assert any("local_tail_witness" in k for k in sd)

    def test_checkpoint_roundtrip(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        model.local_head_witness.witness.weight.data.fill_(42.0)
        sd = model.state_dict()

        model2 = _FakeGPTModel()
        install_witness(model2, buffer_size=10)
        model2.load_state_dict(sd)
        assert torch.equal(model2.local_head_witness.witness.weight.data, model.local_head_witness.witness.weight.data)

    def test_disabled_no_submodule(self) -> None:
        model = _FakeGPTModel()
        assert not hasattr(model, "local_head_witness")

    def test_middle_pp_stage_modifies_input_tensor(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        # Sequence-first: [s=4, b=1, h=16].
        hidden = torch.randn(4, 1, 16)
        model.decoder.input_tensor = hidden.clone()
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[0, 1, 2, 3]]))
        assert torch.equal(out, hidden)

    def test_middle_pp_stage_produces_gradient(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        # Sequence-first: [s=4, b=1, h=16].
        model.decoder.input_tensor = torch.randn(4, 1, 16, requires_grad=True)
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[5, 5, 5, 5]]))
        out.sum().backward()
        assert 5 in model.local_head_witness.witness.weight.grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_forward_bitwise_zero_bf16(self) -> None:
        witness = _DataWitness(buffer_size=10).to(dtype=torch.bfloat16)
        witness.witness.weight.data.fill_(3.14)
        ids = torch.tensor([[0, 1, 2]])
        hidden = torch.randn(3, 1, 8, dtype=torch.bfloat16)
        result = witness(ids, hidden)
        assert torch.equal(result, hidden)


class TestZeroWitnessRows:
    def test_zero_witness_rows_clears_distributed_optimizer_shard_rows(self) -> None:
        """Stale rows inside the local dist-opt shard are zeroed in the fp32 shard and its Adam state."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data.fill_(1.0)
        dist_opt, main_shard = _make_distributed_optimizer(witness.witness.weight, start=4, end=10)
        optimizer = ChainedOptimizer([dist_opt])

        _zero_witness_rows(witness=witness, idx=torch.tensor([2, 5, 7]), optimizer=optimizer)

        weight = witness.witness.weight
        for stale_row in (2, 5, 7):
            assert weight.data[stale_row].item() == 0.0
        assert weight.data[4].item() == 1.0
        # The shard covers rows [4, 10): row 5 -> local 1, row 7 -> local 3; row 2 is outside.
        state = dist_opt.optimizer.state[main_shard]
        for tensor in (main_shard, state["exp_avg"], state["exp_avg_sq"]):
            assert tensor[1].item() == 0.0
            assert tensor[3].item() == 0.0
            assert tensor[0].item() == 1.0

    def test_zero_witness_rows_skips_distributed_optimizer_without_the_param(self) -> None:
        """A chained dist-opt instance owning no shard of the witness param is skipped, the owner still clears."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data.fill_(1.0)
        dist_opt, main_shard = _make_distributed_optimizer(witness.witness.weight, start=0, end=10)
        expert_opt = DistributedOptimizer.__new__(DistributedOptimizer)
        expert_opt.config = dist_opt.config
        expert_opt.model_param_gbuf_map = {}
        optimizer = ChainedOptimizer([dist_opt, expert_opt])

        _zero_witness_rows(witness=witness, idx=torch.tensor([3]), optimizer=optimizer)

        assert witness.witness.weight.data[3].item() == 0.0
        assert main_shard[3].item() == 0.0

    def test_zero_witness_rows_rejects_non_distributed_optimizer(self) -> None:
        """Any optimizer other than (chained) DistributedOptimizer is rejected outright."""
        witness = _DataWitness(buffer_size=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.01)

        with pytest.raises(AssertionError, match="unsupported optimizer"):
            _zero_witness_rows(witness=witness, idx=torch.tensor([1]), optimizer=optimizer)

    def test_zero_witness_rows_raises_when_populated_state_misses_witness_shard(self) -> None:
        """A populated optimizer state without the witness shard entry fails loudly instead of silently skipping."""
        witness = _DataWitness(buffer_size=10)
        dist_opt, _main_shard = _make_distributed_optimizer(witness.witness.weight, start=0, end=10)
        unrelated = torch.ones(4)
        dist_opt.optimizer.state = {unrelated: {"exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)}}

        with pytest.raises(AssertionError, match="witness main shard missing"):
            _zero_witness_rows(witness=witness, idx=torch.tensor([1]), optimizer=ChainedOptimizer([dist_opt]))


# ---------------------------------------------------------------------------
# Helpers for witness_dump_and_clear_stale tests
# ---------------------------------------------------------------------------


def _make_fake_chunk(buffer_size: int = 10) -> nn.Module:
    """Create a fake model chunk with .module.local_head_witness and .module.local_tail_witness."""
    inner = nn.Module()
    inner.local_head_witness = _DataWitness(buffer_size=buffer_size)
    inner.local_tail_witness = _DataWitness(buffer_size=buffer_size)
    chunk = nn.Module()
    chunk.module = inner
    return chunk


def _make_distributed_optimizer(
    model_weight: nn.Parameter,
    *,
    start: int,
    end: int,
) -> tuple[DistributedOptimizer, torch.Tensor]:
    """Build a minimal fake DistributedOptimizer owning rows [start, end) of model_weight."""
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    main_shard = model_weight.detach().view(-1)[start:end].clone().float().requires_grad_()
    inner = torch.optim.Adam([main_shard], lr=0.01)
    inner.state[main_shard] = {
        "exp_avg": torch.ones(end - start),
        "exp_avg_sq": torch.ones(end - start),
    }
    optimizer.optimizer = inner
    optimizer.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        optimizer_cpu_offload=False,
        optimizer="adam",
    )
    optimizer.model_param_gbuf_map = {model_weight: (0, torch.bfloat16, 0)}
    optimizer.gbuf_ranges = [{torch.bfloat16: [{"param_map": {model_weight: {"param": Range(start, end)}}}]}]
    optimizer.model_param_group_index_map = {model_weight: (0, 0)}
    return optimizer, main_shard


def _make_chained_optimizer_for_witnesses(model: list[nn.Module]) -> ChainedOptimizer:
    """Build a fake ChainedOptimizer with one fully-owning dist-opt instance per witness weight."""
    weights = [
        getattr(chunk.module, attr).witness.weight
        for chunk in model
        for attr in ("local_head_witness", "local_tail_witness")
    ]
    return ChainedOptimizer([_make_distributed_optimizer(w, start=0, end=w.numel())[0] for w in weights])


class TestWitnessDumpAndClearStale:
    def test_witness_dump_and_clear_stale_logs_all_witnesses(self) -> None:
        """2 chunks x 2 witnesses = 4 log events with correct instance_ids."""
        chunk0 = _make_fake_chunk()
        chunk1 = _make_fake_chunk()
        chunk0.module.local_head_witness.witness.weight.data[1] = 1.0
        chunk0.module.local_tail_witness.witness.weight.data[2] = 1.0
        chunk1.module.local_head_witness.witness.weight.data[3] = 1.0
        chunk1.module.local_tail_witness.witness.weight.data[4] = 1.0

        model = [chunk0, chunk1]
        optimizer = _make_chained_optimizer_for_witnesses(model)
        witness_info = WitnessInfo(witness_ids=[1, 2, 3, 4], stale_ids=[5, 6])

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger, patch(
            "miles.utils.audit_utils.witness.module.get_parallel_state"
        ) as mock_get_parallel_state:
            mock_get_parallel_state.return_value.pp.rank = 0
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

            assert mock_logger.log.call_count == 4
            logged_instance_ids = [call[0][1]["instance_id"] for call in mock_logger.log.call_args_list]
            assert logged_instance_ids == [
                "pp0_chunk0.local_head",
                "pp0_chunk0.local_tail",
                "pp0_chunk1.local_head",
                "pp0_chunk1.local_tail",
            ]

            logged_stale_ids = [call[0][1]["stale_ids"] for call in mock_logger.log.call_args_list]
            for stale in logged_stale_ids:
                assert stale == [5, 6]

    def test_witness_dump_and_clear_stale_clears_stale_rows(self) -> None:
        """Stale IDs should have their weight rows zeroed after the call."""
        chunk = _make_fake_chunk(buffer_size=10)
        chunk.module.local_head_witness.witness.weight.data.fill_(1.0)
        chunk.module.local_tail_witness.witness.weight.data.fill_(1.0)

        model = [chunk]
        optimizer = _make_chained_optimizer_for_witnesses(model)
        witness_info = WitnessInfo(witness_ids=[0], stale_ids=[3, 7])

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger, patch(
            "miles.utils.audit_utils.witness.module.get_parallel_state"
        ) as mock_get_parallel_state:
            mock_get_parallel_state.return_value.pp.rank = 0
            mock_get_logger.return_value = MagicMock()
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        for witness_attr in ("local_head_witness", "local_tail_witness"):
            witness = getattr(chunk.module, witness_attr)
            assert witness.witness.weight.data[3].item() == 0.0
            assert witness.witness.weight.data[7].item() == 0.0
            assert witness.witness.weight.data[0].item() == 1.0

    def test_witness_dump_and_clear_stale_empty_stale_ids(self) -> None:
        """Empty stale_ids should not trigger any zeroing."""
        chunk = _make_fake_chunk(buffer_size=10)
        chunk.module.local_head_witness.witness.weight.data.fill_(1.0)
        chunk.module.local_tail_witness.witness.weight.data.fill_(1.0)

        model = [chunk]
        optimizer = _make_chained_optimizer_for_witnesses(model)
        witness_info = WitnessInfo(witness_ids=[0], stale_ids=[])

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger, patch(
            "miles.utils.audit_utils.witness.module.get_parallel_state"
        ) as mock_get_parallel_state:
            mock_get_parallel_state.return_value.pp.rank = 0
            mock_get_logger.return_value = MagicMock()
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        for witness_attr in ("local_head_witness", "local_tail_witness"):
            witness = getattr(chunk.module, witness_attr)
            assert torch.all(witness.witness.weight.data == 1.0)

    def test_record_and_log_witness_param_includes_stale_ids(self) -> None:
        """Log event should contain the correct stale_ids field."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[2] = 1.0

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.head", stale_ids=[8, 9])

            mock_logger.log.assert_called_once()
            partial = mock_logger.log.call_args[0][1]
            assert partial["stale_ids"] == [8, 9]


class TestAbsBroadcastAddForward:
    def test_forward_value_matches_plain_addition(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 2, 1)
        result = _abs_broadcast_add(hidden, addend)
        expected = hidden + addend
        assert torch.equal(result, expected)

    def test_forward_preserves_zero_addend(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.zeros(4, 2, 1)
        result = _abs_broadcast_add(hidden, addend)
        assert torch.equal(result, hidden)

    def test_forward_assert_addend_last_dim_must_be_1(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 2, 3)
        with pytest.raises(AssertionError, match="addend last dim must be 1"):
            _abs_broadcast_add(hidden, addend)

    def test_forward_assert_leading_dims_must_match(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 3, 1)  # second dim differs
        with pytest.raises(AssertionError, match="must match on all dims except last"):
            _abs_broadcast_add(hidden, addend)

    def test_forward_assert_ndim_must_match(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(2, 1)  # 2D vs 3D
        with pytest.raises(AssertionError, match="must match on all dims except last"):
            _abs_broadcast_add(hidden, addend)


class TestAbsBroadcastAddBackwardHiddenStates:
    def test_hidden_states_gradient_is_pass_through(self) -> None:
        hidden = torch.randn(4, 2, 8, requires_grad=True)
        addend = torch.randn(4, 2, 1, requires_grad=True)
        result = _abs_broadcast_add(hidden, addend)
        loss = result.sum()
        loss.backward()
        # hidden gradient = all ones (pass-through from sum)
        assert torch.equal(hidden.grad, torch.ones_like(hidden))


class TestAbsBroadcastAddBackwardAddend:
    def test_addend_gradient_is_abs_sum_over_last_dim(self) -> None:
        hidden = torch.randn(3, 2, 4, requires_grad=True)
        addend = torch.zeros(3, 2, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        # Use a loss that produces a known gradient at result
        upstream_grad = torch.tensor(
            [
                [[1.0, -2.0, 3.0, -4.0], [0.5, -0.5, 0.5, -0.5]],
                [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                [[0.0, 0.0, 0.0, 0.0], [2.0, -1.0, 0.5, -0.3]],
            ]
        )
        result.backward(upstream_grad)

        # Expected addend grad: abs(upstream_grad).sum(dim=-1, keepdim=True)
        expected = upstream_grad.abs().sum(dim=-1, keepdim=True)
        assert torch.allclose(addend.grad, expected)

    def test_addend_gradient_no_cancellation_with_mixed_signs(self) -> None:
        """The key property: mixed-sign gradients don't cancel to zero."""
        hidden = torch.randn(1, 1, 8, requires_grad=True)
        addend = torch.zeros(1, 1, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        # Gradient with equal positive and negative values (sums to zero normally)
        upstream_grad = torch.tensor([[[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]])
        result.backward(upstream_grad)

        # Plain broadcast backward would give sum = 0
        assert upstream_grad.sum(dim=-1, keepdim=True).item() == 0.0
        # But abs broadcast gives sum of absolute values = 8
        assert addend.grad.item() == 8.0

    def test_addend_gradient_matches_plain_sum_when_all_positive(self) -> None:
        """When all gradients are positive, abs().sum() == sum()."""
        hidden = torch.randn(2, 1, 4, requires_grad=True)
        addend = torch.zeros(2, 1, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        upstream_grad = torch.abs(torch.randn(2, 1, 4))  # all positive
        result.backward(upstream_grad)

        expected = upstream_grad.sum(dim=-1, keepdim=True)
        assert torch.allclose(addend.grad, expected)

    def test_addend_gradient_always_non_negative(self) -> None:
        """abs().sum() is always >= 0."""
        hidden = torch.randn(10, 5, 16, requires_grad=True)
        addend = torch.zeros(10, 5, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        upstream_grad = torch.randn(10, 5, 16)
        result.backward(upstream_grad)

        assert (addend.grad >= 0).all()


class TestAbsBroadcastAddGradientFlow:
    def test_gradient_flows_through_to_embedding(self) -> None:
        """End-to-end: gradient reaches embedding weight via abs broadcast add."""
        vocab_size = 16
        hidden_dim = 8
        seq_len = 4

        embedding = torch.nn.Embedding(vocab_size, 1)
        torch.nn.init.zeros_(embedding.weight)
        output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        witness_ids = torch.tensor([[0, 1, 2, 3]])  # [1, seq_len]
        w = embedding(witness_ids)  # [1, 4, 1]
        out = w - w.detach()

        hidden_states = torch.randn(seq_len, 1, hidden_dim, requires_grad=True)
        tail_out = out.transpose(0, 1).contiguous()  # [4, 1, 1]

        combined = _abs_broadcast_add(hidden_states, tail_out)
        logits = output_layer(combined)
        loss = logits.sum()
        loss.backward()

        # All 4 witness_ids should have nonzero gradient
        nonzero_rows = (embedding.weight.grad.abs() > 0).squeeze(-1)
        assert nonzero_rows[:4].all(), f"Expected rows 0-3 nonzero, got {embedding.weight.grad[:4]}"

    def test_gradient_nonzero_even_when_plain_broadcast_cancels(self) -> None:
        """Simulates the exact scenario: output_layer gradient cancels under plain broadcast."""
        vocab_size = 4
        hidden_dim = 4
        seq_len = 2

        embedding = torch.nn.Embedding(8, 1)
        torch.nn.init.zeros_(embedding.weight)

        # Construct output_layer weight where ROW sums are constant (each row sums to 4),
        # so that dL/d_combined.sum_over_last_dim = sum_i dL/dlogit_i * row_sum_i = 0
        # (using softmax-loss sum-to-zero property). Row values differ so per-element
        # grads are nonzero — only their sum cancels. This is exactly the scenario
        # where plain broadcast (sum) cancels but abs broadcast (abs.sum) preserves.
        W = torch.tensor([[2.0, 0.0, 1.0, 1.0], [0.0, 2.0, 1.0, 1.0], [1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 0.0, 2.0]])
        output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        output_layer.weight.data = W

        witness_ids = torch.tensor([[0, 1]])
        w = embedding(witness_ids)
        out = w - w.detach()

        hidden_states = torch.randn(seq_len, 1, hidden_dim, requires_grad=True)
        tail_out = out.transpose(0, 1).contiguous()

        # With abs broadcast add, gradient should be nonzero
        combined = _abs_broadcast_add(hidden_states, tail_out)
        logits = output_layer(combined)
        targets = torch.tensor([[0, 1]])
        log_probs = torch.nn.functional.log_softmax(logits.squeeze(1), dim=-1)
        loss = -log_probs.gather(1, targets.T).sum()
        loss.backward()

        assert embedding.weight.grad is not None
        assert (
            embedding.weight.grad[:2].abs() > 0
        ).all(), f"Expected nonzero grad for witness rows 0,1, got {embedding.weight.grad[:2]}"


class TestAbsBroadcastAddDoubleBackward:
    def test_gradcheck(self) -> None:
        """Verify numerical gradient correctness with torch.autograd.gradcheck."""
        hidden = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
        addend = torch.randn(2, 2, 1, dtype=torch.float64, requires_grad=True)
        # gradcheck only tests hidden_states gradient (which is pass-through)
        # For addend, abs() is not differentiable at 0, so we test separately
        assert torch.autograd.gradcheck(
            lambda h: _AbsBroadcastAdd.apply(h, addend.detach()),
            (hidden,),
        )
