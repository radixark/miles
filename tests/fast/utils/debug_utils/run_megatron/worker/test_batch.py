from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.cp_utils import slice_with_cp
from miles.utils.debug_utils.run_megatron.worker.batch import (
    _build_labels,
    loss_func,
    prepare_batch,
)


def _zigzag_slice(tokens: torch.Tensor, *, cp_rank: int, cp_size: int) -> torch.Tensor:
    return slice_with_cp(
        tokens,
        pad_value=0,
        parallel_state=SimpleNamespace(cp_rank=cp_rank, cp_size=cp_size),
        qkv_format="bshd",
        max_seq_len=len(tokens),
    )


# ---------------------------------------------------------------------------
# TestBuildLabels
# ---------------------------------------------------------------------------


class TestBuildLabels:
    def test_cp1_next_token_shift(self) -> None:
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        position_ids = torch.arange(3).unsqueeze(0)
        global_input_ids = input_ids.clone()

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=1,
        )
        assert labels.shape == (1, 3)
        assert labels.dtype == torch.long
        assert labels[0].tolist() == [20, 30, -100]

    def test_cp1_batch_size_2(self) -> None:
        input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long)
        position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)
        global_input_ids = input_ids.clone()

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=1,
        )
        assert labels.shape == (2, 3)
        assert labels[0].tolist() == [20, 30, -100]
        assert labels[1].tolist() == [50, 60, -100]

    def test_cp1_single_token(self) -> None:
        input_ids = torch.tensor([[99]], dtype=torch.long)
        position_ids = torch.tensor([[0]])
        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=input_ids.clone(),
            cp_size=1,
        )
        assert labels[0].tolist() == [-100]

    def test_cp_gt1_uses_position_ids(self) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[0, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels.shape == (1, 2)
        assert labels[0, 0].item() == 20  # next token after position 0
        assert labels[0, 1].item() == -100  # position 3 is last -> ignored

    def test_cp_gt1_middle_positions(self) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        position_ids = torch.tensor([[1, 4]], dtype=torch.long)
        input_ids = torch.tensor([[20, 50]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0, 0].item() == 30  # next after pos 1
        assert labels[0, 1].item() == 60  # next after pos 4

    def test_cp_gt1_second_to_last_position_is_valid(self) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[2, 3]], dtype=torch.long)
        input_ids = torch.tensor([[30, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0, 0].item() == 40  # pos 2 -> next is pos 3 = token 40
        assert labels[0, 1].item() == -100  # pos 3 is last

    def test_cp_gt1_batch_size_2(self) -> None:
        global_input_ids = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long
        )
        position_ids = torch.tensor([[0, 3], [0, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 40], [50, 80]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels.shape == (2, 2)
        assert labels[0].tolist() == [20, -100]
        assert labels[1].tolist() == [60, -100]

    def test_cp_gt1_all_positions_valid(self) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        position_ids = torch.tensor([[0, 2, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 30, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0].tolist() == [20, 40, 50]  # all valid, none is last pos

    def test_cp_gt1_all_positions_are_last(self) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[3, 3]], dtype=torch.long)
        input_ids = torch.tensor([[40, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0].tolist() == [-100, -100]


# ---------------------------------------------------------------------------
# TestLossFunc
# ---------------------------------------------------------------------------


class TestLossFunc:
    def test_return_type(self) -> None:
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, -100]], dtype=torch.long)
        result = loss_func(labels=labels, output_tensor=logits)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)
        assert "loss" in result[1]

    def test_loss_scalar(self) -> None:
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss, metrics = loss_func(labels=labels, output_tensor=logits)
        assert loss.dim() == 0
        assert metrics["loss"].dim() == 0

    def test_loss_finite(self) -> None:
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert torch.isfinite(loss)

    def test_loss_detached_in_metrics(self) -> None:
        logits = torch.randn(1, 4, 10, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        _, metrics = loss_func(labels=labels, output_tensor=logits)
        assert not metrics["loss"].requires_grad

    def test_ignores_neg100(self) -> None:
        logits = torch.randn(1, 4, 10)
        all_masked = torch.full((1, 4), -100, dtype=torch.long)
        loss, _ = loss_func(labels=all_masked, output_tensor=logits)
        assert torch.isnan(loss) or loss.item() == 0.0

    def test_perfect_prediction(self) -> None:
        vocab_size = 5
        logits = torch.full((1, 3, vocab_size), -100.0)
        labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
        for pos in range(3):
            logits[0, pos, labels[0, pos]] = 100.0
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert loss.item() < 0.01

    def test_batch_size_2(self) -> None:
        vocab_size = 10
        logits = torch.randn(2, 4, vocab_size)
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# TestPrepareBatch — cp_size=1
# ---------------------------------------------------------------------------


class TestPrepareBatchCP1:
    def test_all_keys_present(self) -> None:
        batch = prepare_batch(token_ids=list(range(8)), batch_size=1, device="cpu")
        expected_keys = {"input_ids", "position_ids", "attention_mask", "labels", "global_input_ids"}
        assert set(batch.keys()) == expected_keys

    def test_all_tensors_long_except_mask(self) -> None:
        batch = prepare_batch(token_ids=list(range(8)), batch_size=1, device="cpu")
        assert batch["input_ids"].dtype == torch.long
        assert batch["position_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["global_input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.bool

    def test_shapes(self) -> None:
        token_ids = list(range(8))
        batch = prepare_batch(token_ids=token_ids, batch_size=2, device="cpu")
        assert batch["input_ids"].shape == (2, 8)
        assert batch["position_ids"].shape == (2, 8)
        assert batch["attention_mask"].shape == (2, 1, 8, 8)
        assert batch["labels"].shape == (2, 8)
        assert batch["global_input_ids"].shape == (2, 8)

    def test_input_ids_values(self) -> None:
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["input_ids"][0].tolist() == [10, 20, 30]

    def test_position_ids_sequential(self) -> None:
        token_ids = list(range(5))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["position_ids"][0].tolist() == [0, 1, 2, 3, 4]

    def test_causal_mask(self) -> None:
        token_ids = list(range(4))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        mask = batch["attention_mask"][0, 0]
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask, expected)

    def test_labels_next_token(self) -> None:
        token_ids = [10, 20, 30, 40]
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["labels"][0].tolist() == [20, 30, 40, -100]

    def test_global_input_ids_equals_input_ids(self) -> None:
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=2, device="cpu")
        assert torch.equal(batch["global_input_ids"], batch["input_ids"])

    def test_batch_dim_broadcast(self) -> None:
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=3, device="cpu")
        for i in range(3):
            assert batch["input_ids"][i].tolist() == [10, 20, 30]
            assert batch["position_ids"][i].tolist() == [0, 1, 2]

    def test_single_token(self) -> None:
        batch = prepare_batch(token_ids=[42], batch_size=1, device="cpu")
        assert batch["input_ids"][0].tolist() == [42]
        assert batch["labels"][0].tolist() == [-100]
        assert batch["attention_mask"].shape == (1, 1, 1, 1)
        assert batch["attention_mask"][0, 0, 0, 0].item() is True


# ---------------------------------------------------------------------------
# TestPrepareBatch — cp_size > 1 (zigzag)
# ---------------------------------------------------------------------------


class TestPrepareBatchZigzag:
    """Tests that exercise the real slice_with_cp zigzag CP path."""

    @pytest.fixture()
    def seq8_cp2(self) -> dict[str, dict[str, torch.Tensor]]:
        """Prepare batches for all ranks with seq_len=8, cp_size=2.

        Zigzag with cp_size=2, seq_len=8:
          chunk_size = ceil(8 / 4) = 2
          rank 0: tokens[0:2] + tokens[6:8] = positions [0,1,6,7]
          rank 1: tokens[2:4] + tokens[4:6] = positions [2,3,4,5]
        """
        token_ids = [100, 200, 300, 400, 500, 600, 700, 800]
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for rank in range(2):
            batches[f"rank{rank}"] = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=2,
                device="cpu",
            )
        return batches

    @pytest.fixture()
    def seq16_cp4(self) -> dict[str, dict[str, torch.Tensor]]:
        """Prepare batches for all ranks with seq_len=16, cp_size=4.

        chunk_size = ceil(16 / 8) = 2
        rank 0: tokens[0:2] + tokens[14:16]
        rank 1: tokens[2:4] + tokens[12:14]
        rank 2: tokens[4:6] + tokens[10:12]
        rank 3: tokens[6:8] + tokens[8:10]
        """
        token_ids = list(range(1000, 1016))
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for rank in range(4):
            batches[f"rank{rank}"] = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=4,
                device="cpu",
            )
        return batches

    # -- shapes --

    def test_local_seq_len_is_half(self, seq8_cp2: dict) -> None:
        for rank_key in seq8_cp2:
            batch = seq8_cp2[rank_key]
            assert batch["input_ids"].shape == (1, 4)
            assert batch["position_ids"].shape == (1, 4)
            assert batch["labels"].shape == (1, 4)
            assert batch["attention_mask"].shape == (1, 1, 4, 4)

    def test_global_input_ids_shape_unchanged(self, seq8_cp2: dict) -> None:
        for rank_key in seq8_cp2:
            assert seq8_cp2[rank_key]["global_input_ids"].shape == (1, 8)

    def test_cp4_local_seq_len(self, seq16_cp4: dict) -> None:
        for rank_key in seq16_cp4:
            batch = seq16_cp4[rank_key]
            assert batch["input_ids"].shape == (1, 4)
            assert batch["position_ids"].shape == (1, 4)

    # -- zigzag token assignment --

    def test_rank0_tokens_cp2(self, seq8_cp2: dict) -> None:
        batch = seq8_cp2["rank0"]
        assert batch["input_ids"][0].tolist() == [100, 200, 700, 800]

    def test_rank1_tokens_cp2(self, seq8_cp2: dict) -> None:
        batch = seq8_cp2["rank1"]
        assert batch["input_ids"][0].tolist() == [300, 400, 500, 600]

    def test_rank0_tokens_cp4(self, seq16_cp4: dict) -> None:
        batch = seq16_cp4["rank0"]
        assert batch["input_ids"][0].tolist() == [1000, 1001, 1014, 1015]

    def test_rank1_tokens_cp4(self, seq16_cp4: dict) -> None:
        batch = seq16_cp4["rank1"]
        assert batch["input_ids"][0].tolist() == [1002, 1003, 1012, 1013]

    def test_rank2_tokens_cp4(self, seq16_cp4: dict) -> None:
        batch = seq16_cp4["rank2"]
        assert batch["input_ids"][0].tolist() == [1004, 1005, 1010, 1011]

    def test_rank3_tokens_cp4(self, seq16_cp4: dict) -> None:
        batch = seq16_cp4["rank3"]
        assert batch["input_ids"][0].tolist() == [1006, 1007, 1008, 1009]

    # -- zigzag position assignment --

    def test_rank0_positions_cp2(self, seq8_cp2: dict) -> None:
        assert seq8_cp2["rank0"]["position_ids"][0].tolist() == [0, 1, 6, 7]

    def test_rank1_positions_cp2(self, seq8_cp2: dict) -> None:
        assert seq8_cp2["rank1"]["position_ids"][0].tolist() == [2, 3, 4, 5]

    def test_rank0_positions_cp4(self, seq16_cp4: dict) -> None:
        assert seq16_cp4["rank0"]["position_ids"][0].tolist() == [0, 1, 14, 15]

    def test_rank3_positions_cp4(self, seq16_cp4: dict) -> None:
        assert seq16_cp4["rank3"]["position_ids"][0].tolist() == [6, 7, 8, 9]

    # -- all ranks together cover the full sequence --

    def test_all_ranks_cover_full_sequence_cp2(self, seq8_cp2: dict) -> None:
        all_positions = set()
        for rank_key in seq8_cp2:
            all_positions.update(seq8_cp2[rank_key]["position_ids"][0].tolist())
        assert all_positions == set(range(8))

    def test_all_ranks_cover_full_sequence_cp4(self, seq16_cp4: dict) -> None:
        all_positions = set()
        for rank_key in seq16_cp4:
            all_positions.update(seq16_cp4[rank_key]["position_ids"][0].tolist())
        assert all_positions == set(range(16))

    def test_all_ranks_cover_all_tokens_cp2(self, seq8_cp2: dict) -> None:
        all_tokens = []
        for rank_key in seq8_cp2:
            all_tokens.extend(seq8_cp2[rank_key]["input_ids"][0].tolist())
        assert sorted(all_tokens) == [100, 200, 300, 400, 500, 600, 700, 800]

    def test_no_position_overlap_between_ranks_cp4(self, seq16_cp4: dict) -> None:
        seen: set[int] = set()
        for rank_key in seq16_cp4:
            positions = set(seq16_cp4[rank_key]["position_ids"][0].tolist())
            assert seen.isdisjoint(positions), f"Overlap at {seen & positions}"
            seen.update(positions)

    # -- global_input_ids is the same for all ranks --

    def test_global_input_ids_same_across_ranks(self, seq8_cp2: dict) -> None:
        expected = [100, 200, 300, 400, 500, 600, 700, 800]
        for rank_key in seq8_cp2:
            assert seq8_cp2[rank_key]["global_input_ids"][0].tolist() == expected

    # -- labels correctness under zigzag --

    def test_labels_rank0_cp2(self, seq8_cp2: dict) -> None:
        batch = seq8_cp2["rank0"]
        labels = batch["labels"][0].tolist()
        # positions [0, 1, 6, 7]
        # pos 0 -> next token at pos 1 = 200
        # pos 1 -> next token at pos 2 = 300
        # pos 6 -> next token at pos 7 = 800
        # pos 7 -> last position -> -100
        assert labels == [200, 300, 800, -100]

    def test_labels_rank1_cp2(self, seq8_cp2: dict) -> None:
        batch = seq8_cp2["rank1"]
        labels = batch["labels"][0].tolist()
        # positions [2, 3, 4, 5]
        # pos 2 -> next at pos 3 = 400
        # pos 3 -> next at pos 4 = 500
        # pos 4 -> next at pos 5 = 600
        # pos 5 -> next at pos 6 = 700
        assert labels == [400, 500, 600, 700]

    def test_labels_rank0_cp4(self, seq16_cp4: dict) -> None:
        batch = seq16_cp4["rank0"]
        labels = batch["labels"][0].tolist()
        # positions [0, 1, 14, 15]
        # pos 0 -> token at pos 1 = 1001
        # pos 1 -> token at pos 2 = 1002
        # pos 14 -> token at pos 15 = 1015
        # pos 15 -> last -> -100
        assert labels == [1001, 1002, 1015, -100]

    def test_labels_rank3_cp4(self, seq16_cp4: dict) -> None:
        batch = seq16_cp4["rank3"]
        labels = batch["labels"][0].tolist()
        # positions [6, 7, 8, 9]
        # pos 6 -> token at 7 = 1007
        # pos 7 -> token at 8 = 1008
        # pos 8 -> token at 9 = 1009
        # pos 9 -> token at 10 = 1010
        assert labels == [1007, 1008, 1009, 1010]

    def test_labels_all_ranks_exactly_one_neg100_cp2(self, seq8_cp2: dict) -> None:
        """Only the rank holding the last global position should have -100."""
        neg100_count = sum(
            (seq8_cp2[rk]["labels"] == -100).sum().item()
            for rk in seq8_cp2
        )
        assert neg100_count == 1  # only rank 0 has pos 7 (last)

    def test_labels_all_ranks_exactly_one_neg100_cp4(self, seq16_cp4: dict) -> None:
        neg100_count = sum(
            (seq16_cp4[rk]["labels"] == -100).sum().item()
            for rk in seq16_cp4
        )
        assert neg100_count == 1  # only rank 0 has pos 15 (last)

    # -- attention mask under CP --

    def test_attention_mask_is_causal_locally(self, seq8_cp2: dict) -> None:
        for rank_key in seq8_cp2:
            mask = seq8_cp2[rank_key]["attention_mask"][0, 0]
            local_len = mask.shape[0]
            expected = torch.tril(torch.ones(local_len, local_len, dtype=torch.bool))
            assert torch.equal(mask, expected)

    # -- batch dimension under CP --

    def test_batch_dim_broadcast_cp2(self) -> None:
        token_ids = list(range(100, 108))
        batch = prepare_batch(
            token_ids=token_ids, batch_size=3, cp_rank=0, cp_size=2, device="cpu"
        )
        assert batch["input_ids"].shape[0] == 3
        for i in range(3):
            assert batch["input_ids"][i].tolist() == batch["input_ids"][0].tolist()
            assert batch["labels"][i].tolist() == batch["labels"][0].tolist()

    # -- token_ids / position_ids consistency --

    def test_input_ids_match_global_at_positions(self, seq8_cp2: dict) -> None:
        """For each rank, input_ids[i] == global_input_ids[position_ids[i]]."""
        for rank_key in seq8_cp2:
            batch = seq8_cp2[rank_key]
            global_ids = batch["global_input_ids"][0]
            pos_ids = batch["position_ids"][0]
            input_ids = batch["input_ids"][0]
            gathered = global_ids[pos_ids]
            assert torch.equal(input_ids, gathered)

    def test_input_ids_match_global_at_positions_cp4(self, seq16_cp4: dict) -> None:
        for rank_key in seq16_cp4:
            batch = seq16_cp4[rank_key]
            global_ids = batch["global_input_ids"][0]
            pos_ids = batch["position_ids"][0]
            input_ids = batch["input_ids"][0]
            gathered = global_ids[pos_ids]
            assert torch.equal(input_ids, gathered)

    # -- positions are locally sorted (zigzag chunk1 < chunk2) --

    def test_positions_monotonic_within_each_chunk(self, seq8_cp2: dict) -> None:
        """Each half (chunk) of local positions should be monotonically increasing."""
        for rank_key in seq8_cp2:
            pos = seq8_cp2[rank_key]["position_ids"][0]
            half = len(pos) // 2
            chunk1 = pos[:half]
            chunk2 = pos[half:]
            assert all(chunk1[i] < chunk1[i + 1] for i in range(len(chunk1) - 1))
            assert all(chunk2[i] < chunk2[i + 1] for i in range(len(chunk2) - 1))


# ---------------------------------------------------------------------------
# TestPrepareBatch — zigzag with padding (seq_len not divisible by 2*cp_size)
# ---------------------------------------------------------------------------


class TestPrepareBatchZigzagPadding:
    """When seq_len is not evenly divisible by 2*cp_size, slice_with_cp pads."""

    def test_seq7_cp2_shapes(self) -> None:
        """seq_len=7, cp_size=2 -> chunk_size=ceil(7/4)=2, padded_len=8, local=4."""
        token_ids = list(range(10, 17))  # 7 tokens
        batch = prepare_batch(
            token_ids=token_ids, batch_size=1, cp_rank=0, cp_size=2, device="cpu"
        )
        assert batch["input_ids"].shape == (1, 4)
        assert batch["global_input_ids"].shape == (1, 7)

    def test_seq7_cp2_padded_token_is_zero(self) -> None:
        """The padding token (pad_value=0) should appear in the local slice."""
        token_ids = list(range(10, 17))  # 7 tokens
        batch_r0 = prepare_batch(
            token_ids=token_ids, batch_size=1, cp_rank=0, cp_size=2, device="cpu"
        )
        # rank 0: tokens[0:2] + tokens[6:8] where tokens[7]=0 (padded)
        local_ids = batch_r0["input_ids"][0].tolist()
        assert local_ids[0] == 10
        assert local_ids[1] == 11
        assert local_ids[2] == 16  # original token at index 6
        assert local_ids[3] == 0   # padded

    def test_seq7_cp2_all_ranks_cover_original_tokens(self) -> None:
        token_ids = list(range(10, 17))
        all_token_pos: list[tuple[int, int]] = []
        for rank in range(2):
            batch = prepare_batch(
                token_ids=token_ids, batch_size=1, cp_rank=rank, cp_size=2, device="cpu"
            )
            pos = batch["position_ids"][0].tolist()
            ids = batch["input_ids"][0].tolist()
            all_token_pos.extend(zip(pos, ids))

        original_positions = {p for p, _ in all_token_pos if p < 7}
        assert original_positions == set(range(7))

    def test_seq5_cp2_shapes(self) -> None:
        """seq_len=5, cp_size=2 -> chunk_size=ceil(5/4)=2, padded=8, local=4."""
        token_ids = list(range(5))
        batch = prepare_batch(
            token_ids=token_ids, batch_size=1, cp_rank=0, cp_size=2, device="cpu"
        )
        assert batch["input_ids"].shape == (1, 4)

    @pytest.mark.parametrize("seq_len", [3, 5, 7, 9, 11, 13])
    def test_odd_seq_len_cp2_no_crash(self, seq_len: int) -> None:
        token_ids = list(range(seq_len))
        for rank in range(2):
            batch = prepare_batch(
                token_ids=token_ids, batch_size=1, cp_rank=rank, cp_size=2, device="cpu"
            )
            assert batch["input_ids"].shape[0] == 1
            assert batch["input_ids"].shape[1] == batch["position_ids"].shape[1]
            assert batch["labels"].shape == batch["input_ids"].shape

    @pytest.mark.parametrize("seq_len", [5, 7, 10, 13, 15])
    def test_various_seq_len_cp4_no_crash(self, seq_len: int) -> None:
        token_ids = list(range(seq_len))
        for rank in range(4):
            batch = prepare_batch(
                token_ids=token_ids, batch_size=1, cp_rank=rank, cp_size=4, device="cpu"
            )
            assert batch["input_ids"].shape[0] == 1
            assert batch["labels"].shape == batch["input_ids"].shape


# ---------------------------------------------------------------------------
# TestPrepareBatch — label cross-validation across all ranks
# ---------------------------------------------------------------------------


class TestLabelsCrossRankConsistency:
    """Verify that labels across all CP ranks, when reassembled, form correct next-token targets."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_reassembled_labels_match_naive(self, cp_size: int) -> None:
        """Gather (position, label) from all ranks and verify against naive next-token."""
        seq_len = cp_size * 4  # evenly divisible
        token_ids = list(range(1000, 1000 + seq_len))

        position_to_label: dict[int, int] = {}
        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            positions = batch["position_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            for pos, label in zip(positions, labels):
                position_to_label[pos] = label

        for pos in range(seq_len - 1):
            assert position_to_label[pos] == token_ids[pos + 1], (
                f"At pos={pos}, expected label={token_ids[pos + 1]}, got {position_to_label[pos]}"
            )
        assert position_to_label[seq_len - 1] == -100

    @pytest.mark.parametrize("cp_size,seq_len", [(2, 7), (2, 11), (4, 9), (4, 15)])
    def test_reassembled_labels_match_naive_with_padding(
        self, cp_size: int, seq_len: int
    ) -> None:
        """Same cross-rank check but with non-divisible seq_len (padding involved)."""
        token_ids = list(range(1000, 1000 + seq_len))

        position_to_label: dict[int, int] = {}
        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            positions = batch["position_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            for pos, label in zip(positions, labels):
                if pos < seq_len:
                    position_to_label[pos] = label

        for pos in range(seq_len - 1):
            assert position_to_label.get(pos) == token_ids[pos + 1], (
                f"At pos={pos}, expected label={token_ids[pos + 1]}, got {position_to_label.get(pos)}"
            )


# ---------------------------------------------------------------------------
# TestPrepareBatch — slice_with_cp consistency
# ---------------------------------------------------------------------------


class TestSliceWithCPConsistency:
    """Verify that prepare_batch's CP slicing matches direct slice_with_cp calls."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_input_ids_match_direct_slice(self, cp_size: int) -> None:
        seq_len = cp_size * 4
        token_ids = list(range(1000, 1000 + seq_len))
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long)

        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            expected = _zigzag_slice(tokens_tensor, cp_rank=rank, cp_size=cp_size)
            assert torch.equal(batch["input_ids"][0], expected)

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_position_ids_match_direct_slice(self, cp_size: int) -> None:
        seq_len = cp_size * 4
        token_ids = list(range(seq_len))
        positions_tensor = torch.arange(seq_len, dtype=torch.long)

        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            expected = _zigzag_slice(positions_tensor, cp_rank=rank, cp_size=cp_size)
            assert torch.equal(batch["position_ids"][0], expected)
