from __future__ import annotations

import torch

from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch


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


class TestPrepareBatchCP1:
    def test_all_keys_present(self) -> None:
        batch = prepare_batch(token_ids=list(range(8)), batch_size=1, device="cpu")
        expected_keys = {"input_ids", "position_ids", "attention_mask", "labels"}
        assert set(batch.keys()) == expected_keys

    def test_all_tensors_long_except_mask(self) -> None:
        batch = prepare_batch(token_ids=list(range(8)), batch_size=1, device="cpu")
        assert batch["input_ids"].dtype == torch.long
        assert batch["position_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["attention_mask"] is None

    def test_shapes(self) -> None:
        token_ids = list(range(8))
        batch = prepare_batch(token_ids=token_ids, batch_size=2, device="cpu")
        assert batch["input_ids"].shape == (2, 8)
        assert batch["position_ids"].shape == (2, 8)
        assert batch["attention_mask"] is None
        assert batch["labels"].shape == (2, 8)

    def test_input_ids_values(self) -> None:
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["input_ids"][0].tolist() == [10, 20, 30]

    def test_position_ids_sequential(self) -> None:
        token_ids = list(range(5))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["position_ids"][0].tolist() == [0, 1, 2, 3, 4]

    def test_attention_mask_is_none(self) -> None:
        token_ids = list(range(4))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["attention_mask"] is None

    def test_labels_next_token(self) -> None:
        token_ids = [10, 20, 30, 40]
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["labels"][0].tolist() == [20, 30, 40, -100]

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
        assert batch["attention_mask"] is None
