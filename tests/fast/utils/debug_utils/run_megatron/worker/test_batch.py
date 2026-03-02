from __future__ import annotations

from typing import Any

import pytest
import torch

from miles.utils.debug_utils.run_megatron.worker.batch import (
    _build_labels,
    loss_func,
    prepare_batch,
)


@pytest.fixture()
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all torch factory calls from device="cuda" to device="cpu"."""
    for fn_name in ("tensor", "arange", "ones", "full"):
        orig = getattr(torch, fn_name)

        def _make_wrapper(f: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return f(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(torch, fn_name, _make_wrapper(orig))


class TestBuildLabels:
    def test_cp1_next_token_shift(self, cpu_device: None) -> None:
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        position_ids = torch.arange(3).unsqueeze(0)
        global_input_ids = input_ids.clone()

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=1,
        )
        assert labels[0, 0].item() == 20
        assert labels[0, 1].item() == 30
        assert labels[0, 2].item() == -100

    def test_cp1_batch_size_2(self, cpu_device: None) -> None:
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

    def test_cp_gt1_uses_position_ids(self, cpu_device: None) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[0, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0, 0].item() == 20

    def test_cp_gt1_last_pos_neg100(self, cpu_device: None) -> None:
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[0, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0, 1].item() == -100


class TestLossFunc:
    def test_return_type(self) -> None:
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, -100]], dtype=torch.long)
        result = loss_func(labels=labels, output_tensor=logits)
        assert isinstance(result, tuple)
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)

    def test_loss_finite(self) -> None:
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert torch.isfinite(loss)

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


class TestPrepareBatch:
    def test_shapes_cp1(self, cpu_device: None) -> None:
        token_ids = list(range(8))
        batch = prepare_batch(token_ids=token_ids, batch_size=2)
        assert batch["input_ids"].shape == (2, 8)
        assert batch["position_ids"].shape == (2, 8)
        assert batch["attention_mask"].shape == (2, 1, 8, 8)
        assert batch["labels"].shape == (2, 8)
        assert batch["global_input_ids"].shape == (2, 8)

    def test_input_ids_values(self, cpu_device: None) -> None:
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=1)
        assert batch["input_ids"][0].tolist() == [10, 20, 30]

    def test_position_ids_sequential(self, cpu_device: None) -> None:
        token_ids = list(range(5))
        batch = prepare_batch(token_ids=token_ids, batch_size=1)
        assert batch["position_ids"][0].tolist() == [0, 1, 2, 3, 4]

    def test_causal_mask(self, cpu_device: None) -> None:
        token_ids = list(range(4))
        batch = prepare_batch(token_ids=token_ids, batch_size=1)
        mask = batch["attention_mask"][0, 0]
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask, expected)

    def test_labels_next_token(self, cpu_device: None) -> None:
        token_ids = [10, 20, 30, 40]
        batch = prepare_batch(token_ids=token_ids, batch_size=1)
        assert batch["labels"][0].tolist() == [20, 30, 40, -100]

    def test_global_input_ids(self, cpu_device: None) -> None:
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=2)
        assert batch["global_input_ids"][0].tolist() == [10, 20, 30]
        assert batch["global_input_ids"][1].tolist() == [10, 20, 30]

    def test_batch_size_1(self, cpu_device: None) -> None:
        token_ids = list(range(4))
        batch = prepare_batch(token_ids=token_ids, batch_size=1)
        assert batch["input_ids"].shape[0] == 1
