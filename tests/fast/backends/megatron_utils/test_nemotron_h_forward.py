import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def install_shim(monkeypatch: pytest.MonkeyPatch, model_class: type) -> None:
    stub = ModuleType("megatron.core.models.mamba")
    stub.MambaModel = model_class
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    path = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_forward_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_mamba_model_loss_mask_shim()
    module._install_mamba_model_loss_mask_shim()


@pytest.mark.parametrize("mtp_kwargs", [None, {}, {"mtp_labels": None}])
def test_forward_without_mtp_keyword(monkeypatch: pytest.MonkeyPatch, mtp_kwargs: dict[str, None] | None) -> None:
    class Model:
        def forward(self, tokens: torch.Tensor, *, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return tokens, loss_mask

    install_shim(monkeypatch, Model)
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[0, 1, 1]])
    result = Model().forward(tokens, loss_mask=mask, mtp_kwargs=mtp_kwargs)
    assert result[0] is tokens
    assert result[1] is mask


def test_forward_shifts_mtp_labels_once_without_mutating_input(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def forward(self, *, loss_mask: torch.Tensor, mtp_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return loss_mask, mtp_labels

    install_shim(monkeypatch, Model)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(labels)
    mtp_kwargs = {"mtp_labels": labels}
    result = Model().forward(loss_mask=mask, mtp_kwargs=mtp_kwargs)
    assert result[0] is mask
    torch.testing.assert_close(result[1], torch.tensor([[2, 3, 1], [5, 6, 4]]))
    torch.testing.assert_close(labels, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert mtp_kwargs["mtp_labels"] is labels
