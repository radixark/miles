from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import checkpoint


@pytest.mark.parametrize(
    "fp16,bf16,load_main", [(False, False, False), (True, False, False), (False, True, False), (True, False, True)]
)
def test_real_hf_load_refreshes_optimizer_masters_from_backend_precision(
    raw_mbridge_loader: Callable[[bool], list[set[str]]], fp16: bool, bf16: bool, load_main: bool
) -> None:
    """HF weight loading uses backend precision to refresh masters or reject checkpoint masters."""
    observed = raw_mbridge_loader(False)
    model = torch.nn.Linear(2, 2)
    calls = []
    optimizer = SimpleNamespace(reload_model_params=lambda: calls.append("reload"))
    args = SimpleNamespace(
        megatron_to_hf_mode="raw",
        offload_train=False,
        backend=SimpleNamespace(fp16=fp16, bf16=bf16, load_main_params_from_ckpt=load_main),
    )

    if load_main:
        with pytest.raises(AssertionError):
            checkpoint._load_checkpoint_hf([model], optimizer=optimizer, args=args, load_path="/hf")
    else:
        assert checkpoint._load_checkpoint_hf([model], optimizer=optimizer, args=args, load_path="/hf") == (0, 0)

    assert observed == [{"weight", "bias"}]
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))
    assert calls == (["reload"] if (fp16 or bf16) and not load_main else [])
