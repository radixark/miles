from unittest.mock import MagicMock, patch

import torch

from miles.backends.training_utils import offload

_MODULE = "miles.backends.training_utils.offload"


def _optimizer_with_state(device):
    param = torch.nn.Parameter(torch.zeros(2, device=device))
    optimizer = torch.optim.AdamW([param])
    optimizer.state[param]["exp_avg"] = torch.zeros(2, device=device)
    return optimizer


def test_offload_and_reload_move_modules_and_optimizer_state_together():
    module = MagicMock()
    optimizer = _optimizer_with_state("cpu")
    with (
        patch(f"{_MODULE}.dist"),
        patch(f"{_MODULE}.get_gloo_group"),
        patch(f"{_MODULE}.clear_memory") as clear,
        patch(f"{_MODULE}.print_memory"),
        patch(f"{_MODULE}.move_optimizer_state") as move_state,
    ):
        offload.offload_to_host([module], [optimizer])
        module.cpu.assert_called_once()
        move_state.assert_called_with([optimizer], "cpu")
        clear.assert_called_once()

        offload.reload_to_device([module], [optimizer])
        module.cuda.assert_called_once()
        move_state.assert_called_with([optimizer], "cuda")
