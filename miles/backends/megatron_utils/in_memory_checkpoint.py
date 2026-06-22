import logging
from collections.abc import Sequence
from typing import Any

from megatron.training.global_vars import get_args

from miles.backends.megatron_utils.model import save

logger = logging.getLogger(__name__)


class InMemoryCheckpointManager:
    """ref: nvidia_resiliency_ext's LocalCheckpointManager."""

    def __init__(self) -> None:
        self.latest_iteration: int = -1
        self._state_dict: object = None
        self.local_ckpt_dir: str = "<in-memory>"

        _assert_args_for_in_memory_checkpoint(get_args())

    def save(self, state_dict: object, iteration: int, is_async: bool = False) -> None:
        """Store state_dict object reference in memory."""
        assert not is_async

        assert self._state_dict is None
        self._state_dict = state_dict
        self.latest_iteration = iteration

    def find_latest(self) -> int:
        return self.latest_iteration

    def load(self) -> tuple[object, str]:
        # Idempotent: Megatron's load_checkpoint calls _load_base_checkpoint twice
        # (once for format detection at line 1508, once for actual load at line 1712).
        # We must NOT consume `_state_dict` on first call.
        assert self.latest_iteration >= 0, "No in-memory checkpoint available"
        assert self._state_dict is not None
        return self._state_dict, f"in-memory-ckpt-iter-{self.latest_iteration}"


def save_to_memory(
    iteration: int,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
) -> object:
    """Save checkpoint to in-memory manager via model.save (with forward hook protection)."""
    manager = InMemoryCheckpointManager()
    save(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        checkpointing_context={"local_checkpoint_manager": manager},
        non_persistent_ckpt=True,
    )
    state_dict, _ = manager.load()
    return state_dict


def _assert_args_for_in_memory_checkpoint(args: Any) -> None:
    assert args.non_persistent_ckpt_type == "local", (
        f"Expected non_persistent_ckpt_type='local', " f"got {getattr(args, 'non_persistent_ckpt_type', None)!r}"
    )
    assert args.non_persistent_local_ckpt_algo is not None, "args.non_persistent_local_ckpt_algo must be set"
