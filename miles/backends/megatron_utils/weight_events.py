"""Hooks for model plugins that cache a derived form of their trainable parameters.

TOP's FP8 masters keep serving's processed FP8 bytes beside the BF16 masters. Miles notifies the
plugin whenever the masters may have changed (optimizer step, checkpoint load, weight restore) or
serving now holds the weight-sync quantizer's bytes (weight sync), instead of importing it. It also
tells the plugin when the train state is about to be offloaded and when a forward is about to read
restored masters (training starts, or another model's weights are switched in), so derived state can
be dropped before the offload and rebuilt before that forward allocates activations.
"""

from collections.abc import Callable

_ON_WEIGHTS_CHANGED: list[Callable[[], None]] = []
_ON_OFFLOAD: list[Callable[[], int]] = []
_ON_TRAIN_START: list[Callable[[], None]] = []
_INSTALLED_HF_CHECKPOINT: str | None = None
_HOLDS_INSTALLED_HF = False


def on_weights_changed(hook: Callable[[], None]) -> None:
    """Register ``hook``; registering the same hook again is a no-op."""
    if hook not in _ON_WEIGHTS_CHANGED:
        _ON_WEIGHTS_CHANGED.append(hook)


def weights_changed() -> None:
    for hook in _ON_WEIGHTS_CHANGED:
        hook()


def on_offload(hook: Callable[[], int]) -> None:
    """Register ``hook``, run before the train offload; it returns the bytes it freed."""
    if hook not in _ON_OFFLOAD:
        _ON_OFFLOAD.append(hook)


def offloading() -> int:
    return sum(hook() for hook in _ON_OFFLOAD)


def on_train_start(hook: Callable[[], None]) -> None:
    """Register ``hook``, run once the masters are restored and before the next forward reads them (training
    starts, or a model switch restores another model's weights)."""
    if hook not in _ON_TRAIN_START:
        _ON_TRAIN_START.append(hook)


def train_starting() -> None:
    for hook in _ON_TRAIN_START:
        hook()


def mark_hf_checkpoint_installed(path: str) -> None:
    """The model provider already loaded the parameters from the HF checkpoint at ``path``."""
    global _INSTALLED_HF_CHECKPOINT, _HOLDS_INSTALLED_HF
    _INSTALLED_HF_CHECKPOINT = path
    _HOLDS_INSTALLED_HF = True


def installed_hf_checkpoint() -> str | None:
    return _INSTALLED_HF_CHECKPOINT


def masters_left_installed_hf() -> None:
    """An optimizer step or a Megatron load replaced the installed HF checkpoint's weights in the model."""
    global _HOLDS_INSTALLED_HF
    _HOLDS_INSTALLED_HF = False


def holds_installed_hf_checkpoint() -> bool:
    """Whether the model still holds the weights the provider installed from the HF checkpoint."""
    return _INSTALLED_HF_CHECKPOINT is not None and _HOLDS_INSTALLED_HF
