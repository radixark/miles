"""Keep the language model's packed-document kwargs out of the Qwen3.5 vision tower.

The FSDP actor passes HF's padding-free boundary kwargs (``cu_seq_lens_*``, ``max_length_*``,
``seq_idx``) so GatedDeltaNet resets its recurrence and conv state per packed document. HF's
multimodal forward also hands its ``**kwargs`` to the vision tower, whose flash-attention blocks set
``cu_seq_lens_q`` themselves, so any batch with images would fail with a duplicate keyword argument.
"""

import functools

import torch.nn as nn

from miles.backends.fsdp_utils.adaptations.packing import HF_PACKING_KWARG_NAMES


def keep_packing_kwargs_out_of_vision(model: nn.Module) -> None:
    """Drop the language model's packing kwargs at the vision tower's forward; a no-op for text-only
    checkpoints and for a tower already wrapped."""
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None or getattr(visual.forward, "_drops_lm_packing_kwargs", False):
        return
    vision_forward = visual.forward

    @functools.wraps(vision_forward)
    def forward(*args, **kwargs):
        return vision_forward(*args, **{k: v for k, v in kwargs.items() if k not in HF_PACKING_KWARG_NAMES})

    forward._drops_lm_packing_kwargs = True
    visual.forward = forward
