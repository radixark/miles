# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron


def load_checkpoint():
    if _is_hf_checkpoint(TODO):
        return _load_checkpoint_hf()
    else:
        return _load_checkpoint_megatron()


def _is_hf_checkpoint():
    return TODO


def _load_checkpoint_hf():
    return TODO
