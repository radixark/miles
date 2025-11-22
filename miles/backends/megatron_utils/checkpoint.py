# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    fn = (
        _load_checkpoint_hf
        if _is_hf_checkpoint(TODO)
        else _load_checkpoint_megatron
    )
    return fn(
        ddp_model=ddp_model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        checkpointing_context=checkpointing_context,
        skip_load_to_model_and_opt=skip_load_to_model_and_opt,
    )


def _is_hf_checkpoint():
    return TODO


def _load_checkpoint_hf(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    return TODO
