# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.global_vars import get_args
from transformers import AutoConfig


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    load_path = get_args().load

    if _is_hf_checkpoint(load_path):
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            load_path=load_path,
        )
    else:
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )


def _is_hf_checkpoint(path: str):
    try:
        AutoConfig.from_pretrained(path)
        return True
    except (ValueError, OSError):
        return False


def _load_checkpoint_hf(ddp_model, load_path: str):
    TODO

    iteration = TODO
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far
