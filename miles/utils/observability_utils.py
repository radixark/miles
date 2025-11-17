from miles.utils.wandb_utils import init_wandb_primary, init_wandb_secondary


def init_observability(args, primary: bool = True, **kwargs):
    if primary:
        init_wandb_primary(args, **kwargs)
    else:
        init_wandb_secondary(args, **kwargs)
