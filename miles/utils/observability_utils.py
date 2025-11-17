from miles.utils.wandb_utils import init_wandb_secondary


def init_observability(args, primary: bool = True):
    init_wandb_secondary(args)
