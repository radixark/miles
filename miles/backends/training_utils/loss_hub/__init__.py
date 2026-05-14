from argparse import Namespace

from miles.backends.training_utils.loss_hub.base_types import LossFunction
from miles.backends.training_utils.loss_hub.losses import policy_loss_function, sft_loss_function, value_loss_function
from miles.utils.misc import load_function


def get_loss_function(args: Namespace) -> LossFunction:
    match args.loss_type:
        case "policy_loss":
            return policy_loss_function
        case "value_loss":
            return value_loss_function
        case "sft_loss":
            return sft_loss_function
        case "custom_loss":
            return load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
