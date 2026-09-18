from argparse import Namespace


def update_adaptive_kl(args: Namespace, loss_dict: dict[str, float]) -> None:
    """Rescale args.kl_loss_coef in place from the reduced optimizer-step metrics.

    Ziegler et al.: coef *= 1 + clip(kl / target - 1, -0.2, 0.2) * steps / horizon. The state is
    the coefficient itself; it is not stored in checkpoints, so a resumed run restarts from the
    --kl-loss-coef flag. Reports the coefficient for the next step as kl_loss_coef.
    """
    if getattr(args, "kl_ctrl", "fixed") != "adaptive" or "kl_loss" not in loss_dict:
        return

    kl = loss_dict["kl_loss"]
    proportional_error = max(-0.2, min(0.2, kl / getattr(args, "kl_target", 0.1) - 1.0))
    multiplier = 1.0 + proportional_error * getattr(args, "kl_ctrl_steps", 1) / getattr(args, "kl_horizon", 10000.0)
    args.kl_loss_coef = max(0.0, args.kl_loss_coef * multiplier)
    # Report the coefficient the next step will use, so the controller is visible in the train metrics.
    loss_dict["kl_loss_coef"] = args.kl_loss_coef
