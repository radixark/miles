"""Fail before a Sokoban policy update if an auxiliary MTP head is present."""

import json
import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def before_train_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Check the constructed model, including heads omitted from CLI config."""
    for chunk in model:
        module = chunk
        while hasattr(module, "module"):
            module = module.module
        config = module.config
        assert config.mtp_num_layers is None, f"MTP layers present: {config.mtp_num_layers}"
        assert not getattr(module, "mtp_process", False), "MTP forward is active"
        assert getattr(module, "mtp", None) is None, "MTP module is present"
        assert not any("mtp" in name.lower() for name, _ in module.named_parameters()), "MTP weights are present"
    assert not args.enable_mtp_training, "MTP training must be disabled"
    if rollout_id == 0 and step_id == 0:
        assert args.start_rollout_id in (None, 0), "Expected a fresh rollout cursor"
        assert args.finetune and args.no_load_optim and args.no_load_rng, "Expected fresh training state"
        assert args.load == args.hf_checkpoint, "Expected initialization from original HF weights"
        logger.info(
            "SOKOBAN_MTP_DISABLED %s",
            json.dumps({"rank": dist.get_rank(), "mtp_layers": None, "mtp_parameters": 0, "fresh_hf": True}),
        )
