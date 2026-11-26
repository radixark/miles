"""One reward pool worker: loads its pool's scorer on the GPU its bundle names, or on the CPU, and scores batches."""

import torch

from miles.platforms import current_platform
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import _to_local_gpu_id


class RewardWorker:
    def __init__(self, *, args, pool: RewardPoolConfig, gpu_ids: list[int]) -> None:
        configure_logger(args, source=SimpleProcessIdentity(component="reward_worker"))
        if gpu_ids:
            # gpu_ids are the bundle's physical GPU ids; an inherited visible-device mask renumbers them in this process.
            device = torch.device(current_platform.device_type, _to_local_gpu_id(gpu_ids[0]))
            torch.accelerator.set_device_index(device.index)
        else:
            device = torch.device("cpu")
        self._scorer = load_function(pool.scorer_path)(device=device, **pool.scorer_kwargs)

    def score_batch(self, outputs: list, prompts: list[str]) -> list[float | None]:
        return self._scorer.score_batch(outputs, prompts)
