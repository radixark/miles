"""Backend selection and checkpoint writing for Megatron HF exports."""

import logging
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.utils import is_lora_model
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)


@cache
def _get_hf_bridge(hf_checkpoint: str):
    # Local: megatron.bridge is only needed on the bridge export path.
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


def save_hf_model(
    args,
    rollout_id: int,
    model: Sequence[DDP],
    *,
    publisher: SnapshotPublisher,
    path: str | Path | None = None,
    raise_on_error: bool = False,
) -> None:
    """Collectively write an HF model, with an additional HF adapter for LoRA.

    Writes a ``.complete`` marker after all ranks finish. Export errors are logged
    unless ``raise_on_error`` is set.
    """
    should_log = get_parallel_state().effective_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0
    path = Path(path if path is not None else args.save_hf.format(rollout_id=rollout_id))

    def write_weights(checkpoint_dir: Path):
        if args.megatron_to_hf_mode == "raw" and not is_lora_model(model):
            # LoRA needs Bridge to merge the adapter into the base weights
            publisher.write_model(
                checkpoint_dir,
                weights=dict(named_params_and_buffers(args, model, convert_to_global_name=True)),
                hf_checkpoint=args.hf_checkpoint,
            )
        else:
            bridge = _get_hf_bridge(args.hf_checkpoint)
            with patch_megatron_model(model):
                bridge.save_hf_pretrained(model, path=checkpoint_dir)
            torch.distributed.barrier(group=get_gloo_group())
            missing_weights = [False]
            if torch.distributed.get_rank() == 0:
                missing_weights[0] = not any(checkpoint_dir.glob("*.safetensors")) and not any(
                    checkpoint_dir.glob("*.bin")
                )
            torch.distributed.broadcast_object_list(missing_weights, src=0, group=get_gloo_group())
            if missing_weights[0]:
                raise RuntimeError(
                    f"HF export to {path} produced no weight files — the megatron "
                    f"bridge likely has no mapping for this model architecture."
                )
        if is_lora_model(model):
            publisher.write_adapter(None, checkpoint_dir / "adapter")

    if should_log:
        logger.info(f"Saving model in HuggingFace format to {path}")
    try:
        write_checkpoint_dir(path, write_weights, completion_marker=HF_EXPORT_COMPLETE_MARKER)
    except Exception as e:
        if raise_on_error:
            raise
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")
    else:
        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
