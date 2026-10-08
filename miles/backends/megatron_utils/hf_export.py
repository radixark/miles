"""Backend selection and checkpoint writing for Megatron HF exports."""

import json
import logging
import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import safetensors
import safetensors.torch
import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.utils import is_lora_model
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)

# Speculative-draft (MTP) tensors: Qwen3.5/3.6 and Qwen3-Next name them "mtp.", MiMo "model.mtp_layers.",
# and DeepSeek-V3/V4 and GLM append them to the decoder as layers past num_hidden_layers.
_DRAFT_TENSOR = re.compile(r"(^|\.)mtp(_layers)?\.")
_DECODER_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


@cache
def _get_hf_bridge(hf_checkpoint: str):
    # Local: megatron.bridge is only needed on the bridge export path.
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


def _weight_map(checkpoint: Path) -> dict[str, str] | None:
    """Each tensor of an HF safetensors checkpoint, mapped to the shard that holds it."""
    if (index := checkpoint / "model.safetensors.index.json").is_file():
        return json.loads(index.read_text())["weight_map"]
    if (single := checkpoint / "model.safetensors").is_file():
        with safetensors.safe_open(single, "pt") as shard:
            return dict.fromkeys(shard.keys(), single.name)
    return None


def _draft_tensors(source: Path, weight_map: dict[str, str]) -> dict[str, str]:
    """The source checkpoint's speculative-draft (MTP) tensors, each mapped to its shard."""
    config = json.loads((source / "config.json").read_text())
    num_layers = config.get("text_config", config).get("num_hidden_layers")

    def is_draft(name: str) -> bool:
        if _DRAFT_TENSOR.search(name):
            return True
        layer = _DECODER_LAYER.match(name)
        return layer is not None and num_layers is not None and int(layer[1]) >= num_layers

    return {name: shard for name, shard in weight_map.items() if is_draft(name)}


def _complete_draft(export_dir: Path, hf_checkpoint: str, *, trained: bool) -> None:
    """Make the export hold the speculative draft that its config.json, the source checkpoint's, declares.

    A trainer that trains the draft exports its updated weights, and missing ones are an error: a trained
    tensor is never filled in from the source. A trainer that does not train the draft builds no MTP
    layers, so the draft is the source checkpoint's, copied in.
    """
    source = Path(hf_checkpoint)
    source_map = _weight_map(source) if source.is_dir() else None
    draft = _draft_tensors(source, source_map) if source_map is not None else {}
    if not draft:
        return
    exported = _weight_map(export_dir)
    assert exported is not None, f"HF export to {export_dir} has no safetensors weights to hold the speculative draft"
    missing = {name: shard for name, shard in draft.items() if name not in exported}
    if trained:
        assert not missing, f"the trainer trains the speculative draft, but the export lacks {sorted(missing)}"
        return
    if not missing:
        return

    frozen: dict[str, list[str]] = {}
    for name, shard in missing.items():
        frozen.setdefault(shard, []).append(name)
    index_path = export_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    total_size = index.get("metadata", {}).get("total_size")
    if total_size is None:
        total_size = sum((export_dir / shard).stat().st_size for shard in set(exported.values()))
    for i, (shard, names) in enumerate(sorted(frozen.items()), start=1):
        shard_name = f"model-frozen-draft-{i:05d}.safetensors"
        with safetensors.safe_open(source / shard, "pt") as source_shard:
            tensors = {name: source_shard.get_tensor(name) for name in names}
        safetensors.torch.save_file(tensors, export_dir / shard_name)
        exported.update(dict.fromkeys(names, shard_name))
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    index_path.write_text(json.dumps({"metadata": {"total_size": total_size}, "weight_map": exported}, indent=2))
    logger.info(f"Copied the source checkpoint's {len(missing)} speculative-draft tensors into {export_dir}")


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
        # After every collective: write_checkpoint_dir reports a failure here to all ranks and marks nothing.
        if torch.distributed.get_rank() == 0:
            _complete_draft(checkpoint_dir, args.hf_checkpoint, trained=bool(args.mtp_num_layers))

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
