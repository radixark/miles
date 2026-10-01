"""Publish exported model weights to the Hugging Face Hub."""

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

logger = logging.getLogger(__name__)


def add_hub_arguments(parser: argparse.ArgumentParser) -> None:
    """Register optional Hub publishing for HF exports."""
    parser.add_argument("--push-to-hub", action="store_true", help="Publish HF exports to the Hub.")
    parser.add_argument("--hub-model-id", type=str, default=None, help="Hub model repository, e.g. username/model.")
    parser.add_argument(
        "--hub-private-repo",
        action="store_true",
        help="Create a private Hub repository (existing visibility is unchanged).",
    )
    parser.add_argument(
        "--hub-strategy",
        choices=["end", "every_save"],
        default="every_save",
        help="Publish only the final model, or after every save including the final save.",
    )


def validate_hub_args(args: argparse.Namespace) -> None:
    """Reject publishing configurations that cannot produce periodic HF exports."""
    if not args.push_to_hub:
        if args.hub_model_id or args.hub_private_repo or args.hub_strategy != "every_save":
            raise ValueError("Hub options require --push-to-hub")
        return
    if args.train_backend != "megatron":
        raise ValueError("--push-to-hub currently requires --train-backend megatron")
    if not args.hub_model_id:
        raise ValueError("--push-to-hub requires --hub-model-id")
    validate_repo_id(args.hub_model_id)
    if not args.save or not args.save_hf or args.save_interval is None or args.save_interval <= 0:
        raise ValueError("--push-to-hub requires --save, --save-hf, and a positive --save-interval")
    if args.dumper_enable:
        raise ValueError("--push-to-hub is not supported in dumper mode, which disables saving")
    if args.debug_rollout_only:
        raise ValueError("--push-to-hub is not supported with --debug-rollout-only")


def push_model_to_hub(
    *, checkpoint_dir: str, repo_id: str, private: bool, strategy: str, rollout_id: int, is_final: bool
) -> None:
    """Synchronously publish a completed export; upload failures leave training running.

    The caller must invoke this only on the actor's global rank zero. The model
    lives at the repository root; earlier versions remain accessible by commit.
    """
    if strategy == "end" and not is_final:
        return
    if strategy not in ("end", "every_save"):
        raise ValueError(f"Unsupported Hub strategy: {strategy}")
    # Export failures are logged rather than raised by the exporter. Its marker
    # ensures a partial export never replaces a previously published model.
    if not (Path(checkpoint_dir) / ".complete").is_file():
        logger.warning("Skipping incomplete HF export at %s", checkpoint_dir)
        return
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint_dir,
            commit_message=f"Upload Miles model at rollout {rollout_id}",
            ignore_patterns=[".complete"],
            # Remove obsolete shards in the same commit without deleting a
            # user-maintained model card or other repository documentation.
            delete_patterns=["*.safetensors", "pytorch_model*.bin", "*.index.json", "adapter/*"],
        )
    except Exception as error:
        # Publishing is optional. Avoid exception payloads that may include
        # authentication or HTTP request details.
        logger.warning(
            "Hub upload failed (%s); local export remains at %s. Retry with hf upload.",
            type(error).__name__,
            checkpoint_dir,
        )
        return
    logger.info("Published rollout %s to %s at commit %s", rollout_id, repo_id, commit.oid)
