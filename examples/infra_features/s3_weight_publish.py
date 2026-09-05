"""Disk-delta post-write hook that publishes each version to S3 via the H200 helper.

Object-store-backed rollout (a remote Trainium SGLang fleet) cannot share a POSIX
filesystem with the H200 trainer, so disk-delta writes each version to the local
``--update-weight-disk-dir`` and this hook publishes it to S3 before the engines
are asked to pull it. A receiver on the rollout host mirrors the version from S3
onto the engine's ``source_dir``.

Publication is delegated to the deployment's tested uploader (``h200-publish.sh``)
rather than reimplemented here: it uses AWS CLI v2 CRT transfers, validates the
manifest, refuses to overwrite an already-published version, and — critically —
uploads ``model.safetensors.index.json`` LAST so the index acts as the version's
completion marker. A naive recursive upload could publish the index before its
payload and let a receiver consume a partial version.

Wire it in with::

    --custom-update-weight-post-write-path examples.infra_features.s3_weight_publish.publish_version_to_s3

Configuration is read from the environment so nothing about the deployment is
hardcoded in the launcher::

    export MILES_H200_PUBLISH_HELPER=/path/on/h200/h200-publish.sh
    # optional: a different run prefix, honored by the helper
    export SGLANG_S3_PUBLISH_PREFIX=s3://.../qwen3-30b-a3b/run-YYYYMMDD/

The hook runs on every trainer rank; only rank 0 writes delta files, so it gates
itself to rank 0. The baseline sync republishes an empty dir (the base is seeded
from the engine's initial checkpoint via pull0), so an empty version is a no-op.
"""

import logging
import os
import subprocess

import torch.distributed as dist

logger = logging.getLogger(__name__)


def publish_version_to_s3(args, version_dir: str, rollout_engines) -> None:
    """Publish ``version_dir`` to S3 via ``$MILES_H200_PUBLISH_HELPER``.

    Signature matches the disk-delta post-write hook contract:
    ``hook(args, version_dir, rollout_engines) -> None``.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    if not any(os.path.isfile(os.path.join(version_dir, f)) for f in os.listdir(version_dir)):
        # The baseline sync clears and republishes an empty dir; nothing to upload.
        logger.info(f"disk-delta S3 publish: {os.path.basename(version_dir)} is empty, skipping")
        return

    helper = os.environ.get("MILES_H200_PUBLISH_HELPER")
    assert helper, (
        "publish_version_to_s3 requires MILES_H200_PUBLISH_HELPER to point at the deployment's "
        "h200-publish.sh (CRT uploader that publishes the index last)."
    )
    assert os.path.isfile(helper), f"MILES_H200_PUBLISH_HELPER={helper!r} is not a file on the trainer"

    logger.info(f"disk-delta S3 publish: uploading {os.path.basename(version_dir)} via {helper}")
    # The helper validates shards, refuses to overwrite, uploads payload then the
    # index last, and inherits the caller's AWS profile/SSO from the environment.
    subprocess.run(["bash", helper, version_dir], check=True)
    logger.info(f"disk-delta S3 publish: published {os.path.basename(version_dir)}")
