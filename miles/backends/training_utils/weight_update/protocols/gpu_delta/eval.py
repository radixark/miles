"""Pin immutable HF-relative publications on the dedicated eval engines."""

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from miles.backends.training_utils.weight_update.protocols.gpu_delta.client import get_gpu_delta_info
from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import (
    activate_engine,
    committed_endpoint,
    engine_identities,
)


def uses_gpu_delta_eval(args):
    """Dedicated eval engines pin GPU-delta snapshots; trainer and controller configs lack --eval-num-gpus."""
    return args.update_weight_transfer_mode == "gpu-delta" and any(
        model.name == "eval" for model in args.sglang.models
    )


def write_eval_snapshot(updater, path, rollout_id, weight_version):
    """Publish READY, then mark the eval staging directory complete collectively."""
    # Trainer-only dependencies: the inference controller imports this module for eval pins.
    from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
    from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

    write_checkpoint_dir(
        path,
        lambda directory: updater.export_gpu_delta_snapshot(directory, rollout_id, weight_version),
        completion_marker=HF_EXPORT_COMPLETE_MARKER,
    )


def _eval_stream_dir(update_weight_disk_dir, run_uuid):
    return Path(update_weight_disk_dir) / f"eval-stream-{run_uuid}"


def record_eval_stream(update_weight_disk_dir, run_uuid, stream_id, predecessor=None):
    """Allow this launch and the exact checkpoint it restored as eval snapshot sources."""
    directory = _eval_stream_dir(update_weight_disk_dir, run_uuid)
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / "STREAM"
    content = json.dumps({"stream_id": stream_id, "predecessor": predecessor})
    if record.is_file() and record.read_text() == content:
        return None
    staged = directory / f".STREAM.{stream_id}"
    staged.write_text(content)
    staged.replace(record)
    return directory


def _read_snapshot(checkpoint_dir, hf_checkpoint, stream_record):
    directory = (Path(checkpoint_dir) / "gpu_delta").resolve()
    descriptor = json.loads((directory / "READY.json").read_bytes())
    manifest_path = Path(descriptor["manifest_path"]).resolve(strict=True)
    if manifest_path.name != "manifest.json":
        raise ValueError("Eval snapshot must reference a shared publication manifest")
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != descriptor["manifest_sha256"]:
        raise ValueError("Eval snapshot manifest SHA256 differs")
    manifest = json.loads(content)
    for key in (
        "stream_id",
        "base_stream_id",
        "publication_id",
        "base_version",
        "target_version",
        "training_step",
        "plan_digest",
    ):
        if manifest[key] != descriptor[key]:
            raise ValueError(f"Eval snapshot completion differs from manifest: {key}")
    if manifest["base_stream_id"] is not None or manifest["base_version"] != 0 or manifest["target_version"] < 1:
        raise ValueError("Eval snapshot must be immutable and relative to the HF base")
    # Both sides trust the same immutable HF prior; this checks configuration identity.
    if Path(manifest["base_checkpoint"]).resolve() != Path(hf_checkpoint).resolve():
        raise ValueError("Eval snapshot uses a different configured HF base")
    allowed = json.loads(stream_record.read_bytes())
    source = {key: manifest[key] for key in ("stream_id", "target_version", "training_step")}
    if source["stream_id"] != allowed["stream_id"] and source != allowed["predecessor"]:
        raise ValueError("Eval snapshot belongs to an unrelated trainer stream or checkpoint")
    return descriptor


class EvalSnapshotUpdater:
    """Cancel each engine's last snapshot before applying its next one.

    The executor's existing eval lock excludes generation between these calls.
    Every cancel or apply starts from the version reported by all original ranks.
    READY descriptors may be reclaimed by the ordinary eval queue; their
    immutable publications remain in the sender's shared run storage.
    """

    def __init__(self, args):
        self.hf_checkpoint = args.hf_checkpoint
        self.run_uuid = args.run_uuid
        self._stream_record = _eval_stream_dir(args.update_weight_disk_dir, args.run_uuid) / "STREAM"
        self._engines = {}

    async def pin(self, clients, checkpoint_dir, weight_version, timeout):
        initial = Path(checkpoint_dir).resolve() == Path(self.hf_checkpoint).resolve()
        publication = (
            None
            if initial
            else await asyncio.to_thread(_read_snapshot, checkpoint_dir, self.hf_checkpoint, self._stream_record)
        )
        descriptions = await asyncio.wait_for(
            asyncio.gather(
                *[
                    get_gpu_delta_info(client, engine_id=f"{self.run_uuid}:eval:{client.server_url}")
                    for client in clients
                ]
            ),
            timeout,
        )
        identities = engine_identities(descriptions)
        results = await asyncio.gather(
            *[
                self._pin_engine(client, participants, description, publication, weight_version, timeout)
                for client, participants, description in zip(
                    clients, identities.participants, descriptions, strict=True
                )
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise RuntimeError("GPU-delta eval pin failed; uncertain receivers require restart") from result

    async def _pin_engine(self, client, participants, description, publication, weight_version, timeout):
        engine_id = participants[0]["engine_id"]
        previous = self._engines.get(engine_id)
        plan_digest = (publication or previous or description["participants"][0])["plan_digest"]
        current = committed_endpoint(description, participants, plan_digest)
        destination = (publication["stream_id"], publication["target_version"]) if publication else (None, 0)
        if current == destination:
            await asyncio.wait_for(client.update_weight_version(weight_version), timeout)
        else:
            if current != (None, 0):
                if previous is None or current != (previous["stream_id"], previous["target_version"]):
                    raise ValueError("Eval receiver's committed weights have no known cancellation publication")
                await activate_engine(client, previous, participants, uuid.uuid4().hex, "0", timeout, reverse=True)
            if publication is not None:
                # Keep the immutable mask even if the reply is lost. The next pin
                # reads the actual endpoint before using it in either direction.
                self._engines[engine_id] = publication
                await activate_engine(client, publication, participants, uuid.uuid4().hex, weight_version, timeout)
        self._engines[engine_id] = publication
