"""Pin immutable HF-relative publications on the dedicated eval engines."""

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import _activate_engine, receiver_pool
from miles.utils.gpu_delta.publication import configured_skip_payload_hash

_UNKNOWN = object()


def uses_gpu_delta_eval(args):
    return args.update_weight_transfer_mode == "gpu-delta" and any(
        model.name == "eval" for model in args.sglang.models
    )


def write_eval_snapshot(updater, path, rollout_id, weight_version):
    """Publish READY, then mark the eval staging directory complete collectively."""
    from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
    from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

    write_checkpoint_dir(
        path,
        lambda directory: updater.export_gpu_delta_snapshot(directory, rollout_id, weight_version),
        completion_marker=HF_EXPORT_COMPLETE_MARKER,
    )


def _read_snapshot(checkpoint_dir, hf_checkpoint):
    directory = (Path(checkpoint_dir) / "gpu_delta").resolve()
    descriptor = json.loads((directory / "READY.json").read_bytes())
    manifest_path = Path(descriptor["manifest_path"]).resolve(strict=True)
    if manifest_path.name != "manifest.json":
        raise ValueError("Eval snapshot must reference a shared publication manifest")
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != descriptor["manifest_sha256"]:
        raise ValueError("Eval snapshot manifest SHA256 differs")
    manifest = json.loads(content)
    for key in ("stream_id", "publication_id", "base_version", "target_version", "plan_digest"):
        if manifest[key] != descriptor[key]:
            raise ValueError(f"Eval snapshot completion differs from manifest: {key}")
    if manifest["base_version"] != 0 or manifest["target_version"] != 1:
        raise ValueError("Eval snapshot must be immutable and relative to the HF base")
    # Both sides trust the same immutable HF prior; this checks configuration identity.
    if Path(manifest["base_checkpoint"]).resolve() != Path(hf_checkpoint).resolve():
        raise ValueError("Eval snapshot uses a different configured HF base")
    return descriptor


class EvalSnapshotUpdater:
    """Cancel each engine's last snapshot before applying its next one.

    The executor's existing eval lock excludes generation between these calls.
    A failed or ambiguous call requires receiver replacement, never XOR replay.
    READY descriptors may be reclaimed by the ordinary eval queue; their
    immutable publications remain in the sender's shared run storage.
    """

    def __init__(self, hf_checkpoint, run_uuid):
        self.hf_checkpoint = hf_checkpoint
        self.run_uuid = run_uuid
        self._engines = {}

    async def pin(self, clients, checkpoint_dir, weight_version, timeout):
        initial = Path(checkpoint_dir).resolve() == Path(self.hf_checkpoint).resolve()
        publication = None if initial else await asyncio.to_thread(_read_snapshot, checkpoint_dir, self.hf_checkpoint)
        descriptions = await asyncio.wait_for(
            asyncio.gather(
                *[
                    client.get_gpu_delta_info(engine_id=f"{self.run_uuid}:eval:{client.server_url}")
                    for client in clients
                ]
            ),
            timeout,
        )
        pool = receiver_pool(descriptions)
        results = await asyncio.gather(
            *[
                self._pin_engine(client, participants, description, publication, weight_version, timeout)
                for client, participants, description in zip(clients, pool.participants, descriptions, strict=True)
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise RuntimeError("GPU-delta eval pin failed; uncertain receivers require restart") from result

    async def _pin_engine(self, client, participants, description, publication, weight_version, timeout):
        engine_id = participants[0]["engine_id"]
        incarnation = tuple(sorted(identity["rank_id"] for identity in participants))
        previous_incarnation, previous = self._engines.get(engine_id, (None, None))
        if previous_incarnation != incarnation:
            if any(row["state"] != "IDLE" or row["version"] != 0 for row in description["participants"]):
                raise ValueError("Eval snapshot requires fresh receiver state after controller or engine replacement")
            previous = None
            self._engines[engine_id] = incarnation, None
        if previous is _UNKNOWN:
            raise RuntimeError("Eval receiver has an uncertain XOR result and must be replaced")
        if publication is None:
            if previous is not None:
                raise ValueError("Initial HF eval requires fresh receiver state")
            await asyncio.wait_for(client.update_weight_version(weight_version), timeout)
            return
        # No operation is replayable after dispatch: a lost reply could hide a
        # completed XOR. Healthy engines keep their independent progress.
        self._engines[engine_id] = incarnation, _UNKNOWN
        if previous is not None:
            await self._apply(client, previous, participants, "0", timeout)
        await self._apply(client, publication, participants, weight_version, timeout)
        self._engines[engine_id] = incarnation, publication

    async def _apply(self, client, publication, participants, weight_version, timeout):
        request = {
            key: publication[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "stream_id",
                "base_version",
                "target_version",
                "plan_digest",
            )
        }
        request.update(
            session_id=uuid.uuid4().hex,
            weight_version=weight_version,
            validate_payload_hash=not configured_skip_payload_hash(),
        )
        await _activate_engine(client, request, participants, timeout)
