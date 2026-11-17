"""Immutable GPU-delta eval snapshots use the ordinary asynchronous eval queue."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.backends.training_utils.weight_update.protocols.gpu_delta.eval import (
    EvalSnapshotUpdater,
    record_eval_stream,
)
from miles.ray.rollout.eval_fleet import EVAL_WEIGHT_LOAD_TIMEOUT_SECS, InferenceControllerEvalFleet


def write_snapshot(path, hf_checkpoint, **overrides):
    directory = path.parent / "shared-publications" / path.name
    directory.mkdir(parents=True)
    descriptor_dir = path / "gpu_delta"
    descriptor_dir.mkdir(parents=True)
    manifest = {
        "stream_id": "job-stream",
        "publication_id": path.name,
        "base_version": 0,
        "target_version": 1,
        "plan_digest": "plan",
        "base_checkpoint": str(hf_checkpoint),
        "payload_checksum_format": "sha256",
    } | overrides
    content = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(content)
    descriptor = manifest | {
        "manifest_path": str(directory / "manifest.json"),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
    }
    (descriptor_dir / "READY.json").write_text(json.dumps(descriptor))
    (path / ".complete").touch()
    return descriptor


def make_client(engine_id, version=0):
    identity = {"engine_id": engine_id, "rank_id": f"{engine_id}-rank"}

    async def apply(**request):
        receipt = {
            key: request[key]
            for key in ("session_id", "stream_id", "manifest_sha256", "target_version", "plan_digest")
        } | {"identity": identity, "state": "RESUMED"}
        return {
            "success": True,
            "participants": [receipt],
            "applied_participants": [receipt | {"state": "APPLIED"}],
            "timings": {},
        }

    client = SimpleNamespace(
        server_url=f"http://{engine_id}",
        get_gpu_delta_info=AsyncMock(
            return_value={
                "success": True,
                "participants": [{"identity": identity, "state": "IDLE", "version": version}],
            }
        ),
        update_weight_version=AsyncMock(),
        update_weights_from_gpu_delta=AsyncMock(side_effect=apply),
    )

    async def make_request(endpoint, payload):
        # The GPU-delta controls post through the client's request helper; each route records its payload.
        return await getattr(client, endpoint)(**payload)

    client._make_request = make_request
    return client


def make_updater(tmp_path, hf_checkpoint, stream_id="job-stream"):
    """An eval updater whose run's trainer recorded ``stream_id``, the stream ``write_snapshot`` publishes by default."""
    disk_dir = tmp_path / "publication-root"
    record_eval_stream(disk_dir, "job", stream_id)
    return EvalSnapshotUpdater(
        SimpleNamespace(hf_checkpoint=hf_checkpoint, run_uuid="job", update_weight_disk_dir=disk_dir)
    )


async def test_nonconsecutive_eval_points_pin_independent_base_relative_snapshots(tmp_path):
    base = tmp_path / "base"
    clients = [make_client("e0"), make_client("e1")]
    updater = make_updater(tmp_path, base)
    previous_staging = None
    for step in (3, 9):
        if previous_staging is not None:
            shutil.rmtree(previous_staging)
        path = tmp_path / f"step_{step}"
        descriptor = write_snapshot(path, base)
        await updater.pin(clients, path, str(step), 5)
        for client in clients:
            request = client.update_weights_from_gpu_delta.call_args.kwargs
            assert (request["base_version"], request["target_version"]) == (0, 1)
            assert request["weight_version"] == str(step)
            assert request["manifest_sha256"] == descriptor["manifest_sha256"]
            assert request["validate_payload_hash"] is True
            assert request["participants"] == tuple(
                row["identity"] for row in client.get_gpu_delta_info.return_value["participants"]
            )
        previous_staging = path
    for client in clients:
        calls = [call.kwargs for call in client.update_weights_from_gpu_delta.await_args_list]
        assert [call["weight_version"] for call in calls] == ["3", "0", "9"]
        # Pins keep the receiver's defaults, retract and flush, so eval reuses no KV computed with older weights.
        assert not any("pause_mode" in call for call in calls)
        assert calls[0]["manifest_path"] == calls[1]["manifest_path"]
        assert Path(calls[1]["manifest_path"]).exists()
        assert len({call["session_id"] for call in calls}) == 3


@pytest.mark.parametrize("invalid", ["hash", "base", "prior"])
async def test_invalid_snapshot_never_reaches_xor_apply(tmp_path, invalid):
    base, path = tmp_path / "base", tmp_path / "snapshot"
    overrides = {"base_version": 2} if invalid == "base" else {}
    descriptor = write_snapshot(path, tmp_path / "other" if invalid == "prior" else base, **overrides)
    if invalid == "hash":
        with Path(descriptor["manifest_path"]).open("ab") as out:
            out.write(b" ")
    client = make_client("e0")
    with pytest.raises(ValueError):
        await make_updater(tmp_path, base).pin([client], path, "5", 5)
    client.update_weights_from_gpu_delta.assert_not_awaited()


async def test_resumed_job_never_pins_an_earlier_trainers_stream(tmp_path):
    base = tmp_path / "base"
    # The resume-time point reads the previous launch's --save-hf snapshot; this run's trainer recorded a new stream.
    write_snapshot(tmp_path / "hf_9", base, stream_id="earlier-stream")
    write_snapshot(tmp_path / "hf_10", base)
    client = make_client("e0")
    updater = make_updater(tmp_path, base)

    with pytest.raises(ValueError, match="earlier trainer stream"):
        await updater.pin([client], tmp_path / "hf_9", "9", 5)
    # It is skipped before any engine is contacted, so the fresh engine still takes this run's stream.
    client.get_gpu_delta_info.assert_not_awaited()
    client.update_weights_from_gpu_delta.assert_not_awaited()
    await updater.pin([client], tmp_path / "hf_10", "10", 5)
    request = client.update_weights_from_gpu_delta.call_args.kwargs
    assert (request["stream_id"], request["weight_version"]) == ("job-stream", "10")


async def test_failed_pin_settles_other_engines_without_replaying_xor(tmp_path):
    base, path = tmp_path / "base", tmp_path / "snapshot"
    write_snapshot(path, base)
    failed, healthy = make_client("failed"), make_client("healthy")
    updater = make_updater(tmp_path, base)
    finished = asyncio.Event()

    async def fail(**request):
        raise TimeoutError("applied, but reply lost")

    healthy_apply = healthy.update_weights_from_gpu_delta.side_effect

    async def finish(**request):
        await asyncio.sleep(0)
        finished.set()
        return await healthy_apply(**request)

    failed.update_weights_from_gpu_delta.side_effect = fail
    healthy.update_weights_from_gpu_delta.side_effect = finish
    with pytest.raises(RuntimeError):
        await updater.pin([failed, healthy], path, "5", 5)
    assert finished.is_set()
    failed.update_weights_from_gpu_delta.assert_awaited_once()
    healthy.update_weights_from_gpu_delta.assert_awaited_once()
    # A later eval may advance healthy engines, but never retries the uncertain
    # incarnation. A replacement rank starts directly from the immutable base.
    with pytest.raises(RuntimeError) as uncertain:
        await updater.pin([failed, healthy], path, "9", 5)
    assert "uncertain XOR result" in str(uncertain.value.__cause__)
    failed.update_weights_from_gpu_delta.assert_awaited_once()
    assert healthy.update_weights_from_gpu_delta.await_count == 3
    replacement = make_client("failed")
    replacement.get_gpu_delta_info.return_value["participants"][0]["identity"]["rank_id"] = "new-rank"
    await updater.pin([replacement, healthy], path, "12", 5)
    replacement.update_weights_from_gpu_delta.assert_awaited_once()


@pytest.mark.parametrize("version", [0, 1])
async def test_initial_hf_eval_only_labels_fresh_base_engines(tmp_path, version):
    client = make_client("e0", version)
    # The initial HF eval reads no snapshot, so it needs no recorded stream.
    updater = EvalSnapshotUpdater(
        SimpleNamespace(hf_checkpoint=tmp_path, run_uuid="job", update_weight_disk_dir=tmp_path / "no-record")
    )
    if version == 0:
        await updater.pin([client], tmp_path, "0", 5)
        client.update_weight_version.assert_awaited_once_with("0")
    else:
        with pytest.raises(RuntimeError, match="require restart"):
            await updater.pin([client], tmp_path, "0", 5)
        client.update_weight_version.assert_not_awaited()
    client.update_weights_from_gpu_delta.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["success", "lost_reply", "stale_version"])
async def test_dedicated_eval_pins_gpu_delta_once_and_verifies_serving_version(outcome):
    client = SimpleNamespace(
        get_weight_version=AsyncMock(return_value="5" if outcome == "success" else "4"),
        update_weights_from_disk=AsyncMock(),
    )
    controller = InferenceControllerEvalFleet(
        SimpleNamespace(
            update_weight_transfer_mode="gpu-delta",
            sglang=SimpleNamespace(models=[SimpleNamespace(name="eval")]),
            hf_checkpoint="/base",
            run_uuid="job",
            update_weight_disk_dir="/publications",
        ),
        srv=SimpleNamespace(context_lock=asyncio.Lock(), api_clients=[client]),
    )
    controller._gpu_delta.pin = AsyncMock(
        side_effect=TimeoutError("lost activation reply") if outcome == "lost_reply" else None
    )

    assert await controller._pin_fleet("/snap/step_5", "5") is (outcome == "success")

    controller._gpu_delta.pin.assert_awaited_once_with([client], "/snap/step_5", "5", EVAL_WEIGHT_LOAD_TIMEOUT_SECS)
    client.update_weights_from_disk.assert_not_awaited()
    if outcome == "lost_reply":
        client.get_weight_version.assert_not_awaited()
    else:
        client.get_weight_version.assert_awaited_once()
