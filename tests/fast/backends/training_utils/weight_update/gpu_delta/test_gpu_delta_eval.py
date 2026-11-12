"""Immutable GPU-delta eval snapshots use the ordinary asynchronous eval queue."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.backends.training_utils.weight_update.protocols.gpu_delta.eval import EvalSnapshotUpdater
from miles.ray.rollout.eval_dispatch import EvalDispatcher
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

    return SimpleNamespace(
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


async def test_nonconsecutive_eval_points_pin_independent_base_relative_snapshots(tmp_path):
    base = tmp_path / "base"
    clients = [make_client("e0"), make_client("e1")]
    updater = EvalSnapshotUpdater(base, "job")
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
            assert request["participants"] == tuple(
                row["identity"] for row in client.get_gpu_delta_info.return_value["participants"]
            )
        previous_staging = path
    for client in clients:
        calls = [call.kwargs for call in client.update_weights_from_gpu_delta.await_args_list]
        assert [call["weight_version"] for call in calls] == ["3", "0", "9"]
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
        await EvalSnapshotUpdater(base, "job").pin([client], path, "5", 5)
    client.update_weights_from_gpu_delta.assert_not_awaited()


async def test_failed_pin_settles_other_engines_without_replaying_xor(tmp_path):
    base, path = tmp_path / "base", tmp_path / "snapshot"
    write_snapshot(path, base)
    failed, healthy = make_client("failed"), make_client("healthy")
    updater = EvalSnapshotUpdater(base, "job")
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
    with pytest.raises(RuntimeError):
        await updater.pin([failed, healthy], path, "9", 5)
    failed.update_weights_from_gpu_delta.assert_awaited_once()
    assert healthy.update_weights_from_gpu_delta.await_count == 3
    replacement = make_client("failed")
    replacement.get_gpu_delta_info.return_value["participants"][0]["identity"]["rank_id"] = "new-rank"
    await updater.pin([replacement, healthy], path, "12", 5)
    replacement.update_weights_from_gpu_delta.assert_awaited_once()


@pytest.mark.parametrize("version", [0, 1])
async def test_initial_hf_eval_only_labels_fresh_base_engines(tmp_path, version):
    client = make_client("e0", version)
    updater = EvalSnapshotUpdater(tmp_path, "job")
    if version == 0:
        await updater.pin([client], tmp_path, "0", 5)
        client.update_weight_version.assert_awaited_once_with("0")
    else:
        with pytest.raises(RuntimeError, match="require restart"):
            await updater.pin([client], tmp_path, "0", 5)
        client.update_weight_version.assert_not_awaited()
    client.update_weights_from_gpu_delta.assert_not_awaited()


async def test_gpu_delta_eval_exports_coherent_snapshot_then_queues_generation(tmp_path):
    args = SimpleNamespace(
        update_weight_transfer_mode="gpu-delta",
        eval_uses_snapshots=True,
        eval_hf_dir=str(tmp_path),
        eval_max_in_flight=1,
        eval_overflow_policy="skip",
        eval_keep_snapshots=0,
        offload_train=False,
    )
    actor = SimpleNamespace(export_eval_snapshot=AsyncMock(), update_weights=AsyncMock())
    entered, release = asyncio.Event(), asyncio.Event()

    async def evaluate(*args, **kwargs):
        entered.set()
        await release.wait()

    executor = SimpleNamespace(eval=AsyncMock(side_effect=evaluate), report_eval_skip=AsyncMock())
    dispatcher = EvalDispatcher(args, actor, executor)
    try:
        await dispatcher.dispatch(3)
        await asyncio.wait_for(entered.wait(), 5)
        actor.export_eval_snapshot.assert_awaited_once_with(3, str(tmp_path / "step_3"))
        actor.update_weights.assert_not_awaited()
        await dispatcher.dispatch(9)
        executor.report_eval_skip.assert_awaited_once_with(9, "busy")
        assert actor.export_eval_snapshot.await_count == 1
    finally:
        release.set()
        await dispatcher.drain()


@pytest.mark.parametrize("outcome", ["success", "lost_reply", "stale_version"])
async def test_eval_fleet_pins_gpu_delta_once_and_verifies_serving_version(outcome):
    client = SimpleNamespace(
        get_weight_version=AsyncMock(return_value="5" if outcome == "success" else "4"),
        update_weights_from_disk=AsyncMock(),
    )
    fleet = InferenceControllerEvalFleet(
        SimpleNamespace(update_weight_transfer_mode="gpu-delta", hf_checkpoint="/base", run_uuid="job"),
        srv=SimpleNamespace(context_lock=asyncio.Lock(), api_clients=[client]),
    )
    fleet._gpu_delta.pin = AsyncMock(
        side_effect=TimeoutError("lost activation reply") if outcome == "lost_reply" else None
    )

    assert await fleet._pin_fleet("/snap/step_5", "5") is (outcome == "success")

    fleet._gpu_delta.pin.assert_awaited_once_with([client], "/snap/step_5", "5", EVAL_WEIGHT_LOAD_TIMEOUT_SECS)
    client.update_weights_from_disk.assert_not_awaited()
    if outcome == "lost_reply":
        client.get_weight_version.assert_not_awaited()
    else:
        client.get_weight_version.assert_awaited_once()
