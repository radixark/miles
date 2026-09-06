"""Checkpoint authorization must survive the training model's lease."""

import asyncio
import json
from pathlib import Path

import pytest
from tests.fast.tinker.harness import await_settled, created_model, make_service

from miles.tinker.core.promise import DONE, FAILED
from miles.tinker.core.types import UserInputError


@pytest.fixture
async def checkpoint_service(tmp_path):
    service = make_service(checkpoint_root=str(tmp_path))
    runner = asyncio.create_task(service.run())
    yield service
    runner.cancel()
    service._sweep_task.cancel()
    await asyncio.gather(runner, service._sweep_task, return_exceptions=True)


async def _save(service, model_id, tenant="tenant"):
    request_id = service.submit(tenant, "save_state", {"model_id": model_id, "seq_id": 1, "name": "saved"})
    return await await_settled(service, tenant, request_id)


async def _load(service, model_id, path, tenant="tenant", seq_id=1):
    request_id = service.submit(
        tenant, "load_state", {"model_id": model_id, "seq_id": seq_id, "path": path, "optimizer": True}
    )
    return await await_settled(service, tenant, request_id)


async def test_checkpoint_survives_source_lease_expiry(checkpoint_service):
    service = checkpoint_service
    source_id = await created_model(service)
    saved = await _save(service, source_id)
    session_id = service.create_session("tenant", {})
    service.sessions[session_id]["last_heartbeat"] = float("-inf")
    await service._sweep_once()
    assert source_id not in service.models

    target_id = await created_model(service)
    loaded = await _load(service, target_id, saved.result["path"])
    assert loaded.state == DONE
    assert service.backend.named("load_slot")[-1]["ckpt_path"].endswith(f"{source_id}/weights/saved")


async def test_checkpoint_owner_survives_gateway_restart(checkpoint_service):
    original = checkpoint_service
    source_id = await created_model(original)
    saved = await _save(original, source_id)
    restarted = make_service(checkpoint_root=original.config.checkpoint_root)
    runner = asyncio.create_task(restarted.run())
    try:
        target_id = await created_model(restarted)
        assert (await _load(restarted, target_id, saved.result["path"])).state == DONE
        foreign_id = await created_model(restarted, tenant="foreign")
        loaded = await _load(restarted, foreign_id, saved.result["path"], tenant="foreign")
        assert (loaded.state, loaded.error_category) == (FAILED, "user")
        assert len([call for call in restarted.backend.named("load_slot") if call["ckpt_path"]]) == 1
    finally:
        runner.cancel()
        restarted._sweep_task.cancel()
        await asyncio.gather(runner, restarted._sweep_task, return_exceptions=True)


async def test_metadata_does_not_store_the_api_key(checkpoint_service):
    service = checkpoint_service
    key = "tml-private-owner-credential"
    model_id = await created_model(service, tenant=key)
    assert (await _save(service, model_id, tenant=key)).state == DONE
    path = Path(service._checkpoint_dir(model_id, "weights", "saved")) / "tinker.json"
    text = path.read_text()
    assert key not in text
    assert json.loads(text)["base_model"] == "base"


async def test_failed_save_does_not_write_owner_metadata(checkpoint_service):
    service = checkpoint_service
    model_id = await created_model(service)
    service.backend.fail_next = RuntimeError("save failed")
    assert (await _save(service, model_id)).state == FAILED
    assert not list(Path(service.config.checkpoint_root).rglob("tinker.json"))


async def test_metadata_rejects_incompatible_target(checkpoint_service):
    service = checkpoint_service
    source_id = await created_model(service)
    saved = await _save(service, source_id)
    target_id = await created_model(service)
    service.models[target_id].lora_rank = 4
    loaded = await _load(service, target_id, saved.result["path"])
    assert (loaded.state, loaded.error_category) == (FAILED, "user")
    assert not [call for call in service.backend.named("load_slot") if call["ckpt_path"]]


async def test_legacy_checkpoint_requires_live_owner(checkpoint_service):
    service = checkpoint_service
    source_id = await created_model(service)
    path = f"tinker://{source_id}/weights/legacy"
    assert (await _load(service, source_id, path)).state == DONE
    del service.models[source_id]
    target_id = await created_model(service)
    loaded = await _load(service, target_id, path)
    assert (loaded.state, loaded.error_category) == (FAILED, "user")


@pytest.mark.parametrize("component", ["", ".", "..", "../escape", "nested/name", "nested\\name", "bad\x00name"])
@pytest.mark.parametrize("position", [0, 2])
def test_checkpoint_components_cannot_escape(tmp_path, component, position):
    service = make_service(checkpoint_root=str(tmp_path))
    parts = ["model-one", "weights", "saved"]
    parts[position] = component
    with pytest.raises(UserInputError):
        service._checkpoint_dir(*parts)


def test_checkpoint_symlink_cannot_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "model-one").symlink_to(outside, target_is_directory=True)
    service = make_service(checkpoint_root=str(root))
    with pytest.raises(UserInputError):
        service._checkpoint_dir("model-one", "weights", "saved")
