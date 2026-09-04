"""TinkerService over a FakeBackend: the protocol properties end to end —
idempotent resubmit, two-phase create, barrier ordering, lease reclamation."""

import asyncio

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, created_model, fb_payload, row

from miles.tinker.core.promise import DONE, FAILED
from miles.tinker.core.types import UserInputError


def _optim_payload(model_id: str, seq_id: int) -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}


async def test_resubmitted_seq_id_reuses_the_promise_and_runs_once(service):
    model_id = await created_model(service)
    payload = fb_payload(model_id, 1, [row()])

    first = service.submit("tenant", "forward_backward", payload)
    second = service.submit("tenant", "forward_backward", payload)
    assert first == second

    promise = await await_settled(service, "tenant", first)
    assert promise.state == DONE
    assert service.submit("tenant", "forward_backward", payload) == first, "a resubmit after completion also dedups"
    assert len(service.backend.named("forward_backward")) == 1, "the resubmit must not run the trainer again"


async def test_create_model_is_two_phase(service):
    request_id, model_id = service.create_model("tenant", {"base_model": "base", "lora_config": {"rank": 8}})

    promise = await await_settled(service, "tenant", request_id)
    assert promise.result == {"kind": "create_model", "model_id": model_id}
    assert service.backend.named("load_slot")[0]["rank"] == 8
    assert model_id in service.models


async def test_failed_slot_init_returns_the_slot(service):
    free_before = set(service.free_slots)
    service.backend.fail_next = RuntimeError("init blew up")
    request_id, model_id = service.create_model("tenant", {"base_model": "base"})

    promise = await await_settled(service, "tenant", request_id)
    assert (promise.state, promise.error_category) == (FAILED, "internal")
    assert model_id not in service.models
    assert service.free_slots == free_before


async def test_no_free_slots_is_a_user_error(service):
    for _ in range(service.config.n_slots):
        await created_model(service)
    with pytest.raises(UserInputError, match="no free adapter slots"):
        service.create_model("tenant", {"base_model": "base"})


async def test_the_wrong_base_model_is_rejected(service):
    with pytest.raises(UserInputError, match="serves"):
        service.create_model("tenant", {"base_model": "other"})


async def test_out_of_order_chunks_complete_and_the_barrier_waits(service):
    model_id = await created_model(service)

    optim = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    late = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [row()]))
    await asyncio.sleep(0.05)
    assert not service.backend.named("optim_step"), "the barrier must wait for the whole window"

    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    for request_id in (early, late, optim):
        assert (await await_settled(service, "tenant", request_id)).state == DONE

    kinds = [name for name, _ in service.backend.calls if name != "load_slot"]
    assert kinds.index("optim_step") > max(i for i, k in enumerate(kinds) if k == "forward_backward")


async def test_admission_failure_fails_the_promise_not_the_stream(service):
    model_id = await created_model(service)

    oversized = service.submit(
        "tenant", "forward_backward", fb_payload(model_id, 1, [row(service.config.max_tokens_per_datum + 1)])
    )
    assert (await await_settled(service, "tenant", oversized)).state == FAILED

    healthy = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [row()]))
    assert (await await_settled(service, "tenant", healthy)).state == DONE, "the stream must keep flowing"


async def test_forward_backward_outputs_align_to_datums(service):
    model_id = await created_model(service)
    request_id = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row(2), row(5)]))

    promise = await await_settled(service, "tenant", request_id)
    assert [len(output["logprobs"]) for output in promise.result["outputs"]] == [2, 5]


async def test_optim_step_returns_the_slot_grad_norm(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    request_id = service.submit("tenant", "optim_step", _optim_payload(model_id, 2))

    promise = await await_settled(service, "tenant", request_id)
    assert promise.result["metrics"] == {"grad_norm": 0.5 + slot}


async def test_save_then_load_roundtrip_paths(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    saved_path = (await await_settled(service, "tenant", save)).result["path"]
    assert saved_path == f"tinker://{model_id}/weights/ckpt"

    load = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": saved_path, "optimizer": False}
    )
    assert (await await_settled(service, "tenant", load)).state == DONE
    weights_only = service.backend.named("load_slot")[-1]
    assert weights_only["load_optimizer"] is False
    assert weights_only["ckpt_path"].endswith(f"{model_id}/weights/ckpt")


async def test_a_foreign_checkpoint_is_rejected_as_a_user_error(service):
    victim = await created_model(service, tenant="tenant-a")
    thief = await created_model(service, tenant="tenant-b")

    request_id = service.submit(
        "tenant-b",
        "load_state",
        {"model_id": thief, "seq_id": 1, "path": f"tinker://{victim}/weights/ckpt", "optimizer": True},
    )
    promise = await await_settled(service, "tenant-b", request_id)
    assert (promise.state, promise.error_category) == (FAILED, "user")


async def test_sampler_save_bumps_the_version_and_pushes(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id})
        promise = await await_settled(service, "tenant", request_id)
        assert promise.result["path"] == f"tinker://{model_id}/sampler_weights/{seq_id}"
    assert [push["lora_name"] for push in service.backend.named("push_slot")] == [f"{model_id}@1", f"{model_id}@2"]


async def test_sampling_resolves_against_the_pushed_version(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    sampler_path = (await await_settled(service, "tenant", save)).result["path"]

    request_id, sequence_ids = service.submit_sample(
        "tenant",
        {
            "model_path": sampler_path,
            "num_samples": 2,
            "prompt_tokens": [1, 2],
            "sampling_params": {"max_tokens": 4},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    assert len(sequence_ids) == 2
    promise = await await_settled(service, "tenant", request_id)
    assert len(promise.result["sequences"]) == 2
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@1"


async def test_lease_expiry_reclaims_the_tenant(service):
    session_id = service.create_session("tenant", {})
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    queued = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    await await_settled(service, "tenant", queued)
    stale = service.submit("tenant", "forward_backward", fb_payload(model_id, 3, [row()]))  # gapped: stays queued

    service.sessions[session_id]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()

    assert model_id not in service.models
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]
    assert service.retrieve("tenant", stale).state == FAILED


async def test_a_fresh_heartbeat_keeps_the_model(service):
    session_id = service.create_session("tenant", {})
    model_id = await created_model(service)

    service.heartbeat(session_id)
    await service._sweep_once()

    assert model_id in service.models
