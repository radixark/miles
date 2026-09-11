"""The service preserves request ordering, tenant isolation, and failure recovery."""

import asyncio
from contextlib import suppress

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, created_model, datum, fb_payload, make_service, rl_datum

from miles.tinker.core.future import DONE, FAILED
from miles.tinker.core.types import OwnershipError, UserInputError


def _optim_payload(model_id: str, seq_id: int) -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}


async def test_resubmitted_seq_id_reuses_the_future_and_runs_once(service):
    model_id = await created_model(service)
    payload = fb_payload(model_id, 1, [datum()])

    first = service.submit("tenant", "forward_backward", payload)
    second = service.submit("tenant", "forward_backward", payload)
    assert first == second

    future = await await_settled(service, "tenant", first)
    assert future.state == DONE
    assert service.submit("tenant", "forward_backward", payload) == first, "a resubmit after completion also dedups"
    assert len(service.backend.named("forward_backward")) == 1, "the resubmit must not run the trainer again"


async def test_create_model_is_two_phase(service):
    request_id, model_id = service.create_model("tenant", {"base_model": "base", "lora_config": {"rank": 8}})

    future = await await_settled(service, "tenant", request_id)
    assert future.result == {"op": "create_model", "model_id": model_id}
    assert service.backend.named("load_slot")[0]["rank"] == 8
    assert model_id in service.models


async def test_failed_slot_init_returns_the_slot(service):
    free_before = set(service.free_slots)
    service.backend.fail_next = RuntimeError("init blew up")
    request_id, model_id = service.create_model("tenant", {"base_model": "base"})

    future = await await_settled(service, "tenant", request_id)
    assert (future.state, future.error_category) == (FAILED, "server")
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
    late = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    await asyncio.sleep(0.05)
    assert not service.backend.named("optim_step"), "the barrier must wait for every batch op ahead"

    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    for request_id in (early, late, optim):
        assert (await await_settled(service, "tenant", request_id)).state == DONE

    calls = [name for name, _ in service.backend.calls if name != "load_slot"]
    assert calls.index("optim_step") > max(i for i, name in enumerate(calls) if name == "forward_backward")


async def test_admission_failure_fails_the_future_not_the_stream(service):
    model_id = await created_model(service)

    oversized = service.submit(
        "tenant", "forward_backward", fb_payload(model_id, 1, [datum(service.config.max_tokens_per_datum + 1)])
    )
    assert (await await_settled(service, "tenant", oversized)).state == FAILED

    healthy = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    assert (await await_settled(service, "tenant", healthy)).state == DONE, "the stream must keep flowing"


async def test_forward_backward_outputs_align_to_datums(service):
    model_id = await created_model(service)
    request_id = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum(2), datum(5)]))

    future = await await_settled(service, "tenant", request_id)
    assert [len(output["logprobs"]) for output in future.result["outputs"]] == [2, 5]


async def test_optim_step_returns_the_slot_grad_norm(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    request_id = service.submit("tenant", "optim_step", _optim_payload(model_id, 2))

    future = await await_settled(service, "tenant", request_id)
    assert future.result["metrics"] == {"grad_norm": 0.5 + slot}


async def test_save_then_load_roundtrip_paths(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt", "overwrite": False}
    )
    saved_path = (await await_settled(service, "tenant", save)).result["path"]
    assert saved_path == f"tinker://{model_id}/weights/ckpt"

    load = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": saved_path, "optimizer": False}
    )
    assert (await await_settled(service, "tenant", load)).state == DONE
    weights_only = service.backend.named("load_slot")[-1]
    assert weights_only["load_optimizer"] is False
    assert weights_only["ckpt_path"].endswith(f"{model_id}/weights/ckpt")


async def test_sampler_save_bumps_the_version_and_pushes(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id})
        future = await await_settled(service, "tenant", request_id)
        assert future.result["path"] == f"tinker://{model_id}/sampler_weights/{seq_id}"
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
    future = await await_settled(service, "tenant", request_id)
    assert len(future.result["sequences"]) == 2
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@1"


async def test_sampler_requests_carry_the_published_checkpoint_path(service):
    """Push and sample requests must carry the export path for engine backfill."""
    model_id = await created_model(service)
    request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", request_id)).result["path"]

    disk_dir = service._checkpoint_dir(model_id, "sampler_weights", "1")
    assert service.backend.named("export_slot")[0]["path"] == disk_dir
    assert service.backend.named("push_slot")[0]["lora_path"] == disk_dir

    sample_id, _ = service.submit_sample(
        "tenant",
        {
            "model_path": path,
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    await await_settled(service, "tenant", sample_id)
    assert service.backend.named("sample")[0]["lora_path"] == disk_dir


async def test_warm_push_failure_still_publishes_the_version(service):
    """A failed warm push must not invalidate an exported sampler version."""
    model_id = await created_model(service)
    service.backend.fail_on["push_slot"] = RuntimeError("engine down")
    request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    future = await await_settled(service, "tenant", request_id)
    assert future.result["path"] == f"tinker://{model_id}/sampler_weights/1"

    sample_id, _ = service.submit_sample(
        "tenant",
        {
            "model_path": future.result["path"],
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    future = await await_settled(service, "tenant", sample_id)
    assert future.result["sequences"]


async def test_failed_export_burns_the_version_number(service):
    """A failed export must leave its version unpublished and never reuse its number."""
    model_id = await created_model(service)
    service.backend.fail_on["export_slot"] = RuntimeError("disk full")
    failed = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    future = await await_settled(service, "tenant", failed)
    assert (future.state, future.error_category) == (FAILED, "server")

    retried = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2})
    future = await await_settled(service, "tenant", retried)
    assert future.result["path"] == f"tinker://{model_id}/sampler_weights/2"

    with pytest.raises(UserInputError):
        service.submit_sample(
            "tenant",
            {
                "model_path": f"tinker://{model_id}/sampler_weights/1",
                "num_samples": 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )


async def test_lease_expiry_reclaims_the_tenant(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    queued = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    await await_settled(service, "tenant", queued)
    stale = service.submit("tenant", "forward_backward", fb_payload(model_id, 3, [datum()]))  # gapped: stays queued

    service.sessions[session_id]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()

    assert model_id not in service.models
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]
    assert service.retrieve_future("tenant", stale).state == FAILED


async def test_a_fresh_heartbeat_keeps_the_model(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service)

    service.heartbeat("tenant", session_id)
    await service._sweep_once()

    assert model_id in service.models


@pytest.mark.parametrize("name", ["../evil", "a/b", "..", ".hidden"], ids=["parent", "slash", "dotdot", "hidden"])
async def test_traversal_checkpoint_names_are_rejected(service, name):
    model_id = await created_model(service)
    request_id = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": name, "overwrite": False}
    )
    future = await await_settled(service, "tenant", request_id)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "invalid checkpoint path segment" in future.error
    assert not service.backend.named("save_slot"), "nothing may touch the disk for a rejected name"


async def test_traversal_sampler_paths_are_rejected(service):
    model_id = await created_model(service)
    with pytest.raises(UserInputError):
        service.submit_sample(
            "tenant",
            {
                "model_path": f"tinker://{model_id}/weights/../../etc",
                "num_samples": 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )


async def test_expired_result_on_resubmit_fails_instead_of_410(service, monkeypatch):
    from miles.tinker.core import future as future_module

    model_id = await created_model(service)
    first = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    await await_settled(service, "tenant", first)
    monkeypatch.setattr(future_module, "_FINISHED_TTL_S", -1.0)
    assert service.retrieve_future("tenant", first) is None, "the result must have aged out"

    resubmitted = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert resubmitted != first
    monkeypatch.setattr(future_module, "_FINISHED_TTL_S", 3600.0)  # only the first result aged out
    future = service.retrieve_future("tenant", resubmitted)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "expired" in future.error
    assert len(service.backend.named("forward_backward")) == 1, "an executed command must never re-run"


async def test_save_state_refuses_to_overwrite_unless_asked(service):
    model_id = await created_model(service)
    payload = {"model_id": model_id, "name": "ckpt", "overwrite": False}
    first = service.submit("tenant", "save_state", payload | {"seq_id": 1})
    assert (await await_settled(service, "tenant", first)).state == DONE

    clobber = service.submit("tenant", "save_state", payload | {"seq_id": 2})
    future = await await_settled(service, "tenant", clobber)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "overwrite" in future.error

    replace = service.submit("tenant", "save_state", payload | {"seq_id": 3, "overwrite": True})
    assert (await await_settled(service, "tenant", replace)).state == DONE


async def test_checkpoints_outlive_the_lease(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "kept", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]

    service.sessions[session_id]["last_heartbeat"] = -1e9
    await service._sweep_once()
    assert model_id not in service.models, "the lease sweep must reclaim the model"

    service.create_session("tenant")
    fresh_id = await created_model(service)
    load = service.submit("tenant", "load_state", {"model_id": fresh_id, "seq_id": 1, "path": path, "optimizer": True})
    assert (await await_settled(service, "tenant", load)).state == DONE


async def test_a_foreign_tenants_checkpoint_does_not_load(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "mine", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]

    thief_model = await created_model(service, tenant="thief")
    load = service.submit(
        "thief", "load_state", {"model_id": thief_model, "seq_id": 1, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "thief", load)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "belong" in future.error


async def test_a_failed_batch_discards_the_whole_batch_run(service):
    model_id = await created_model(service)
    first = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum(), datum()]))
    second = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    service.backend.fail_next = RuntimeError("cuda died")
    assert (await await_settled(service, "tenant", first)).state == FAILED
    assert (
        await await_settled(service, "tenant", second)
    ).state == FAILED, "the batch run shares one gradient accumulation; a sibling's failure poisons it"
    assert service.backend.named("zero_grads"), "the poisoned accumulation must be dropped"


async def test_merged_optim_settles_each_slot_on_its_own(service):
    model_a = await created_model(service)
    model_b = await created_model(service)
    slot_b = service.models[model_b].slot
    service.backend.optim_outcomes[slot_b] = {"error": "boom"}

    ok = service.submit("tenant", "optim_step", _optim_payload(model_a, 1))
    bad = service.submit("tenant", "optim_step", _optim_payload(model_b, 1))
    resolved = await await_settled(service, "tenant", ok)
    failed = await await_settled(service, "tenant", bad)
    assert resolved.state == DONE and "grad_norm" in resolved.result["metrics"]
    assert (failed.state, failed.error_category) == (FAILED, "server") and "boom" in failed.error


async def test_a_nonfinite_step_reports_the_skip(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.optim_outcomes[slot] = {"skipped_nonfinite": 1.0}
    request_id = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    future = await await_settled(service, "tenant", request_id)
    assert future.state == DONE
    assert future.result["metrics"] == {"skipped_nonfinite": 1.0}


async def test_num_samples_is_capped(service):
    model_id = await created_model(service)
    version = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", version)).result["path"]
    with pytest.raises(UserInputError):
        service.submit_sample(
            "tenant",
            {
                "model_path": path,
                "num_samples": service.config.max_samples_per_request + 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )


async def test_malformed_loss_inputs_are_rejected_at_admission(service):
    model_id = await created_model(service)
    cases = {
        "unknown loss_fn": {"loss_fn": "made_up", "datums": [datum()]},
        "missing input": {"loss_fn": "importance_sampling", "datums": [datum()]},
        "wrong length": {
            "loss_fn": "importance_sampling",
            "datums": [datum() | {"sampling_logprobs": [0.0], "advantages": [1.0, 2.0, 3.0, 4.0]}],
        },
        # datum() carries weights, which importance_sampling never reads
        "unread input": {
            "loss_fn": "importance_sampling",
            "datums": [datum() | {"sampling_logprobs": [0.0] * 3, "advantages": [1.0] * 3}],
        },
    }
    for seq_id, payload in enumerate(cases.values(), start=1):
        request_id = service.submit(
            "tenant",
            "forward_backward",
            {"model_id": model_id, "seq_id": seq_id, "loss_fn_config": {}, **payload},
        )
        future = await await_settled(service, "tenant", request_id)
        assert (future.state, future.error_category) == (FAILED, "user")
    assert not service.backend.named("forward_backward"), "rejected datums must never reach the trainer"

    healthy = service.submit(
        "tenant",
        "forward_backward",
        {
            "model_id": model_id,
            "seq_id": 5,
            "loss_fn": "importance_sampling",
            "loss_fn_config": {},
            "datums": [rl_datum(3)],
        },
    )
    assert (await await_settled(service, "tenant", healthy)).state == DONE


async def test_discarded_gradients_fail_the_next_optim_step(service):
    model_id = await created_model(service)
    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", early)).state == DONE, "the early batch resolves before the failure"

    service.backend.fail_next = RuntimeError("cuda died")
    late = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    assert (await await_settled(service, "tenant", late)).state == FAILED

    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    future = await await_settled(service, "tenant", step)
    assert (
        future.state == FAILED and "discarded" in future.error
    ), "the early batch's gradients were discarded after the failed batch; stepping would be a silent no-op"
    assert not service.backend.named("optim_step")

    retry = service.submit("tenant", "forward_backward", fb_payload(model_id, 4, [datum()]))
    assert (await await_settled(service, "tenant", retry)).state == DONE
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 5))
    assert (await await_settled(service, "tenant", step)).state == DONE, "a fresh accumulation steps normally"


async def test_unsupported_lora_configs_are_rejected(service):
    for lora_config in ({"rank": 8, "seed": 7}, {"rank": 8, "train_unembed": True}, {"rank": 8, "train_mlp": False}):
        with pytest.raises(UserInputError):
            service.create_model("tenant", {"base_model": service.config.base_model, "lora_config": lora_config})
    _, model_id = service.create_model(
        "tenant",
        {
            "base_model": service.config.base_model,
            "lora_config": {"rank": 8, "train_attn": True, "train_mlp": True, "train_unembed": False},
        },
    )
    assert model_id in service.models


async def test_sampler_paths_resolve_after_the_lease_died(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", save)).result["path"]

    service.sessions[session_id]["last_heartbeat"] = -1e9
    await service._sweep_once()
    assert model_id not in service.models

    service.create_session("tenant")
    lora_name, lora_path = service._resolve_sampler("tenant", path)
    assert lora_name == f"{model_id}@1" and lora_path.endswith("/sampler_weights/1")
    with pytest.raises((UserInputError, OwnershipError)):
        service._resolve_sampler("thief", path)


async def test_an_unnamed_sampler_save_returns_a_sampling_session(service):
    model_id = await created_model(service)
    unnamed = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "sampler_path": None}
    )
    result = (await await_settled(service, "tenant", unnamed)).result
    session = service.sampling_sessions[result["sampling_session_id"]]
    assert (session["tenant"], session["model_path"]) == ("tenant", result["path"])

    named = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2, "sampler_path": "ckpt"}
    )
    assert "sampling_session_id" not in (await await_settled(service, "tenant", named)).result


async def test_weights_info_reads_the_checkpoint_not_the_lease(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    del service.models[model_id]

    info = service.weights_info("tenant", path)
    assert info == {
        "base_model": "base",
        "is_lora": True,
        "lora_rank": 8,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }
    with pytest.raises(OwnershipError):
        service.weights_info("thief", path)


async def test_a_failed_optim_step_retires_the_model(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.optim_outcomes[slot] = {"error": "allreduce died"}

    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    future = await await_settled(service, "tenant", step)
    assert future.state == FAILED and "allreduce died" in future.error
    assert model_id not in service.models, "a half-applied step may have diverged the slot across ranks"
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]

    with pytest.raises(UserInputError, match="restore from a checkpoint"):
        service.submit("tenant", "optim_step", _optim_payload(model_id, 2))


async def test_poison_consumption_discards_retried_gradients(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.fail_next = RuntimeError("cuda died")
    failed = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", failed)).state == FAILED

    retry = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    assert (await await_settled(service, "tenant", retry)).state == DONE

    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    assert (await await_settled(service, "tenant", step)).state == FAILED
    assert (
        service.backend.named("zero_grads") == [{"slot": slot}] * 2
    ), "the retried batch accumulated on top of the discard; its gradients must go too"
    assert not service.backend.named("optim_step")

    resubmit = service.submit("tenant", "forward_backward", fb_payload(model_id, 4, [datum()]))
    assert (await await_settled(service, "tenant", resubmit)).state == DONE
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 5))
    assert (await await_settled(service, "tenant", step)).state == DONE


async def test_a_backend_level_optim_failure_retires_every_model_in_the_barrier(service):
    model_a = await created_model(service)
    model_b = await created_model(service)
    service.backend.fail_on["optim_step"] = RuntimeError("allgather died")

    first = service.submit("tenant", "optim_step", _optim_payload(model_a, 1))
    second = service.submit("tenant", "optim_step", _optim_payload(model_b, 1))
    for request_id in (first, second):
        future = await await_settled(service, "tenant", request_id)
        assert future.state == FAILED and "restore from a checkpoint" in future.error
    assert model_a not in service.models and model_b not in service.models
    assert len(service.backend.named("unload_slot")) == 2

    fresh = await created_model(service)  # the dispatch loop survived the failure
    assert fresh in service.models


async def test_a_failed_unload_keeps_the_slot_out_of_the_free_pool(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.fail_on["unload_slot"] = RuntimeError("engine gone")

    service.sessions[session_id]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()

    assert model_id not in service.models
    assert slot not in service.free_slots, "a slot whose unload failed is dirty and must not be reused"
    await service._sweep_once()  # the sweep itself survived


async def test_a_failed_load_state_retires_the_model(service):
    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", saved)).result["path"]

    service.backend.fail_on["load_slot"] = RuntimeError("shard corrupt")
    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "tenant", loaded)
    assert (future.state, future.error_category) == (FAILED, "server")
    assert model_id not in service.models, "a load that failed partway may have left mixed state"


async def test_a_named_sampler_save_uses_the_name_and_rejects_reuse(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "sampler_path": "v1"}
    )
    result = (await await_settled(service, "tenant", save)).result
    assert result["path"] == f"tinker://{model_id}/sampler_weights/v1"
    assert "sampling_session_id" not in result

    request_id, _ = service.submit_sample(
        "tenant",
        {
            "model_path": result["path"],
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    await await_settled(service, "tenant", request_id)
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@v1"

    reuse = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2, "sampler_path": "v1"}
    )
    future = await await_settled(service, "tenant", reuse)
    assert future.state == FAILED and "already exist" in future.error, "saved versions are immutable"


async def test_checkpoint_meta_stores_a_digest_not_the_credential(service):
    import json
    import os

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    await await_settled(service, "tenant", saved)
    meta_path = os.path.join(service._checkpoint_dir(model_id, "weights", "ck"), "META.json")
    meta = json.loads(open(meta_path).read())
    assert "tenant" not in meta and meta["tenant_digest"] != "tenant", "the bearer credential must not be persisted"
    info = service.weights_info("tenant", f"tinker://{model_id}/weights/ck")
    assert (info["train_attn"], info["train_mlp"], info["train_unembed"]) == (True, True, False)


async def test_a_checkpoint_saved_under_other_settings_does_not_load(service):
    import json
    import os

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", saved)).result["path"]
    meta_path = os.path.join(service._checkpoint_dir(model_id, "weights", "ck"), "META.json")
    meta = json.loads(open(meta_path).read())
    meta["lora_alpha"] = meta["lora_alpha"] + 1  # the same tensors would be scaled differently
    open(meta_path, "w").write(json.dumps(meta))

    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "tenant", loaded)
    assert (future.state, future.error_category) == (FAILED, "user") and "lora_alpha" in future.error
    assert not service.backend.named("load_slot")[1:], "nothing may touch the slot on a mismatch"


async def test_a_recycled_slot_does_not_inherit_poison(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.fail_next = RuntimeError("cuda died")
    failed = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", failed)).state == FAILED

    service.sessions[session_id]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()
    assert slot in service.free_slots

    del service.sessions[session_id]  # no live sessions: no lease to expire
    fresh = await created_model(service)
    assert service.models[fresh].slot == slot
    step_after_fb = service.submit("tenant", "forward_backward", fb_payload(fresh, 1, [datum()]))
    assert (await await_settled(service, "tenant", step_after_fb)).state == DONE
    step = service.submit("tenant", "optim_step", _optim_payload(fresh, 2))
    future = await await_settled(service, "tenant", step)
    assert future.state == DONE, "the poison belonged to the evicted model, not the slot"


async def test_a_unit_escaping_its_handler_retires_the_model_and_keeps_serving(service, monkeypatch):
    model_id = await created_model(service)

    async def broken_handler(unit):
        raise RuntimeError("handler bug")

    monkeypatch.setattr(service, "_run_barrier", broken_handler)
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    future = await await_settled(service, "tenant", step)
    assert future.state == FAILED and "unhandled failure" in future.error
    assert model_id not in service.models, "slots a broken handler touched are unknown state"

    monkeypatch.undo()
    fresh = await created_model(service)
    fb = service.submit("tenant", "forward_backward", fb_payload(fresh, 1, [datum()]))
    assert (await await_settled(service, "tenant", fb)).state == DONE, "the dispatch loop must survive"


async def test_a_dead_trainer_escapes_the_dispatch_loop(tmp_path, monkeypatch):
    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    try:
        model_id = await created_model(gateway)
        monkeypatch.setattr(gateway.backend, "trainer_dead", lambda: True)
        gateway.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
        with pytest.raises(RuntimeError, match="trainer workers died"):
            await asyncio.wait_for(run_task, timeout=2)
    finally:
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
