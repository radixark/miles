"""The service preserves request ordering, tenant isolation, and failure isolation."""

import asyncio
import hashlib
import json
import os
import re
import time
from contextlib import suppress
from pathlib import Path

import pyrage
import pytest
from tests.fast.tinker.harness import (
    ADAM,
    await_settled,
    created_model,
    datum,
    fb_payload,
    make_service,
    model_payload,
    write_legacy_checkpoint,
)

from miles.backends.training_utils.checkpoint.adapter.commit import resolve_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.tinker.core.checkpoint_expiry import CheckpointExpiry
from miles.tinker.core.future import DONE, FAILED
from miles.tinker.core.types import NotFoundError, OwnershipError, UserInputError
from miles.tinker.core.utils import resolve_checkpoint_dir, resolve_sampler_checkpoint


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
    request_id, model_id = service.create_model("tenant", model_payload(service, lora_config={"rank": 8}))

    future = await await_settled(service, "tenant", request_id)
    assert future.result == {"op": "create_model", "model_id": model_id}
    assert service.backend.named("load_slot")[0]["rank"] == 8
    assert model_id in service.models


async def test_failed_slot_init_returns_the_slot(service):
    free_before = set(service.free_slots)
    service.backend.fail_next = {"error": "init blew up"}
    request_id, model_id = service.create_model("tenant", model_payload(service))

    future = await await_settled(service, "tenant", request_id)
    assert (future.state, future.error_category) == (FAILED, "server")
    assert model_id not in service.models
    assert service.free_slots == free_before


async def test_no_free_slots_is_a_user_error(service):
    for _ in range(service.config.n_slots):
        await created_model(service)
    with pytest.raises(UserInputError, match="no free adapter slots"):
        service.create_model("tenant", model_payload(service))


async def test_the_wrong_base_model_is_rejected(service):
    with pytest.raises(UserInputError, match="serves"):
        service.create_model("tenant", model_payload(service, base_model="other"))


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


async def test_admission_failure_terminates_the_training_model_queue(service):
    model_id = await created_model(service)

    oversized = service.submit(
        "tenant", "forward_backward", fb_payload(model_id, 1, [datum(service.config.max_tokens_per_datum + 1)])
    )
    assert (await await_settled(service, "tenant", oversized)).state == FAILED

    with pytest.raises(UserInputError, match="create a new model"):
        service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))


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
    checkpoint = Path(service.config.checkpoint_root) / model_id / "weights" / "ckpt"
    # the trainer reads the commit the name pointed at when the load was scheduled
    assert weights_only["ckpt_path"] == str(checkpoint.resolve()) and checkpoint.is_symlink()


async def test_sampler_save_publishes_successive_versions(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id})
        future = await await_settled(service, "tenant", request_id)
        assert future.result["path"] == f"tinker://{model_id}/sampler_weights/{seq_id}"
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1")).parent
    assert [(export["root"], export["adapter_name"]) for export in service.backend.named("export_slot")] == [
        (str(root), "1"),
        (str(root), "2"),
    ]
    assert all(resolve_adapter_dir(root, name, tenant="tenant") is not None for name in ("1", "2")), "both committed"


async def test_sampling_resolves_against_the_saved_version(service):
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
    assert service.backend.named("sample")[0]["lora_name"].startswith(f"{model_id}@1@")


@pytest.mark.parametrize("requested,expected", [(8192, 7630), (32, 32), (0, 0)])
async def test_sampling_respects_the_remaining_context_without_mutating_the_request(service, requested, expected):
    service.config.max_tokens_per_sample = 65536
    payload = {
        "num_samples": 1,
        "prompt_tokens": [1] * 57906,
        "sampling_params": {"max_tokens": requested, "temperature": 0.7},
    }
    request_id, _ = service.submit_sample("tenant", payload)
    assert (await await_settled(service, "tenant", request_id)).state == DONE
    sent = service.backend.named("sample")[0]["payload"]
    assert sent["sampling_params"] == {"max_tokens": expected, "temperature": 0.7}
    assert sent["prompt_tokens"] == payload["prompt_tokens"]
    assert payload["sampling_params"]["max_tokens"] == requested


@pytest.mark.parametrize("prompt_length", [65536, 65537])
async def test_an_exhausted_context_never_reaches_the_engine(service, prompt_length):
    service.config.max_tokens_per_sample = 65536
    with pytest.raises(UserInputError, match="prompt is too long"):
        service.submit_sample(
            "tenant",
            {"num_samples": 1, "prompt_tokens": [1] * prompt_length, "sampling_params": {"max_tokens": 1}},
        )
    assert service.backend.named("sample") == []


async def test_a_sample_without_max_tokens_never_reaches_the_engine(service):
    """The SDK defaults max_tokens to None; core is the one place that rejects it."""
    with pytest.raises(UserInputError, match="max_tokens must be a nonnegative integer"):
        service.submit_sample(
            "tenant", {"num_samples": 1, "prompt_tokens": [1], "sampling_params": {"max_tokens": None}}
        )
    assert service.backend.named("sample") == []


@pytest.mark.parametrize("scoring", [{"prompt_logprobs": True}, {"topk_prompt_logprobs": 2}])
async def test_a_sequence_sampled_to_the_window_can_be_scored(service, scoring):
    service.config.max_tokens_per_sample = 8
    payload = {"num_samples": 1, "prompt_tokens": [1] * 8, "sampling_params": {"max_tokens": 1}} | scoring
    request_id, _ = service.submit_sample("tenant", payload)
    assert (await await_settled(service, "tenant", request_id)).state == DONE
    assert service.backend.named("sample")[0]["payload"]["sampling_params"] == {"max_tokens": 0}


async def test_a_short_inference_window_does_not_shrink_the_training_window(service):
    service.config.max_tokens_per_sample = 2
    service.config.max_tokens_per_datum = 8
    with pytest.raises(UserInputError, match="prompt is too long"):
        service.submit_sample(
            "tenant", {"num_samples": 1, "prompt_tokens": [1, 2], "sampling_params": {"max_tokens": 1}}
        )
    model_id = await created_model(service)
    request_id = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum(7)]))
    assert (await await_settled(service, "tenant", request_id)).state == DONE


async def test_sampler_requests_carry_the_published_checkpoint_path(service):
    model_id = await created_model(service)
    request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", request_id)).result["path"]

    link = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1"))
    committed = resolve_adapter_dir(link.parent, "1", tenant="tenant")

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
    sample = service.backend.named("sample")[0]
    # the engine gets the physical commit, under a name no later commit of "1" can reuse
    assert sample["lora_path"] == str(committed.path)
    assert sample["lora_name"] == f"{model_id}@1@{committed.commit_id}"


async def test_sample_failure_preserves_the_snapshot_and_training_model_queue(service):
    model_id = await created_model(service)
    request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    future = await await_settled(service, "tenant", request_id)
    assert future.result["path"] == f"tinker://{model_id}/sampler_weights/1"

    service.backend.fail_on["sample"] = {"error": "engine down"}
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
    assert future.state == FAILED
    assert future.error == "engine down"
    assert (
        resolve_sampler_checkpoint(
            service.config, "tenant", f"tinker://{model_id}/sampler_weights/1", service.backend.resolve_adapter
        )[0]
        == service.backend.named("sample")[0]["lora_name"]
    )
    fb = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    assert (await await_settled(service, "tenant", fb)).state == DONE


async def test_lease_expiry_reclaims_the_tenant(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    queued = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    await await_settled(service, "tenant", queued)
    stale = service.submit("tenant", "forward_backward", fb_payload(model_id, 3, [datum()]))  # gapped: stays queued

    service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()

    assert model_id not in service.models
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]
    assert service.retrieve_future("tenant", stale).state == FAILED


async def test_a_fresh_heartbeat_keeps_the_model(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)

    service.heartbeat("tenant", session_id)
    await service._expire_sessions()

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
    model_id = await created_model(service, session_id=session_id)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "kept", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]

    service.sessions[session_id].last_heartbeat = -1e9
    await service._expire_sessions()
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


async def test_a_failed_batch_terminates_the_model_queue(service):
    model_id = await created_model(service)
    first = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum(), datum()]))
    second = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    service.backend.fail_next = {"error": "batch input rejected"}
    assert (await await_settled(service, "tenant", first)).state == FAILED
    assert (
        await await_settled(service, "tenant", second)
    ).state == FAILED, "the failed accumulation terminates model training"
    assert model_id not in service.models
    assert service.backend.named("unload_slot")


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
    for payload in cases.values():
        model_id = await created_model(service)
        seq_id = 1
        request_id = service.submit(
            "tenant",
            "forward_backward",
            {"model_id": model_id, "seq_id": seq_id, "loss_fn_config": {}, **payload},
        )
        future = await await_settled(service, "tenant", request_id)
        assert (future.state, future.error_category) == (FAILED, "user")
    assert not service.backend.named("forward_backward"), "rejected datums must never reach the trainer"


async def test_a_failed_window_stops_already_queued_training(service):
    model_id = await created_model(service)
    healthy = await created_model(service)
    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", early)).state == DONE

    service.backend.fail_next = {"error": "batch rejected"}
    queued = [service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))]
    queued.append(service.submit("tenant", "optim_step", _optim_payload(model_id, 3)))
    queued.append(service.submit("tenant", "forward_backward", fb_payload(model_id, 4, [datum()])))
    queued.append(service.submit("tenant", "optim_step", _optim_payload(model_id, 5)))
    for request_id in queued:
        assert (await await_settled(service, "tenant", request_id)).state == FAILED
    assert service.retrieve_future("tenant", early).state == DONE
    assert not service.backend.named("optim_step")
    assert len(service.backend.named("forward_backward")) == 2
    with pytest.raises(UserInputError, match="create a new model"):
        service.submit("tenant", "optim_step", _optim_payload(model_id, 6))
    step = service.submit("tenant", "optim_step", _optim_payload(healthy, 1))
    assert (await await_settled(service, "tenant", step)).state == DONE


async def test_unsupported_lora_configs_are_rejected(service):
    for lora_config in ({"rank": 8, "seed": 7}, {"rank": 8, "train_unembed": True}, {"rank": 8, "train_mlp": False}):
        with pytest.raises(UserInputError):
            service.create_model("tenant", model_payload(service, lora_config=lora_config))
    _, model_id = service.create_model(
        "tenant",
        model_payload(service, lora_config={"rank": 8, "train_attn": True, "train_mlp": True, "train_unembed": False}),
    )
    assert model_id in service.models


@pytest.mark.parametrize("expire_lease", [False, True])
async def test_sampler_paths_resolve_independently_of_the_lease(service, expire_lease):
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", save)).result["path"]

    if expire_lease:
        service.sessions[session_id].last_heartbeat = -1e9
        await service._expire_sessions()
        assert model_id not in service.models
        service.create_session("tenant")
    lora_name, lora_path = resolve_sampler_checkpoint(service.config, "tenant", path, service.backend.resolve_adapter)
    # the physical commit behind the name, which an eviction backfill reloads
    assert lora_name.startswith(f"{model_id}@1@") and "/sampler_weights/.1." in lora_path
    with pytest.raises((UserInputError, OwnershipError)):
        resolve_sampler_checkpoint(service.config, "thief", path, service.backend.resolve_adapter)
    service.config.experts_shared_outer_loras = not service.config.experts_shared_outer_loras
    with pytest.raises(UserInputError, match="experts_shared_outer_loras"):
        resolve_sampler_checkpoint(service.config, "tenant", path, service.backend.resolve_adapter)
    service.config.experts_shared_outer_loras = not service.config.experts_shared_outer_loras
    service.config.base_model = "other"
    with pytest.raises(UserInputError, match="base_model"):
        resolve_sampler_checkpoint(service.config, "tenant", path, service.backend.resolve_adapter)


async def test_an_unnamed_sampler_save_returns_a_sampling_session(service):
    model_id = await created_model(service)
    unnamed = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "sampler_path": None}
    )
    result = (await await_settled(service, "tenant", unnamed)).result
    session = service.sampling_sessions[result["sampling_session_id"]]
    assert (session.tenant, session.model_path) == ("tenant", result["path"])

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
    service.backend.optim_outcomes[slot] = {"error": "slot update rejected"}

    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    future = await await_settled(service, "tenant", step)
    assert future.state == FAILED and "slot update rejected" in future.error
    assert model_id not in service.models, "a half-applied step may have diverged the slot across ranks"
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]

    with pytest.raises(UserInputError, match="restore from a checkpoint"):
        service.submit("tenant", "optim_step", _optim_payload(model_id, 2))


async def test_an_out_of_order_validation_failure_blocks_later_training(service):
    model_id = await created_model(service)
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    rejected = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [], loss_fn="unknown"))
    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", early)).state == DONE
    assert (await await_settled(service, "tenant", rejected)).state == FAILED
    assert (await await_settled(service, "tenant", step)).state == FAILED
    assert len(service.backend.named("forward_backward")) == 1
    assert not service.backend.named("optim_step")


async def test_a_backend_level_optim_failure_retires_every_model_in_the_barrier(service):
    model_a = await created_model(service)
    model_b = await created_model(service)
    service.backend.fail_on["optim_step"] = {"error": "slot updates rejected"}

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
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    service.backend.fail_on["unload_slot"] = {"error": "engine gone"}

    service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()

    assert model_id not in service.models
    assert slot not in service.free_slots, "a slot whose unload failed is dirty and must not be reused"
    await service._expire_sessions()  # the sweep itself survived


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
    assert service.backend.named("sample")[0]["lora_name"].startswith(f"{model_id}@v1@")

    reuse = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2, "sampler_path": "v1"}
    )
    future = await await_settled(service, "tenant", reuse)
    assert future.state == FAILED and "already exist" in future.error, "saved versions are immutable"


async def test_a_checkpoint_stores_a_digest_not_the_credential(service):
    import json

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    await await_settled(service, "tenant", saved)
    link = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck"))
    owner = json.loads((link.resolve() / "tenant.json").read_text())
    assert owner["tenant_digest"] != "tenant", "the bearer credential must not be persisted"
    info = service.weights_info("tenant", f"tinker://{model_id}/weights/ck")
    assert (info["train_attn"], info["train_mlp"], info["train_unembed"]) == (True, True, False)


@pytest.mark.parametrize("field,value", [("base_model", "other"), ("lora_rank", 3)])
async def test_a_checkpoint_of_another_base_or_rank_does_not_load(service, field, value):
    import json

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", saved)).result["path"]
    parametrization = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).resolve()
    fields = json.loads((parametrization / "adapter_parametrization.json").read_text()) | {field: value}
    (parametrization / "adapter_parametrization.json").write_text(json.dumps(fields))

    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "tenant", loaded)
    assert (future.state, future.error_category) == (FAILED, "user") and field in future.error
    assert not service.backend.named("load_slot")[1:], "nothing may touch the slot on a mismatch"


async def test_a_loaded_adapter_keeps_its_own_alpha(service):
    import json

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", saved)).result["path"]
    parametrization = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).resolve()
    fields = json.loads((parametrization / "adapter_parametrization.json").read_text()) | {"lora_alpha": 99.0}
    (parametrization / "adapter_parametrization.json").write_text(json.dumps(fields))

    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    assert (await await_settled(service, "tenant", loaded)).state == DONE
    load = service.backend.named("load_slot")[-1]
    assert load["alpha"] == load["parametrization_fields"]["lora_alpha"] == 99.0
    assert service.models[model_id].lora_alpha == 99.0, "the tenant record adopts the source's alpha"


async def test_a_checkpoint_saved_before_commits_loads_through_the_legacy_path(service):
    model_id = await created_model(service)
    record = service.models[model_id]
    legacy = resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "old")
    write_legacy_checkpoint(legacy, record, service.config)
    loaded = service.submit(
        "tenant",
        "load_state",
        {"model_id": model_id, "seq_id": 1, "path": f"tinker://{model_id}/weights/old", "optimizer": True},
    )
    assert (await await_settled(service, "tenant", loaded)).state == DONE
    assert service.backend.named("load_slot")[-1]["ckpt_path"] == str(Path(legacy).resolve())


async def test_an_overwritten_checkpoint_leaves_no_old_commit(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        saved = service.submit(
            "tenant", "save_state", {"model_id": model_id, "seq_id": seq_id, "name": "ck", "overwrite": True}
        )
        assert (await await_settled(service, "tenant", saved)).state == DONE
    await asyncio.gather(*service._cleanup_tasks)
    weights = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).parent
    assert sorted(p.name for p in weights.iterdir()) == sorted(["ck", Path(os.readlink(weights / "ck")).name])


async def test_a_recycled_slot_belongs_to_a_fresh_model_queue(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    service.backend.fail_next = {"error": "batch input rejected"}
    failed = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", failed)).state == FAILED

    service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()
    assert slot in service.free_slots

    assert session_id not in service.sessions, "the sweep takes the expired session with it"
    fresh = await created_model(service)
    assert service.models[fresh].slot == slot
    step_after_fb = service.submit("tenant", "forward_backward", fb_payload(fresh, 1, [datum()]))
    assert (await await_settled(service, "tenant", step_after_fb)).state == DONE
    step = service.submit("tenant", "optim_step", _optim_payload(fresh, 2))
    future = await await_settled(service, "tenant", step)
    assert future.state == DONE, "a fresh model may use the successfully recycled slot"


@pytest.mark.parametrize("source", ["handler", "worker", "create", "sweep", "export", "load"])
async def test_an_unknown_failure_stops_the_dispatcher(tmp_path, source):
    from tests.fast.tinker.harness import make_service

    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    error = (
        OSError("checkpoint IO failed") if source in ("export", "load") else RuntimeError("fatal execution failure")
    )
    try:
        model_id = await created_model(gateway)
        if source == "export":
            gateway.backend.fail_on["export_slot"] = error
            gateway.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
        elif source == "load":
            saved = gateway.submit(
                "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
            )
            path = (await await_settled(gateway, "tenant", saved)).result["path"]
            gateway.backend.fail_on["load_slot"] = error
            gateway.submit(
                "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
            )
        elif source == "create":
            gateway.backend.fail_on["load_slot"] = RuntimeError("fatal execution failure")
            gateway.create_model("tenant", model_payload(gateway))
        elif source == "sweep":
            gateway.backend.fail_on["unload_slot"] = RuntimeError("fatal execution failure")
            for session in gateway.sessions.values():
                session.last_heartbeat -= gateway.config.lease_timeout_s + 1
            task = asyncio.create_task(gateway._expire_sessions())
            task.add_done_callback(gateway._observe_background_task)
        else:
            if source == "handler":

                async def broken_handler(unit):
                    raise RuntimeError("fatal execution failure")

                gateway._run_barrier = broken_handler
            else:
                gateway.backend.fail_on["optim_step"] = RuntimeError("fatal execution failure")
            gateway.submit("tenant", "optim_step", _optim_payload(model_id, 1))
        with pytest.raises(type(error), match=str(error)):
            await asyncio.wait_for(run_task, timeout=2)
    finally:
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task


async def test_a_dead_trainer_escapes_the_dispatch_loop(tmp_path, monkeypatch):
    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    try:
        model_id = await created_model(gateway)

        async def _dead() -> bool:
            return True

        monkeypatch.setattr(gateway.backend, "trainer_dead", _dead)
        gateway.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
        with pytest.raises(RuntimeError, match="trainer workers died"):
            await asyncio.wait_for(run_task, timeout=2)
    finally:
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task


async def test_a_retried_create_model_returns_the_first_allocation(service):
    """A network retry must not occupy a second slot."""
    payload = model_payload(service)
    first = service.create_model("tenant", payload)
    await await_settled(service, "tenant", first[0])
    assert service.create_model("tenant", payload) == first
    assert len(service.backend.named("load_slot")) == 1


async def test_an_unknown_session_is_rejected(service):
    with pytest.raises(UserInputError, match="unknown session"):
        service.create_model("tenant", model_payload(service, session_id="session-gone"))


async def test_a_retried_sampling_session_returns_the_same_id(service):
    payload = {"session_id": service.create_session("tenant"), "sampling_session_seq_id": 1}
    assert service.create_sampling_session("tenant", payload) == service.create_sampling_session("tenant", payload)


async def test_a_retried_sample_does_not_sample_again(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", save)).result["path"]
    sampling_session_id = service.create_sampling_session(
        "tenant", {"session_id": service.create_session("tenant"), "sampling_session_seq_id": 1, "model_path": path}
    )
    payload = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 1,
        "num_samples": 1,
        "prompt_tokens": [1],
        "sampling_params": {"max_tokens": 2},
    }
    first = service.submit_sample("tenant", payload)
    assert service.submit_sample("tenant", payload) == first, "the retry must reuse the request and sequence ids"
    await await_settled(service, "tenant", first[0])
    assert len(service.backend.named("sample")) == 1


async def test_an_illegal_seq_id_is_rejected(service):
    """seq_id 0 would never enter the ordered model queue; every future would pend."""
    model_id = await created_model(service)
    with pytest.raises(UserInputError, match="seq_id"):
        service.submit("tenant", "optim_step", {"model_id": model_id, "seq_id": 0, "adam_params": dict(ADAM)})


async def test_an_expired_lease_sweeps_its_sessions(service):
    session_id = service.create_session("tenant")
    service.sessions[session_id].last_heartbeat = -1e9
    await service._expire_sessions()
    with pytest.raises(UserInputError, match="unknown session"):
        service.create_model("tenant", model_payload(service, session_id=session_id))


async def test_a_sampling_session_for_another_base_model_is_rejected(service):
    payload = {"session_id": service.create_session("tenant"), "sampling_session_seq_id": 1, "base_model": "other"}
    with pytest.raises(UserInputError, match="serves"):
        service.create_sampling_session("tenant", payload)


async def test_a_read_only_batch_failure_preserves_queued_training(service):
    model_id = await created_model(service)
    service.backend.fail_on["forward_only"] = {"error": "request rejected"}
    readonly = service.submit("tenant", "forward_only", fb_payload(model_id, 1, [datum()]))
    backward = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    assert (await await_settled(service, "tenant", readonly)).state == FAILED
    assert (await await_settled(service, "tenant", backward)).state == DONE
    assert (await await_settled(service, "tenant", step)).state == DONE


@pytest.mark.parametrize("shutdown", ["cancel", "failure"])
async def test_dispatcher_shutdown_stops_model_creation_and_sampling(tmp_path, monkeypatch, shutdown):
    gateway = make_service(tmp_path)
    started = asyncio.Queue()
    fail_sweep = asyncio.Event()

    async def blocked_backend_call(*args):
        started.put_nowait(asyncio.current_task())
        await asyncio.Event().wait()

    async def _run_lease_sweeper():
        await fail_sweep.wait()
        raise RuntimeError("lease sweeper failed")

    monkeypatch.setattr(gateway.backend, "load_slot", blocked_backend_call)
    monkeypatch.setattr(gateway.backend, "sample", blocked_backend_call)
    monkeypatch.setattr(gateway, "_run_lease_sweeper", _run_lease_sweeper)
    run_task = asyncio.create_task(gateway.run())
    gateway.create_model("tenant", model_payload(gateway))
    gateway.submit_sample("tenant", {"num_samples": 1, "prompt_tokens": [1], "sampling_params": {"max_tokens": 1}})
    tasks = [run_task, *gateway._create_tasks, *(task for task, _ in gateway._sample_tasks.values())]
    try:
        backend_tasks = [await asyncio.wait_for(started.get(), timeout=2) for _ in range(2)]
        if shutdown == "cancel":
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run_task, timeout=2)
        else:
            fail_sweep.set()
            with pytest.raises(RuntimeError, match="lease sweeper failed"):
                await asyncio.wait_for(run_task, timeout=2)
        assert all(task.done() for task in backend_tasks), "backend tasks outlived the dispatcher"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_a_sampler_dir_written_before_commits_still_resolves(service):
    model_id = await created_model(service)
    record = service.models[model_id]
    legacy = resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "old")
    write_legacy_checkpoint(legacy, record, service.config)
    assert resolve_sampler_checkpoint(
        service.config, "tenant", f"tinker://{model_id}/sampler_weights/old", service.backend.resolve_adapter
    ) == (
        f"{model_id}@old",
        legacy,
    )


async def test_another_tenants_sampler_export_does_not_resolve(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", save)).result["path"]
    with pytest.raises(OwnershipError):
        resolve_sampler_checkpoint(service.config, "mallory", path, service.backend.resolve_adapter)
    info = service.weights_info("tenant", path)
    assert (info["base_model"], info["lora_rank"]) == (
        service.models[model_id].base_model,
        service.models[model_id].lora_rank,
    )


def _external_payload(model_id: str, seq_id: int, name: str, **overrides) -> dict:
    return {
        "model_id": model_id,
        "seq_id": seq_id,
        "name": name,
        "ttl_seconds": None,
        "age_recipients": [],
    } | overrides


async def test_an_external_save_is_downloadable_by_its_tenant_only(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_external", _external_payload(model_id, 1, "export", ttl_seconds=7200)
    )
    result = (await await_settled(service, "tenant", save)).result
    assert result["path"] == f"tinker://{model_id}/external_weights/export" and result["size_bytes"] > 0
    assert service.expiry.expires_at(result["path"]) is not None

    urls = service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert sorted(urls["urls"]) == ["adapter_config.json", "adapter_model.safetensors"], "tenant.json stays private"
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    assert json.loads(Path(service.download_path(token, "adapter_config.json")).read_text())["r"] == 8
    with pytest.raises(NotFoundError):
        service.download_path(token, "tenant.json")
    with pytest.raises(NotFoundError):
        service.download_path("forged", "adapter_config.json")
    with pytest.raises(OwnershipError):
        service.external_weights_urls("mallory", model_id, "external_weights/export", "http://gateway/")
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 2, "name": "ck", "overwrite": False}
    )
    await await_settled(service, "tenant", save)
    with pytest.raises(UserInputError, match="not an external_weights"):
        service.external_weights_urls("tenant", model_id, "weights/ck", "http://gateway/")

    await service.delete_checkpoint("tenant", model_id, "external_weights/export")
    with pytest.raises(NotFoundError, match="deleted"):
        service.download_path(token, "adapter_config.json")
    assert service.expiry.expires_at(result["path"]) is None


async def test_an_encrypted_external_save_keeps_its_owner(service):
    """Only the export's files are encrypted; tenant.json stays readable, so the owner can list and delete it."""
    model_id = await created_model(service)
    recipient = str(pyrage.x25519.Identity.generate().to_public())
    save = service.submit(
        "tenant", "save_weights_external", _external_payload(model_id, 1, "export", age_recipients=[recipient])
    )
    assert (await await_settled(service, "tenant", save)).state == DONE
    urls = service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert sorted(urls["urls"]) == ["adapter_config.json.age", "adapter_model.safetensors.age"]
    with pytest.raises(OwnershipError):
        service.external_weights_urls("mallory", model_id, "external_weights/export", "http://gateway/")
    await service.delete_checkpoint("tenant", model_id, "external_weights/export")


async def test_a_download_url_does_not_serve_a_resave_under_the_same_name(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    assert (await await_settled(service, "tenant", save)).state == DONE
    urls = service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    await service.delete_checkpoint("tenant", model_id, "external_weights/export")
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 2, "export"))
    assert (await await_settled(service, "tenant", save)).state == DONE
    with pytest.raises(NotFoundError, match="saved again"):
        service.download_path(token, "adapter_config.json")


async def test_an_external_save_refuses_an_existing_name(service):
    model_id = await created_model(service)
    first = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    assert (await await_settled(service, "tenant", first)).state == DONE
    again = service.submit("tenant", "save_weights_external", _external_payload(model_id, 2, "export"))
    assert "already exist" in (await await_settled(service, "tenant", again)).error


async def test_a_download_url_expires_with_its_checkpoint(service, monkeypatch):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_external", _external_payload(model_id, 1, "export", ttl_seconds=3600)
    )
    await await_settled(service, "tenant", save)
    await service.set_checkpoint_ttl("tenant", model_id, "external_weights/export", 3600)
    urls = service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 3601)
    with pytest.raises(NotFoundError, match="expired"):
        service.download_path(token, "adapter_config.json")
    with pytest.raises(NotFoundError, match="has expired"):
        service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")


async def test_an_overwrite_that_crashes_before_its_ttl_does_not_inherit_the_old_expiry(service, monkeypatch):
    model_id = await created_model(service)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": 3600},
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).parent
    old_commit = resolve_adapter_dir(root, "ck", tenant=None).commit_id

    def crash(*args):
        raise RuntimeError("crashed after the commit")

    monkeypatch.setattr(service.expiry, "set", crash)
    with pytest.raises(RuntimeError, match="crashed after the commit"):
        await service._save_state(
            service.models[model_id], {"seq_id": 2, "name": "ck", "overwrite": True, "ttl_seconds": None}
        )
    assert resolve_adapter_dir(root, "ck", tenant=None).commit_id != old_commit, "the overwrite committed"
    assert CheckpointExpiry(service.config.checkpoint_root).expires_at(path) is None


async def test_ttl_is_set_removed_and_swept(service, monkeypatch):
    model_id = await created_model(service)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": 3600},
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    await service.set_checkpoint_ttl("tenant", model_id, "weights/ck", None)
    assert service.expiry.expires_at(path) is None
    with pytest.raises(UserInputError, match="ttl_seconds"):
        await service.set_checkpoint_ttl("tenant", model_id, "weights/ck", 60)
    with pytest.raises(OwnershipError):
        await service.set_checkpoint_ttl("mallory", model_id, "weights/ck", 7200)
    await service.set_checkpoint_ttl("tenant", model_id, "weights/ck", 7200)

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 7201)
    await service._expire_checkpoints()
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).parent
    assert resolve_adapter_dir(root, "ck", tenant=None) is None and service.expiry.expires_at(path) is None
    with pytest.raises(NotFoundError):
        await service.delete_checkpoint("tenant", model_id, "weights/ck")


@pytest.mark.parametrize(
    "filename,content",
    [
        ("adapter_config.json", "{not json"),
        ("adapter_config.json", json.dumps({"lora_alpha": 32})),
        ("adapter_parametrization.json", json.dumps({"lora_rank": 8})),
    ],
    ids=["bad_json", "no_rank", "bad_parametrization"],
)
async def test_a_malformed_import_is_a_user_error_and_the_gateway_keeps_serving(tmp_path, filename, content):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / filename).write_text(content)
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    run_task = asyncio.create_task(service.run())
    try:
        with pytest.raises(UserInputError, match=re.escape(str(adapter.resolve()))):
            service.weights_info("tenant", f"file://{adapter}")
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        loaded = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        assert loaded.state == FAILED and str(adapter.resolve()) in loaded.error
        assert not run_task.done(), "the dispatcher survived"
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


@pytest.mark.parametrize(
    "kind,recorded_base,accepted",
    [
        ("peft", None, True),
        ("peft", "/models/base", True),
        ("peft", "org/other", False),
        ("peft", "/models/other", False),
        ("adapter_dir", "/models/base", True),
        ("adapter_dir", "other", False),
    ],
)
def test_an_import_trained_on_another_base_is_refused(tmp_path, kind, recorded_base, accepted):
    """The gateway serves "base"; a source that records no base (official Tinker writes null) is accepted."""
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    if kind == "peft":
        config = {"r": 8, "lora_alpha": 32, "base_model_name_or_path": recorded_base}
        (adapter / "adapter_config.json").write_text(json.dumps(config))
    else:
        write_parametrization(
            adapter, AdapterParametrization(recorded_base, 8, 32.0, "lora", ("model.layers.0.self_attn.q_proj",), False)
        )
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    if accepted:
        assert service.weights_info("tenant", f"file://{adapter}")["lora_rank"] == 8
    else:
        with pytest.raises(UserInputError, match=f"trained on '{recorded_base}'"):
            service.weights_info("tenant", f"file://{adapter}")


async def test_file_imports_stay_under_the_import_root_and_load_weights_only(tmp_path):
    imports = tmp_path / "imports"
    adapter = imports / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (imports / "link").symlink_to(tmp_path / "outside")
    service = make_service(tmp_path / "root", import_root=str(imports))
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        info = service.weights_info("tenant", f"file://{adapter}")
        assert (info["base_model"], info["lora_rank"]) == ("base", 8)

        def load(model_id, path, optimizer=False):
            payload = {"model_id": model_id, "seq_id": 1, "path": path, "optimizer": optimizer}
            return await_settled(service, "tenant", service.submit("tenant", "load_state", payload))

        assert (await load(model_id, f"file://{adapter}")).state == DONE
        loaded = service.backend.named("load_slot")[-1]
        assert (loaded["ckpt_path"], loaded["load_optimizer"]) == (str(adapter.resolve()), False)
        assert service.models[model_id].lora_alpha == 32
        # a failed load closes its model (freeing its slot), so each refusal gets a new one
        for path, error in (
            (f"file://{imports}/../outside", "outside the import root"),
            (f"file://{imports}/link", "outside the import root"),
            (f"file://{adapter}", "weights only"),
        ):
            model_id = await created_model(service)
            assert error in (await load(model_id, path, optimizer=error == "weights only")).error
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_trainer_ranks_get_only_the_tenant_digest(service):
    """The API key is the tenant's credential; it must not travel to trainer processes."""
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    assert (await await_settled(service, "tenant", save)).state == DONE
    (export,) = service.backend.named("export_slot")
    assert export["tenant_digest"] == hashlib.sha256(b"tenant").hexdigest()
