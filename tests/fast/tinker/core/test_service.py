"""The service preserves request ordering, tenant isolation, and failure isolation."""

import asyncio
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
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

import miles.tinker.core.service as service_module
from miles.backends.training_utils.checkpoint.adapter.commit import _adopt_pre_commit_dir, resolve_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.tinker.core.checkpoint_expiry import CheckpointExpiry
from miles.tinker.core.future import DONE, FAILED
from miles.tinker.core.service import IMPORT_COPIES_DIRNAME
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
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ckpt", "overwrite": False, "ttl_seconds": None},
    )
    saved_path = (await await_settled(service, "tenant", save)).result["path"]
    assert saved_path == f"tinker://{model_id}/weights/ckpt"

    load = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": saved_path, "optimizer": False}
    )
    assert (await await_settled(service, "tenant", load)).state == DONE
    weights_only = service.backend.named("load_slot")[-1]
    assert weights_only["source"]["load_optimizer"] is False
    checkpoint = Path(service.config.checkpoint_root) / model_id / "weights" / "ckpt"
    # the trainer reads the commit the name pointed at when the load was scheduled
    assert weights_only["source"]["path"] == str(checkpoint.resolve()) and checkpoint.is_symlink()


async def test_sampler_save_publishes_successive_versions(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        request_id = service.submit(
            "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id, "ttl_seconds": None}
        )
        future = await await_settled(service, "tenant", request_id)
        assert future.result["path"] == f"tinker://{model_id}/sampler_weights/{seq_id}"
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1")).parent
    assert [(export["root"], export["adapter_name"]) for export in service.backend.named("export_slot")] == [
        (str(root), "1"),
        (str(root), "2"),
    ]
    assert all(resolve_adapter_dir(root, name, tenant="tenant") is not None for name in ("1", "2")), "both committed"


async def test_an_export_records_the_base_fingerprint_the_gateway_serves(service):
    service.config = replace(service.config, base_model_fingerprint="0123456789abcdef")
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    assert (await await_settled(service, "tenant", save)).state == DONE
    (export,) = service.backend.named("export_slot")
    assert export["parametrization_fields"]["base_model_fingerprint"] == "0123456789abcdef"


async def test_sampling_resolves_against_the_saved_version(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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
    request_id = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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
    request_id = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": name, "overwrite": False, "ttl_seconds": None},
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
    payload = {"model_id": model_id, "name": "ckpt", "overwrite": False, "ttl_seconds": None}
    first = service.submit("tenant", "save_state", payload | {"seq_id": 1})
    assert (await await_settled(service, "tenant", first)).state == DONE

    clobber = service.submit("tenant", "save_state", payload | {"seq_id": 2})
    future = await await_settled(service, "tenant", clobber)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "overwrite" in future.error

    replace = service.submit("tenant", "save_state", payload | {"seq_id": 3, "overwrite": True, "ttl_seconds": None})
    assert (await await_settled(service, "tenant", replace)).state == DONE


async def test_checkpoints_outlive_the_lease(service):
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "kept", "overwrite": False, "ttl_seconds": None},
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
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "mine", "overwrite": False, "ttl_seconds": None},
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
    version = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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


@pytest.mark.parametrize("op", ["save_state", "save_weights_for_sampler"])
async def test_the_commit_runs_off_the_event_loop(service, monkeypatch, op):
    model_id = await created_model(service)
    commit = service.backend.commit_adapter
    commit_threads = []

    def recording_commit(root, name, written, overwrite):
        commit_threads.append(threading.current_thread())
        commit(root, name, written, overwrite)

    monkeypatch.setattr(service.backend, "commit_adapter", recording_commit)
    payload = {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None}
    save = service.submit(
        "tenant", op, payload if op == "save_state" else {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    assert (await await_settled(service, "tenant", save)).state == DONE
    assert commit_threads and commit_threads[0] is not threading.current_thread()


async def test_a_failed_sampler_commit_fails_only_that_request(service, monkeypatch):
    model_id = await created_model(service)
    commit = service.backend.commit_adapter

    def failing_commit(*_, **__):
        raise OSError("No space left on device")

    monkeypatch.setattr(service.backend, "commit_adapter", failing_commit)
    failed = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    future = await await_settled(service, "tenant", failed)
    assert (future.state, future.error_category) == (FAILED, "server")
    assert "No space left on device" in future.error

    monkeypatch.setattr(service.backend, "commit_adapter", commit)
    retried = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2, "ttl_seconds": None}
    )
    assert (await await_settled(service, "tenant", retried)).state == DONE


async def test_an_unnamed_sampler_save_returns_a_sampling_session(service):
    model_id = await created_model(service)
    unnamed = service.submit(
        "tenant",
        "save_weights_for_sampler",
        {"model_id": model_id, "seq_id": 1, "sampler_path": None, "ttl_seconds": None},
    )
    result = (await await_settled(service, "tenant", unnamed)).result
    session = service.sampling_sessions[result["sampling_session_id"]]
    assert (session.tenant, session.model_path) == ("tenant", result["path"])

    named = service.submit(
        "tenant",
        "save_weights_for_sampler",
        {"model_id": model_id, "seq_id": 2, "sampler_path": "ckpt", "ttl_seconds": None},
    )
    assert "sampling_session_id" not in (await await_settled(service, "tenant", named)).result


async def test_weights_info_reads_the_checkpoint_not_the_lease(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
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


async def test_weights_info_reports_the_checkpoints_groups_not_the_gateways(service):
    import json

    model_id = await created_model(service)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    # an attn-only adapter, as another gateway or an import would have written it
    committed = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).resolve()
    fields = json.loads((committed / "adapter_parametrization.json").read_text())
    fields["target_modules"] = ["model.layers.0.self_attn.q_proj"]
    (committed / "adapter_parametrization.json").write_text(json.dumps(fields))
    assert service.config.trains_mlp

    info = service.weights_info("tenant", path)

    assert (info["train_attn"], info["train_mlp"], info["train_unembed"]) == (True, False, False)


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
        "tenant",
        "save_weights_for_sampler",
        {"model_id": model_id, "seq_id": 1, "sampler_path": "v1", "ttl_seconds": None},
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
        "tenant",
        "save_weights_for_sampler",
        {"model_id": model_id, "seq_id": 2, "sampler_path": "v1", "ttl_seconds": None},
    )
    future = await await_settled(service, "tenant", reuse)
    assert future.state == FAILED and "already exist" in future.error, "saved versions are immutable"


async def test_a_checkpoint_stores_a_digest_not_the_credential(service):
    import json

    model_id = await created_model(service)
    saved = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
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
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
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


@pytest.mark.parametrize("refused_by", ["gateway", "trainer"])
async def test_a_refused_load_leaves_the_model_usable(service, refused_by):
    import json

    model_id = await created_model(service)
    saved = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
    )
    path = (await await_settled(service, "tenant", saved)).result["path"]
    if refused_by == "gateway":
        committed = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).resolve()
        fields = json.loads((committed / "adapter_parametrization.json").read_text()) | {"lora_rank": 3}
        (committed / "adapter_parametrization.json").write_text(json.dumps(fields))
    else:
        service.backend.fail_on["load_slot"] = {"error": "another adapter layout", "error_category": "user"}

    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "tenant", loaded)
    assert (future.state, future.error_category) == (FAILED, "user")
    saved = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 3, "name": "after", "overwrite": False, "ttl_seconds": None},
    )
    assert (await await_settled(service, "tenant", saved)).state == DONE


async def test_a_loaded_adapter_keeps_its_own_alpha(service):
    import json

    model_id = await created_model(service)
    saved = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
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
    assert load["alpha"] == load["source"]["parametrization_fields"]["lora_alpha"] == 99.0
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
    assert service.backend.named("load_slot")[-1]["source"]["path"] == str(Path(legacy).resolve())


async def test_an_overwrite_interrupted_after_adopting_a_legacy_dir_keeps_it_loadable(service):
    model_id = await created_model(service)
    legacy = resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "old")
    write_legacy_checkpoint(legacy, service.models[model_id], service.config)
    # the state a crash leaves between the adoption and the link swap
    _adopt_pre_commit_dir(Path(legacy).parent, "old")
    path = f"tinker://{model_id}/weights/old"
    assert service.weights_info("tenant", path)["lora_rank"] == service.models[model_id].lora_rank
    with pytest.raises(OwnershipError):
        service.weights_info("intruder", path)
    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 1, "path": path, "optimizer": True}
    )
    assert (await await_settled(service, "tenant", loaded)).state == DONE
    assert service.backend.named("load_slot")[-1]["source"]["path"] == str(Path(legacy).resolve())
    with pytest.raises(UserInputError, match="saved before checkpoints could expire"):
        await service.set_checkpoint_ttl("tenant", model_id, "weights/old", 7200)
    with pytest.raises(UserInputError, match="saved before checkpoints could expire"):
        await service.delete_checkpoint("tenant", model_id, "weights/old")


async def test_an_overwritten_checkpoint_leaves_no_old_commit(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        saved = service.submit(
            "tenant",
            "save_state",
            {"model_id": model_id, "seq_id": seq_id, "name": "ck", "overwrite": True, "ttl_seconds": None},
        )
        assert (await await_settled(service, "tenant", saved)).state == DONE
    await asyncio.gather(*service._cleanup_tasks)
    weights = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).parent
    assert sorted(p.name for p in weights.iterdir()) == sorted(["ck", Path(os.readlink(weights / "ck")).name])


async def test_overlapping_cleanups_run_one_at_a_time(service, monkeypatch):
    model_id = await created_model(service)
    delete = service.backend.delete_retired_adapters
    release, active, peak = threading.Event(), [0], [0]

    def blocking_delete(root):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        release.wait(timeout=5)
        active[0] -= 1
        delete(root)

    monkeypatch.setattr(service.backend, "delete_retired_adapters", blocking_delete)
    for seq_id in (1, 2):
        saved = service.submit(
            "tenant",
            "save_state",
            {"model_id": model_id, "seq_id": seq_id, "name": "ck", "overwrite": True, "ttl_seconds": None},
        )
        assert (await await_settled(service, "tenant", saved)).state == DONE
    await asyncio.sleep(0.1)
    release.set()
    await asyncio.gather(*service._cleanup_tasks)
    assert peak[0] == 1


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


async def test_a_failed_save_commit_fails_only_that_request(service, monkeypatch):
    def failing_commit(*_, **__):
        raise OSError("checkpoint IO failed")

    monkeypatch.setattr(service.backend, "commit_adapter", failing_commit)
    model_id = await created_model(service)
    payload = {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None}
    future = await await_settled(service, "tenant", service.submit("tenant", "save_state", payload))
    assert (future.state, future.error_category) == (FAILED, "server")
    assert "checkpoint IO failed" in future.error
    step = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    assert (await await_settled(service, "tenant", step)).state == DONE


async def test_a_failed_cleanup_after_a_save_keeps_the_gateway_serving(service):
    def failing_delete(root):
        raise OSError("checkpoint IO failed")

    service.backend.delete_retired_adapters = failing_delete
    model_id = await created_model(service)
    for seq_id in (1, 2):
        payload = {"model_id": model_id, "seq_id": seq_id, "name": "ck", "overwrite": True, "ttl_seconds": None}
        assert (await await_settled(service, "tenant", service.submit("tenant", "save_state", payload))).state == DONE
        await asyncio.gather(*service._cleanup_tasks)


async def test_a_stopping_dispatcher_waits_for_pending_cleanups(tmp_path):
    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    release, deleted = threading.Event(), threading.Event()
    delete = gateway.backend.delete_retired_adapters

    def blocking_delete(root):
        release.wait(timeout=5)
        delete(root)
        deleted.set()

    gateway.backend.delete_retired_adapters = blocking_delete
    try:
        model_id = await created_model(gateway)
        saved = gateway.submit(
            "tenant",
            "save_state",
            {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
        )
        assert (await await_settled(gateway, "tenant", saved)).state == DONE
        gateway.backend.fail_on["optim_step"] = RuntimeError("fatal execution failure")
        gateway.submit("tenant", "optim_step", _optim_payload(model_id, 2))
        asyncio.get_running_loop().call_later(0.2, release.set)
        with pytest.raises(RuntimeError, match="fatal execution failure"):
            await asyncio.wait_for(run_task, timeout=2)
        assert deleted.is_set(), "the dispatcher returned while a cleanup was still deleting"
    finally:
        release.set()
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task


async def test_an_overwrite_onto_a_foreign_link_is_a_user_error(service, tmp_path):
    model_id = await created_model(service)
    weights = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck")).parent
    weights.mkdir(parents=True)
    (weights / "ck").symlink_to(tmp_path / "shared_adapter")
    payload = {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": True, "ttl_seconds": None}
    failed = await await_settled(service, "tenant", service.submit("tenant", "save_state", payload))
    assert failed.state == FAILED and "did not write" in failed.error
    payload = {"model_id": model_id, "seq_id": 2, "name": "other", "overwrite": False, "ttl_seconds": None}
    assert (await await_settled(service, "tenant", service.submit("tenant", "save_state", payload))).state == DONE


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
            gateway.submit(
                "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
            )
        elif source == "load":
            saved = gateway.submit(
                "tenant",
                "save_state",
                {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
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
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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


@pytest.mark.parametrize("filename", ["tenant.json", "adapter_parametrization.json"])
async def test_a_truncated_sampler_commit_file_is_a_user_error(service, filename):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1")).parent
    commit_file = resolve_adapter_dir(root, "1", tenant=None).path / filename
    commit_file.write_text(commit_file.read_text()[:-1])
    with pytest.raises(UserInputError, match="cannot read adapter"):
        service.weights_info("tenant", path)
    with pytest.raises(UserInputError, match="cannot read adapter"):
        resolve_sampler_checkpoint(service.config, "tenant", path, service.backend.resolve_adapter)


@pytest.mark.parametrize(
    "filename, content", [("tenant.json", "{}"), ("tenant.json", "[]"), ("adapter_parametrization.json", "{}")]
)
async def test_a_sampler_commit_file_of_the_wrong_schema_is_a_user_error(service, filename, content):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1")).parent
    (resolve_adapter_dir(root, "1", tenant=None).path / filename).write_text(content)
    with pytest.raises(UserInputError, match="cannot read adapter"):
        service.weights_info("tenant", path)
    with pytest.raises(UserInputError, match="cannot read adapter"):
        resolve_sampler_checkpoint(service.config, "tenant", path, service.backend.resolve_adapter)


async def test_another_tenants_sampler_export_does_not_resolve(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
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

    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert sorted(urls["urls"]) == ["adapter_config.json", "adapter_model.safetensors"], "tenant.json stays private"
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    assert json.loads(Path(await service.download_path(token, "adapter_config.json")).read_text())["r"] == 8
    with pytest.raises(NotFoundError):
        await service.download_path(token, "tenant.json")
    with pytest.raises(NotFoundError):
        await service.download_path("forged", "adapter_config.json")
    with pytest.raises(OwnershipError):
        await service.external_weights_urls("mallory", model_id, "external_weights/export", "http://gateway/")
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 2, "name": "ck", "overwrite": False, "ttl_seconds": None},
    )
    await await_settled(service, "tenant", save)
    with pytest.raises(UserInputError, match="not an external_weights"):
        await service.external_weights_urls("tenant", model_id, "weights/ck", "http://gateway/")

    await service.delete_checkpoint("tenant", model_id, "external_weights/export")
    with pytest.raises(NotFoundError, match="deleted"):
        await service.download_path(token, "adapter_config.json")
    assert service.expiry.expires_at(result["path"]) is None


async def test_repeated_url_requests_reuse_one_token_and_the_sweeper_drops_expired_ones(service, monkeypatch):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    await await_settled(service, "tenant", save)

    def token():
        return urls["urls"]["adapter_config.json"].split("/")[-2]

    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    first = token()
    for _ in range(3):
        urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert token() == first and len(service._downloads) == 1, "a client polling for URLs does not grow the table"

    service._drop_expired_downloads(time.time() + 3601)
    assert not service._downloads and not service._download_tokens
    with pytest.raises(NotFoundError):
        await service.download_path(first, "adapter_config.json")
    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert token() != first, "an expired token is never handed out again"


async def test_a_download_token_never_outlives_the_expiry_it_was_issued_with(service, monkeypatch):
    """The URL is the credential: a request after its expiry, before the sweeper runs, must not revive it."""
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    await await_settled(service, "tenant", save)
    now = [time.time()]
    monkeypatch.setattr(service_module.time, "time", lambda: now[0])

    def token(urls):
        return urls["urls"]["adapter_config.json"].split("/")[-2]

    first = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    reported = datetime.fromisoformat(first["expires"]).timestamp()
    now[0] += 3601
    with pytest.raises(NotFoundError):
        await service.download_path(token(first), "adapter_config.json")
    again = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert token(again) != token(first), "an expired token is never handed out again, swept or not"
    with pytest.raises(NotFoundError):
        await service.download_path(token(first), "adapter_config.json")
    assert service._downloads[token(again)].expires_at > now[0]
    # a reused token keeps the expiry it was issued with
    now[0] += 10
    reused = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert token(reused) == token(again) and reused["expires"] == again["expires"]
    # the table never lets a token live past what its response reported
    issued = datetime.fromisoformat(again["expires"]).timestamp()
    assert service._downloads[token(again)].expires_at == pytest.approx(issued, abs=1e-5)
    assert service._downloads.get(token(first)) is None or service._downloads[
        token(first)
    ].expires_at == pytest.approx(reported, abs=1e-5)


async def test_an_encrypted_external_save_keeps_its_owner(service):
    """Only the export's files are encrypted; tenant.json stays readable, so the owner can list and delete it."""
    model_id = await created_model(service)
    recipient = str(pyrage.x25519.Identity.generate().to_public())
    save = service.submit(
        "tenant", "save_weights_external", _external_payload(model_id, 1, "export", age_recipients=[recipient])
    )
    assert (await await_settled(service, "tenant", save)).state == DONE
    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    assert sorted(urls["urls"]) == ["adapter_config.json.age", "adapter_model.safetensors.age"]
    with pytest.raises(OwnershipError):
        await service.external_weights_urls("mallory", model_id, "external_weights/export", "http://gateway/")
    await service.delete_checkpoint("tenant", model_id, "external_weights/export")


async def test_an_external_save_is_refused_for_a_family_without_checked_tinker_names(service):
    service.config.exports_tinker_names = False
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    assert "no checked official Tinker names" in (await await_settled(service, "tenant", save)).error
    assert not service.backend.named("export_external")


async def test_a_failed_external_commit_fails_the_request_not_the_gateway(service, monkeypatch):
    model_id = await created_model(service)
    commit = service.backend.commit_adapter

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(service.backend, "commit_adapter", full_disk)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    settled = await await_settled(service, "tenant", save)
    assert settled.state == FAILED and "No space left" in settled.error
    monkeypatch.setattr(service.backend, "commit_adapter", commit)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 2, "export"))
    assert (await await_settled(service, "tenant", save)).state == DONE, "the gateway keeps serving"


async def test_a_download_url_does_not_serve_a_resave_under_the_same_name(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    assert (await await_settled(service, "tenant", save)).state == DONE
    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    await service.delete_checkpoint("tenant", model_id, "external_weights/export")
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 2, "export"))
    assert (await await_settled(service, "tenant", save)).state == DONE
    with pytest.raises(NotFoundError, match="saved again"):
        await service.download_path(token, "adapter_config.json")


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
    real_time = time.time
    # signed with half an hour of the checkpoint left, less than a URL's hour
    monkeypatch.setattr(time, "time", lambda: real_time() + 1800)
    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    monkeypatch.setattr(time, "time", lambda: real_time() + 3601)
    with pytest.raises(NotFoundError, match="expired"):
        await service.download_path(token, "adapter_config.json")
    with pytest.raises(NotFoundError, match="has expired"):
        await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")


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


@pytest.mark.parametrize("swapped", [False, True], ids=["before_the_swap", "after_the_swap"])
async def test_a_failed_overwrite_commit_keeps_the_old_expiry_only_on_the_old_commit(service, monkeypatch, swapped):
    model_id = await created_model(service)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": 3600},
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    old_expires_at = service.expiry.expires_at(path)
    commit = service.backend.commit_adapter

    def failing_commit(root, name, written, overwrite):
        if swapped:
            commit(root, name, written, overwrite=overwrite)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(service.backend, "commit_adapter", failing_commit)
    failed = await service._save_state(
        service.models[model_id], {"seq_id": 2, "name": "ck", "overwrite": True, "ttl_seconds": None}
    )
    assert failed["error_category"] == "server" and "No space left" in failed["error"]
    expected = None if swapped else old_expires_at
    assert CheckpointExpiry(service.config.checkpoint_root).expires_at(path) == expected


def _full_disk(*args, **kwargs):
    raise OSError(28, "No space left on device")


@pytest.mark.parametrize("op", ["save_state", "save_weights_for_sampler", "save_weights_external"])
async def test_a_failed_expiry_write_after_a_commit_keeps_the_save_and_the_gateway(service, monkeypatch, op):
    model_id = await created_model(service)
    payloads = {
        "save_state": lambda seq_id: {
            "model_id": model_id,
            "seq_id": seq_id,
            "name": f"ck{seq_id}",
            "overwrite": False,
            "ttl_seconds": 3600,
        },
        "save_weights_for_sampler": lambda seq_id: {"model_id": model_id, "seq_id": seq_id, "ttl_seconds": 3600},
        "save_weights_external": lambda seq_id: _external_payload(model_id, seq_id, f"e{seq_id}", ttl_seconds=3600),
    }
    monkeypatch.setattr(service.expiry, "_write", _full_disk)
    if op == "save_weights_external":
        # the commit is done; sizing it is a read after it
        monkeypatch.setattr(service.backend, "resolve_checkpoint_files", _full_disk)
    for seq_id in (1, 2):
        saved = await await_settled(service, "tenant", service.submit("tenant", op, payloads[op](seq_id)))
        assert saved.state == DONE, saved.error
        # recorded nowhere, so it never expires, in this process or after a restart
        assert service.expiry.expires_at(saved.result["path"]) is None
    if op == "save_weights_external":
        assert saved.result["size_bytes"] is None


@pytest.mark.parametrize("fails", ["live_commit_id", "drop"])
async def test_a_failure_before_an_overwrite_commit_fails_only_that_request(service, monkeypatch, fails):
    model_id = await created_model(service)

    def save(seq_id):
        payload = {"model_id": model_id, "seq_id": seq_id, "name": "ck", "overwrite": True, "ttl_seconds": 3600}
        return await_settled(service, "tenant", service.submit("tenant", "save_state", payload))

    path = (await save(1)).result["path"]
    old_expires_at = service.expiry.expires_at(path)
    if fails == "live_commit_id":

        def foreign_link(root, name):
            raise ValueError("ck is a link this library did not write")

        monkeypatch.setattr(service, "_live_commit_id", foreign_link)
    else:
        monkeypatch.setattr(service.expiry, "_write", _full_disk)
    failed = await save(2)
    assert failed.state == FAILED
    assert failed.error_category == ("user" if fails == "live_commit_id" else "server")
    assert service.expiry.expires_at(path) == old_expires_at
    assert CheckpointExpiry(service.config.checkpoint_root).expires_at(path) == old_expires_at
    monkeypatch.undo()
    assert (await save(3)).state == DONE, "the gateway keeps serving"


async def test_a_sweep_with_nothing_expired_does_not_wait_for_the_trainer(service):
    async with service._trainer_lock:
        await asyncio.wait_for(service._expire_checkpoints(), timeout=1.0)


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


async def test_a_failed_expiry_delete_is_retried_and_the_rest_are_swept(service, monkeypatch):
    model_id = await created_model(service)
    paths = {}
    for seq_id, name in enumerate(("stuck", "ok"), start=1):
        payload = {"model_id": model_id, "seq_id": seq_id, "name": name, "overwrite": False, "ttl_seconds": 3600}
        paths[name] = (await await_settled(service, "tenant", service.submit("tenant", "save_state", payload))).result[
            "path"
        ]
    real_delete = service.backend.delete_adapter

    def delete_adapter(root, name, tenant):
        if name == "stuck":
            raise PermissionError(13, "Permission denied")
        real_delete(root, name, tenant)

    monkeypatch.setattr(service.backend, "delete_adapter", delete_adapter)
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 3601)
    await service._expire_checkpoints()
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ok")).parent
    assert resolve_adapter_dir(root, "ok", tenant=None) is None and service.expiry.expires_at(paths["ok"]) is None
    assert resolve_adapter_dir(root, "stuck", tenant=None) is not None
    assert service.expiry.expires_at(paths["stuck"]) is not None, "kept for the next sweep"


async def test_an_expiring_name_relinked_by_an_operator_is_kept_and_the_rest_are_swept(service, tmp_path_factory):
    """The sweep also runs before the dispatcher starts, so an entry it cannot delete must not stop it."""
    model_id = await created_model(service)
    paths = {}
    for seq_id, name in enumerate(("outside", "foreign", "ok"), start=1):
        payload = {"model_id": model_id, "seq_id": seq_id, "name": name, "overwrite": False, "ttl_seconds": 3600}
        settled = await await_settled(service, "tenant", service.submit("tenant", "save_state", payload))
        paths[name] = settled.result["path"]
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ok")).parent
    # e.g. a shared adapter: out of the checkpoint root, or in it but not written by the library
    shared = tmp_path_factory.mktemp("shared")
    (root / "shared.adapter").mkdir()
    for name, target in (("outside", shared), ("foreign", root / "shared.adapter")):
        (root / name).unlink()
        (root / name).symlink_to(target)
    real_time = time.time
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(time, "time", lambda: real_time() + 3601)
        await service._expire_checkpoints()
    assert resolve_adapter_dir(root, "ok", tenant=None) is None and service.expiry.expires_at(paths["ok"]) is None
    assert shared.is_dir() and (root / "shared.adapter").is_dir()
    assert service.expiry.expires_at(paths["outside"]) is not None
    assert service.expiry.expires_at(paths["foreign"]) is not None


@pytest.mark.parametrize(
    "filename,content",
    [
        ("adapter_config.json", "{not json"),
        ("adapter_config.json", json.dumps({"lora_alpha": 32})),
        ("adapter_parametrization.json", json.dumps({"lora_rank": 8})),
        ("adapter_config.json", json.dumps({"r": 8, "lora_alpha": "32"})),
    ],
    ids=["bad_json", "no_rank", "bad_parametrization", "string_alpha"],
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


@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b"], ids=["nul", "lone_surrogate"])
async def test_an_unencodable_import_path_is_a_user_error_and_the_gateway_keeps_serving(tmp_path, bad):
    (tmp_path / "imports").mkdir()
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{tmp_path}/imports/{bad}", "optimizer": False}
        loaded = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        assert loaded.state == FAILED and "cannot read" in loaded.error
        assert not run_task.done(), "the dispatcher survived"
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


def test_an_import_with_a_string_alpha_is_a_user_error_under_python_O(tmp_path):
    """python -O strips asserts; the client's types must be checked without them."""
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": "32"}))
    code = (
        "from miles.tinker.core.types import UserInputError\n"
        "from miles.tinker.runtime import read_imported_adapter\n"
        "try:\n"
        f"    read_imported_adapter({str(tmp_path)!r}, 'file:///adapter')\n"
        "except UserInputError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('accepted a string alpha')\n"
    )
    subprocess.run([sys.executable, "-O", "-c", code], check=True)


@pytest.mark.parametrize(
    "served,kind,recorded_base,accepted",
    [
        ("base", "peft", None, True),
        ("base", "peft", "base", True),
        ("base", "peft", "/models/base", False),
        ("base", "peft", "org/other", False),
        ("{models}/base/hf", "peft", "{models}/base/hf", True),
        ("{models}/base/hf", "peft", "{models}/link/hf", True),
        ("{models}/base/hf", "peft", "{models}/other/hf", False),
        ("{models}/base/hf", "peft", "hf", False),
        ("{models}/base/hf", "adapter_dir", "{models}/base/hf", True),
        ("{models}/base/hf", "adapter_dir", "other", False),
    ],
)
def test_an_import_trained_on_another_base_is_refused(tmp_path, served, kind, recorded_base, accepted):
    """A source that records no base (official Tinker writes null) is accepted."""
    models = tmp_path / "models"
    for name in ("base", "other"):
        (models / name / "hf").mkdir(parents=True)
    (models / "link").symlink_to(models / "base")
    served = served.format(models=models)
    recorded_base = recorded_base and recorded_base.format(models=models)
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    if kind == "peft":
        config = {"r": 8, "lora_alpha": 32, "base_model_name_or_path": recorded_base}
        (adapter / "adapter_config.json").write_text(json.dumps(config))
    else:
        write_parametrization(
            adapter,
            AdapterParametrization(recorded_base, 8, 32.0, "lora", ("model.layers.0.self_attn.q_proj",), False),
        )
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"), base_model=served)
    if accepted:
        assert service.weights_info("tenant", f"file://{adapter}")["lora_rank"] == 8
    else:
        with pytest.raises(UserInputError, match=re.escape(f"trained on '{recorded_base}'")):
            service.weights_info("tenant", f"file://{adapter}")


async def test_file_imports_stay_under_the_import_root_and_load_weights_only(tmp_path):
    imports = tmp_path / "imports"
    adapter = imports / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (imports / "link").symlink_to(tmp_path / "outside")
    (imports / "symlinked").mkdir()
    (imports / "symlinked" / "adapter_config.json").symlink_to(tmp_path / "outside" / "adapter_config.json")
    service = make_service(tmp_path / "root", import_root=str(imports))
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        info = service.weights_info("tenant", f"file://{adapter}")
        assert (info["base_model"], info["lora_rank"]) == ("base", 8)

        def load(model_id, path, optimizer=False, seq_id=1):
            payload = {"model_id": model_id, "seq_id": seq_id, "path": path, "optimizer": optimizer}
            return await_settled(service, "tenant", service.submit("tenant", "load_state", payload))

        assert (await load(model_id, f"file://{adapter}")).state == DONE
        loaded = service.backend.named("load_slot")[-1]
        copies = tmp_path / "root" / IMPORT_COPIES_DIRNAME
        assert Path(loaded["source"]["path"]).parent == copies and loaded["source"]["load_optimizer"] is False
        assert not any(copies.iterdir()), "the copy outlived its load"
        assert service.models[model_id].lora_alpha == 32
        # a refused load leaves the model open
        for seq_id, (path, error) in enumerate(
            (
                (f"file://{imports}/../outside", "outside the import root"),
                (f"file://{imports}/link", "symlink"),
                (f"file://{imports}/symlinked", "symlink"),
                (f"file://{adapter}", "weights only"),
            ),
            start=2,
        ):
            assert error in (await load(model_id, path, optimizer=error == "weights only", seq_id=seq_id)).error
        assert not any(copies.iterdir()), "a refused import left its copy"
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_a_file_import_swapped_after_the_checks_cannot_change_what_the_trainer_reads(tmp_path, monkeypatch):
    """The import walk and the trainer's read used to be two opens of the tenant's path; the trainer reads a copy."""
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    victim = tmp_path / "root" / "victim" / "weights" / "x" / "adapter_config.json"
    victim.parent.mkdir(parents=True)
    victim.write_text(json.dumps({"r": 8, "lora_alpha": 32, "secret": True}))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    load_slot, read = service.backend.load_slot, []

    async def trainer_load_slot(slot, rank, alpha, source=None):
        ckpt_path = None if source is None else source["path"]
        if ckpt_path is not None:
            # the tenant swaps in a link to another tenant's checkpoint once the gateway's checks are done
            (adapter / "adapter_config.json").unlink()
            (adapter / "adapter_config.json").symlink_to(victim)
            read.append(json.loads((Path(ckpt_path) / "adapter_config.json").read_text()))
        return await load_slot(slot, rank, alpha, source=source)

    monkeypatch.setattr(service.backend, "load_slot", trainer_load_slot)
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        assert (await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))).state == DONE
        assert read == [{"r": 8, "lora_alpha": 32}]
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_an_import_that_cannot_be_copied_fails_only_its_load(tmp_path):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    # stands in for an unwritable or full checkpoint root
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / IMPORT_COPIES_DIRNAME).write_text("")
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        loaded = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        assert loaded.state == FAILED and loaded.error_category == "server", loaded.error
        assert not run_task.done(), "the dispatcher survived"
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_a_save_pretrained_or_hf_download_dir_is_importable(tmp_path, monkeypatch):
    """Files no reader opens (save_pretrained's README.md, hf download --local-dir's .cache/) are left behind."""
    adapter = tmp_path / "imports" / "adapter"
    (adapter / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "README.md").write_text("---\nlibrary_name: peft\n---\n")
    (adapter / ".cache" / "huggingface" / "download" / "adapter_model.safetensors.metadata").write_text("abc\n")
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    load_slot, copied = service.backend.load_slot, []

    async def trainer_load_slot(slot, rank, alpha, source=None):
        ckpt_path = None if source is None else source["path"]
        if ckpt_path is not None:
            copied.append(sorted(p.name for p in Path(ckpt_path).iterdir()))
        return await load_slot(slot, rank, alpha, source=source)

    monkeypatch.setattr(service.backend, "load_slot", trainer_load_slot)
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        loaded = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        assert loaded.state == DONE, loaded.error
        assert copied == [["adapter_config.json", "adapter_model.safetensors"]]
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_a_file_import_load_reads_the_import_off_the_event_loop(service, monkeypatch):
    model_id = await created_model(service)
    threads = []

    def copy_import(uri, suffixes):
        threads.append(threading.current_thread())
        return tempfile.mkdtemp()

    def read_import_copy(uri, copy_dir):
        threads.append(threading.current_thread())
        return {"lora_rank": service.models[model_id].lora_rank, "lora_alpha": 32}

    monkeypatch.setattr(service, "_copy_import", copy_import)
    monkeypatch.setattr(service, "_read_import_copy", read_import_copy)
    payload = {"model_id": model_id, "seq_id": 1, "path": "file:///imports/adapter", "optimizer": False}
    assert (await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))).state == DONE
    assert len(threads) == 2 and threading.main_thread() not in threads


async def test_weights_info_rejects_a_sampler_link_whose_target_is_gone(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    link = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1"))
    shutil.rmtree(resolve_adapter_dir(link.parent, "1", tenant="tenant").path)
    with pytest.raises(UserInputError, match="unknown checkpoint"):
        service.weights_info("tenant", path)


async def test_trainer_ranks_get_only_the_tenant_digest(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    assert (await await_settled(service, "tenant", save)).state == DONE
    (export,) = service.backend.named("export_slot")
    assert export["tenant_digest"] == hashlib.sha256(b"tenant").hexdigest()


async def test_a_ttl_change_does_not_wait_for_a_training_call(service):
    """The TTL is serialized with deletes, which the design requires, not with the trainer."""
    model_id = await created_model(service)
    save = service.submit(
        "tenant",
        "save_state",
        {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False, "ttl_seconds": None},
    )
    assert (await await_settled(service, "tenant", save)).state == DONE
    # a training call in progress
    async with service._trainer_lock:
        await asyncio.wait_for(service.set_checkpoint_ttl("tenant", model_id, "weights/ck", 7200), timeout=1.0)
    assert service.expiry.expires_at(f"tinker://{model_id}/weights/ck") is not None


async def test_a_legacy_checkpoint_cannot_be_deleted_through_the_api(service):
    model_id = await created_model(service)
    write_legacy_checkpoint(
        resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "old"),
        service.models[model_id],
        service.config,
    )
    with pytest.raises(UserInputError, match="saved before checkpoints could expire"):
        await service.delete_checkpoint("tenant", model_id, "weights/old")
    with pytest.raises(UserInputError, match="not an external_weights"):
        await service.external_weights_urls("tenant", model_id, "weights/old", "http://gateway/")


async def test_an_unreadable_import_is_a_user_error(tmp_path):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    (adapter / "adapter_config.json").chmod(0)
    try:
        if os.access(adapter / "adapter_config.json", os.R_OK):
            pytest.skip("running as root: chmod 000 does not stop reads")
        with pytest.raises(UserInputError, match="cannot read"):
            service.weights_info("tenant", f"file://{adapter}")
    finally:
        (adapter / "adapter_config.json").chmod(0o644)


def _link_to_blob(link: Path) -> None:
    blob = link.parent.parent / "blobs" / "0123abcd"
    blob.parent.mkdir()
    blob.write_bytes(b"")
    link.symlink_to(blob)


@pytest.mark.parametrize(
    "plant, error",
    [
        # no reader opens it, so where it points does not matter
        (lambda adapter, tmp_path: (adapter / "weights").symlink_to(tmp_path / "outside"), None),
        (lambda adapter, tmp_path: (adapter / "everything").symlink_to("/"), None),
        # an HF cache snapshot links its files to blobs/; hf download --local-dir writes the files themselves
        (lambda adapter, tmp_path: _link_to_blob(adapter / "adapter_model.safetensors"), "symlink"),
        # opening a FIFO blocks until a writer appears
        (lambda adapter, tmp_path: os.mkfifo(adapter / "adapter_parametrization.json"), "not a regular file"),
    ],
)
async def test_a_file_import_reads_only_its_own_adapter_files(tmp_path, plant, error):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (tmp_path / "outside").mkdir()
    plant(adapter, tmp_path)
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        loaded = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        if error is None:
            assert loaded.state == DONE, loaded.error
        else:
            assert loaded.state == FAILED and error in loaded.error
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


@pytest.mark.parametrize("name", ["adapter_config.json", "adapter_parametrization.json"])
def test_a_directory_named_like_an_adapter_file_is_refused_without_leaking_its_fd(tmp_path, name):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / name).mkdir()
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    # a tenant repeating this would otherwise run every tenant's gateway out of file descriptors
    open_fds = len(os.listdir("/dev/fd"))
    for _ in range(3):
        with pytest.raises(UserInputError, match="not a regular file"):
            service.weights_info("tenant", f"file://{adapter}")
    assert len(os.listdir("/dev/fd")) == open_fds


async def test_an_import_larger_than_any_adapter_is_refused_before_it_is_copied(tmp_path):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    # sparse: costs no disk here, but a copy would write every byte onto the shared checkpoint disk
    with open(adapter / "adapter_model.safetensors", "wb") as weights:
        weights.truncate(1 << 40)
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"), import_max_bytes=1 << 20)
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        loaded = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        assert loaded.state == FAILED and "more than any adapter" in loaded.error
        assert not any((tmp_path / "root" / IMPORT_COPIES_DIRNAME).iterdir()), "nothing was left on the disk"
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


def test_weights_info_copies_only_the_files_it_reads(tmp_path):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    with open(adapter / "tokenizer.json", "wb") as other:
        other.truncate(1 << 40)
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"), import_max_bytes=1 << 20)

    assert service.weights_info("tenant", f"file://{adapter}")["lora_rank"] == 8


async def test_a_file_import_load_copies_only_the_files_a_reader_opens(tmp_path, monkeypatch):
    """Thousands of tiny .json files cost a copy each under the trainer lock and an inode each; none is read."""
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    for index in range(50):
        (adapter / f"pad{index}.json").write_text("")
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    load_slot, copied = service.backend.load_slot, []

    async def trainer_load_slot(slot, rank, alpha, source=None):
        ckpt_path = None if source is None else source["path"]
        if ckpt_path is not None:
            copied.append(sorted(p.name for p in Path(ckpt_path).iterdir()))
        return await load_slot(slot, rank, alpha, source=source)

    monkeypatch.setattr(service.backend, "load_slot", trainer_load_slot)
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        assert copied == [["adapter_config.json"]]
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


def test_an_import_json_larger_than_any_adapter_config_is_refused_before_it_is_parsed(tmp_path):
    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    # within the whole import's budget, but parsed whole by the gateway and every trainer rank
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32, "pad": "x" * (2 << 20)}))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"), import_max_bytes=1 << 30)

    with pytest.raises(UserInputError, match="adapter_config.json is larger than"):
        service.weights_info("tenant", f"file://{adapter}")


def test_a_committed_checkpoint_dir_cannot_be_imported(tmp_path):
    """A checkpoint store mounted under the import root would otherwise hand any tenant another tenant's adapter."""
    adapter = tmp_path / "imports" / "mounted"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    (adapter / "tenant.json").write_text(json.dumps({"tenant_digest": "someone"}))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))

    with pytest.raises(UserInputError, match="is a committed checkpoint"):
        service.weights_info("tenant", f"file://{adapter}")


def test_a_training_adapter_dir_cannot_be_imported(tmp_path):
    """The copy keeps only its bf16 PEFT view: importing it would silently round the fp32 masters a tenant expects."""
    adapter = tmp_path / "imports" / "rl-save"
    (adapter / "adapter_state").mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))

    with pytest.raises(UserInputError, match="is a training adapter dir; import its sampler or PEFT export"):
        service.weights_info("tenant", f"file://{adapter}")


class _WritesMarker:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


@pytest.mark.parametrize("pickled", [".metadata", "adapter_state/.metadata", "adapter_megatron_rank0.pt"])
async def test_a_pickle_in_a_file_import_never_reaches_the_trainer(tmp_path, monkeypatch, pickled):
    from miles.backends.training_utils.checkpoint.adapter.adapter_dir import check_adapter_source
    from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

    adapter = tmp_path / "imports" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    run = AdapterParametrization("base", 8, 32.0, "lora", ("model.layers.0.self_attn.q_proj",), False)
    if pickled.startswith("adapter_state/"):
        write_parametrization(adapter, run)
    marker = tmp_path / "code_ran"
    (adapter / pickled).parent.mkdir(exist_ok=True)
    (adapter / pickled).write_bytes(pickle.dumps(_WritesMarker(marker)))
    service = make_service(tmp_path / "root", import_root=str(tmp_path / "imports"))
    load_slot, trainer_read = service.backend.load_slot, []

    async def trainer_load_slot(slot, rank, alpha, source=None):
        ckpt_path = None if source is None else source["path"]
        # None: a new model's empty slot
        if ckpt_path is not None:
            trainer_read.append(sorted(str(p.relative_to(ckpt_path)) for p in Path(ckpt_path).rglob("*")))
            # every trainer rank's dry run of the source (find_load_mismatch), on what the trainer is given
            targets = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
            with suppress(Exception):
                check_adapter_source(Path(ckpt_path), run, targets, load_optimizer=False)
        return await load_slot(slot, rank, alpha, source=source)

    monkeypatch.setattr(service.backend, "load_slot", trainer_load_slot)
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        payload = {"model_id": model_id, "seq_id": 1, "path": f"file://{adapter}", "optimizer": False}
        future = await await_settled(service, "tenant", service.submit("tenant", "load_state", payload))
        if pickled.startswith("adapter_state/"):
            # a training adapter dir is refused before anything is copied
            assert "is a training adapter dir" in future.error and not trainer_read
        else:
            assert trainer_read, "the trainer was never reached"
            assert not any(pickled in files for files in trainer_read)
        assert not marker.exists(), "the client's pickle ran"
        assert not run_task.done()
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_weights_info_reports_a_sampler_exports_own_groups_not_the_gateways(service):
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "ttl_seconds": None}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    root = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1")).parent
    # an mlp-only export, as another gateway would have written it
    parametrization = resolve_adapter_dir(root, "1", tenant=None).path / "adapter_parametrization.json"
    fields = json.loads(parametrization.read_text())
    fields["target_modules"] = ["model.layers.0.mlp.gate_proj"]
    parametrization.write_text(json.dumps(fields))
    assert service.config.trains_attn

    info = service.weights_info("tenant", path)

    assert (info["train_attn"], info["train_mlp"], info["train_unembed"]) == (False, True, False)


async def test_a_checkpoint_name_linked_outside_the_library_is_a_user_error_on_every_api(service, tmp_path):
    """A foreign link under a checkpoint name answers 4xx on delete, TTL, URLs and download, not 500 (SDK retries 5xx)."""
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_external", _external_payload(model_id, 1, "export"))
    await await_settled(service, "tenant", save)
    urls = await service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/")
    token = urls["urls"]["adapter_config.json"].split("/")[-2]
    link = Path(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "external_weights", "export"))
    shared = tmp_path / "shared-adapter"
    shared.mkdir()
    link.unlink()
    link.symlink_to(shared)

    for call in (
        service.delete_checkpoint("tenant", model_id, "external_weights/export"),
        service.set_checkpoint_ttl("tenant", model_id, "external_weights/export", 7200),
        service.external_weights_urls("tenant", model_id, "external_weights/export", "http://gateway/"),
        service.download_path(token, "adapter_config.json"),
    ):
        with pytest.raises(UserInputError, match="this library did not write"):
            await call
    assert shared.is_dir(), "the foreign dir is never touched"
