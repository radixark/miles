import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.multi_lora.backend import MultiLoraOperationBackend
from miles.ray.multi_lora.config import AdapterRunConfig
from miles.ray.multi_lora.registry import AdapterState


def make_backend(max_adapters: int = 4) -> MultiLoraOperationBackend:
    args = SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save="/tmp/tinker-test-save",
        lora_rank=32,
        lora_alpha=64,
        hf_checkpoint="Qwen/Qwen3-0.6B",
    )
    return MultiLoraOperationBackend(args, "http://unused")


def register(backend, name="X", **overrides) -> dict:
    return asyncio.run(backend.register(name, AdapterRunConfig(**overrides)))


def ready_backend(num_step=None):
    backend = make_backend()
    register(backend, num_step=num_step)
    backend.registry.mark_ready(["X"])
    return backend


def reg_key(backend, name="X"):
    """The exact registration key batch commits carry."""
    return (name, backend.registry.find(name).registration_id)


def fb_payload(n=1, loss_fn="cross_entropy"):
    return {
        "samples": [
            {"tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1], "loss_weights": [1.0, 1.0]}
            for _ in range(n)
        ],
        "loss": {"loss_fn": loss_fn},
    }


class TestRegistration:
    def test_resolves_rank_alpha_and_save(self):
        backend = make_backend()
        result = register(backend, rank=8)
        assert result == {"name": "X", "slot": 0}
        config = backend.registry.find("X").config
        assert config.rank == 8 and config.alpha == 64  # alpha is deployment-set
        assert str(config.save).endswith("adapters/X")

    def test_rank_ceiling_and_client_alpha_rejected(self):
        backend = make_backend()
        with pytest.raises(ValueError, match="exceeds the deployment maximum"):
            register(backend, rank=64)
        with pytest.raises(ValueError, match="must not set alpha"):
            register(backend, alpha=16)


class TestPreflight:
    def test_unsupported_loss_is_a_boundary_error(self):
        backend = ready_backend()
        with pytest.raises(ValueError, match="not supported in v1"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", fb_payload(loss_fn="cispo"))

    def test_multimodal_and_nested_targets_rejected(self):
        backend = ready_backend()
        bad = fb_payload()
        bad["samples"][0]["multimodal_inputs"] = {"image": "..."}
        with pytest.raises(ValueError, match="text-only"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", bad)
        nested = fb_payload()
        nested["samples"][0]["loss_weights"] = [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(ValueError, match="1-D"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", nested)

    def test_channel_length_must_match_response(self):
        backend = ready_backend()
        bad = fb_payload()
        bad["samples"][0]["advantages"] = [1.0]
        with pytest.raises(ValueError, match="length response_length"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", bad)

    def test_adam_params_validated(self):
        backend = ready_backend()
        with pytest.raises(ValueError, match="unknown adam_params field"):
            backend.enqueue_operation("X", "op1", 1, "optim_step", {"adam_params": {"lr": 1e-4}})
        with pytest.raises(ValueError, match="finite number"):
            backend.enqueue_operation("X", "op1", 1, "optim_step", {"adam_params": {"learning_rate": "fast"}})

    def test_adam_params_domain_checked_at_the_boundary(self):
        # The GPU-side veto only guards non-finite GRADIENTS: a NaN rate or an
        # out-of-range beta would silently poison the slot's param groups.
        backend = ready_backend()
        rejected = [
            {"learning_rate": float("nan")},
            {"learning_rate": float("inf")},
            {"learning_rate": -1e-4},
            {"beta1": 2.0},
            {"beta2": -0.1},
            {"beta1": 1.0},  # beta < 1 strictly
            {"eps": 0.0},
            {"eps": -1e-8},
            {"weight_decay": float("nan")},
            {"weight_decay": -0.1},
            {"grad_clip_norm": -1.0},
            {"learning_rate": True},  # bool is not a number here
        ]
        for adam in rejected:
            with pytest.raises(ValueError, match="adam_params"):
                backend.enqueue_operation("X", "op1", 1, "optim_step", {"adam_params": adam})
        ok = {"learning_rate": 3e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0}
        assert backend.enqueue_operation("X", "op1", 1, "optim_step", {"adam_params": ok})["state"] == "QUEUED"

    def test_loss_required_channels_preflighted(self):
        backend = ready_backend()
        # CE without loss_weights would only fail inside the GPU loss dispatch.
        ce = fb_payload()
        del ce["samples"][0]["loss_weights"]
        with pytest.raises(ValueError, match="loss_weights"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", ce)
        for missing in ("rollout_log_probs", "advantages"):
            for loss_fn in ("importance_sampling", "ppo"):
                bad = fb_payload(loss_fn=loss_fn)
                del bad["samples"][0]["loss_weights"]
                bad["samples"][0]["rollout_log_probs"] = [-1.0, -1.0]
                bad["samples"][0]["advantages"] = [0.5, 0.5]
                del bad["samples"][0][missing]
                with pytest.raises(ValueError, match=missing):
                    backend.enqueue_operation("X", "op1", 1, "forward_backward", bad)
        # forward has no loss: no channels are required.
        bare = {"samples": [{"tokens": [1, 2, 3, 4], "response_length": 2}]}
        assert backend.enqueue_operation("X", "op2", 1, "forward", bare)["state"] == "QUEUED"

    def test_response_must_leave_a_context_token(self):
        # Targets are shifted: the first response token's logprob conditions on
        # the previous position, so response_length == len(tokens) is invalid.
        backend = ready_backend()
        bad = fb_payload()
        bad["samples"][0].update(response_length=4, loss_mask=[1] * 4, loss_weights=[1.0] * 4)
        with pytest.raises(ValueError, match="response_length"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", bad)

    def test_unknown_kind_and_missing_path(self):
        backend = ready_backend()
        with pytest.raises(ValueError, match="unknown operation kind"):
            backend.enqueue_operation("X", "op1", 1, "publish_snapshot")
        with pytest.raises(ValueError, match="needs a 'path'"):
            backend.enqueue_operation("X", "op1", 1, "load_state", {})

    def test_save_state_tag_must_stay_inside_states(self):
        backend = ready_backend()
        for bad in ("..", ".", "a/b", "a" * 129, ""):
            with pytest.raises(ValueError, match="tag"):
                backend.enqueue_operation("X", f"save-{len(bad)}", 1, "save_state", {"tag": bad})
        assert backend.enqueue_operation("X", "save-ok", 1, "save_state", {"tag": "step_5.final"})


class TestControlClaims:
    def test_claim_requires_ready_and_serialization(self):
        backend = make_backend()
        register(backend)
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        assert backend.claim_ready_control_operations() == {"operations": [], "lease": None}
        backend.registry.mark_ready(["X"])
        claimed = backend.claim_ready_control_operations()
        [op] = claimed["operations"]
        assert op["operation_id"] == "opt1"
        # The claim carries no slot: the batch lease is the single binding truth.
        assert "slot" not in op
        rid = backend.registry.find("X").registration_id
        assert claimed["lease"]["bindings_by_operation"] == [["opt1", ["X", rid, 0]]]

    def test_claim_carries_authoritative_clocks(self):
        backend = ready_backend()
        backend.set_adapter_step("X", 7)
        backend.registry.record_weight_update(["X"])
        backend.enqueue_operation("X", "pub1", 1, "save_weights_for_sampler")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert op["step"] == 7 and op["serving_version"] == 1

    def test_dirty_slot_fails_state_moves_but_allows_publish(self):
        backend = ready_backend()
        backend.commit_tinker_batch([reg_key(backend)], [])
        backend.enqueue_operation("X", "save1", 1, "save_state", {"tag": "t0"})
        assert backend.claim_ready_control_operations() == {"operations": [], "lease": None}
        view = backend.operations.get("save1")
        assert view["state"] == "FAILED" and "unstepped gradients" in view["error"]

        backend.enqueue_operation("X", "pub1", 2, "save_weights_for_sampler")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert op["operation_id"] == "pub1"

    def test_success_advances_step_and_releases_pin(self):
        backend = ready_backend(num_step=2)
        backend.commit_tinker_batch([reg_key(backend)], [])
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations({op["operation_id"]: dict(ok=True, result={"grad_norm": 0.5})})
        record = backend.registry.find("X")
        assert record.step == 1 and not backend.registry.is_dirty("X")

    def test_veto_fails_without_advancing(self):
        backend = ready_backend()
        backend.commit_tinker_batch([reg_key(backend)], [])
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        # The executor's veto zeroed the gradients on every rank, so its
        # outcome carries the consumed bit — only then is the pin released.
        backend.complete_control_operations(
            {op["operation_id"]: dict(ok=False, error="veto", category="server", gradient_window_consumed=True)}
        )
        assert backend.registry.find("X").step == 0
        assert not backend.registry.is_dirty("X")

    def test_failed_chunk_poisons_the_pending_optim(self):
        # The failed chunk's window must discard, never partial-step.
        backend = ready_backend()
        rid = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("X", rid)
        backend.operations.fail("fb1", "bad chunk", "user")
        backend.enqueue_operation("X", "opt2", 2, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert "gradient window" in op["poison"] and "discarded" in op["poison"]
        # The trainer runs the discard on every rank and reports a user
        # failure whose outcome confirms the window was consumed.
        backend.complete_control_operations(
            {"opt2": dict(ok=False, error=op["poison"], category="user", gradient_window_consumed=True)}
        )
        assert backend.registry.find("X").step == 0

        # The executed (poison-consuming) optim delimits: the next round is clean.
        backend.enqueue_operation("X", "fb3", 3, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("X", rid)
        backend.commit_tinker_batch([reg_key(backend)], ["fb3"], {"fb3": [[-0.1, -0.2]]})
        backend.enqueue_operation("X", "opt4", 4, "optim_step")
        [clean] = backend.claim_ready_control_operations()["operations"]
        assert clean["operation_id"] == "opt4" and "poison" not in clean

    def test_pre_mutation_refusal_keeps_dirty_and_poison(self):
        """An optimizer outcome without the consumed bit (executor refusal
        before any gradient mutation — stale binding,
        missing result) must neither release the dirty pin nor delimit the
        poison window: the partial gradients still physically exist and the
        next optim_step must still be routed to a discard."""
        backend = ready_backend()
        rid = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        backend.claim_data_operation("X", rid)
        backend.commit_tinker_batch([reg_key(backend)], ["fb1"], {"fb1": [[-0.1, -0.2]]})
        backend.enqueue_operation("X", "fb2", 2, "forward_backward", fb_payload())
        backend.claim_data_operation("X", rid)
        backend.operations.fail("fb2", "partial backward failed", "server")

        backend.enqueue_operation("X", "opt3", 3, "optim_step")
        [poisoned] = backend.claim_ready_control_operations()["operations"]
        assert poisoned.get("poison")
        backend.complete_control_operations(
            {"opt3": dict(ok=False, error="stale binding: no gradients were cleared", category="server")}
        )

        backend.enqueue_operation("X", "opt4", 4, "optim_step")
        [next_optim] = backend.claim_ready_control_operations()["operations"]
        assert backend.gradient_windows.is_dirty(("X", rid))
        assert next_optim.get("poison"), "a refused optimizer dispatch is not a window delimiter"

    def test_stale_registration_handle_is_fenced(self):
        backend = ready_backend()
        rid1 = backend.registry.find("X").registration_id
        assert backend.enqueue_operation("X", "op1", 1, "optim_step", None, expected_registration_id=rid1)
        backend.registry.deregister("X")
        backend.registry.retire_adapters()
        backend.registry.free_slot("X")
        register(backend, "X")
        rid2 = backend.registry.records["X"].registration_id
        assert rid2 != rid1
        with pytest.raises(ValueError, match="fenced"):
            backend.enqueue_operation("X", "op9", 1, "optim_step", None, expected_registration_id=rid1)
        assert backend.operations.queue_view("X", rid2) == []
        # A stale-handle deregister must never retire the successor.
        asyncio.run(backend.deregister("X", rid1))
        assert backend.registry.records["X"].state is AdapterState.PENDING

    def test_publish_completion_stamps_post_push_serving_identity(self):
        backend = ready_backend()
        backend.registry.record_weight_update(["X"])
        backend.enqueue_operation("X", "pub1", 1, "save_weights_for_sampler")
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations({op["operation_id"]: dict(ok=True, result={})})
        result = backend.operations.get("pub1")["result"]
        assert result["serving_version"] == 1
        reg_id = backend.registry.find("X").registration_id
        assert result["serving_name"] == f"__miles_adapter_X_{reg_id}"

    def test_load_state_repositions_the_clock(self):
        backend = ready_backend()
        backend.enqueue_operation("X", "load1", 1, "load_state", {"path": "/tmp/state"})
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations({op["operation_id"]: dict(ok=True, result={"step": 42})})
        record = backend.registry.find("X")
        assert record.step == 42 and record.start_step == 42


class TestCommitAndFence:
    def test_commit_completes_data_ops_with_row_ordered_logprobs(self):
        backend = ready_backend()
        reg_id = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("X", reg_id)
        backend.commit_tinker_batch([reg_key(backend)], ["fb1"], {"fb1": [[-0.1, -0.2]]})
        result = backend.operations.get("fb1")["result"]
        assert result["logprobs"] == [[-0.1, -0.2]]
        assert result["metrics"]["loss:sum"] == pytest.approx(0.1 + 0.2)  # unit loss_weights
        assert backend.registry.is_dirty("X")

    def test_retirement_fences_open_operations(self, monkeypatch):
        backend = ready_backend()
        backend.enqueue_operation("X", "op1", 1, "forward_backward", fb_payload())

        async def no_abort(name, registration_id):
            pass

        monkeypatch.setattr(backend, "abort_adapter_requests", no_abort)
        asyncio.run(backend.deregister("X"))
        asyncio.run(backend.retire_adapters())
        view = backend.operations.get("op1")
        assert view["state"] == "FAILED" and view["error_category"] == "user"
        with pytest.raises(ValueError, match="not accepting operations"):
            backend.enqueue_operation("X", "op2", 2, "forward_backward", fb_payload())
        assert backend.registry.records["X"].state is AdapterState.CLEANUP


class TestFailTinkerBatch:
    """Data operations must not remain claimed when training exits without
    committing."""

    def _claimed_batch(self, backend):
        rid = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        claim = backend.claim_data_operation("X", rid)
        lease = backend.acquire_batch_lease([("fb1", claim["binding"])])
        from miles.ray.multi_lora.residency import lease_to_metadata

        return lease_to_metadata(lease)

    def test_uncommitted_batch_terminal_fails_claimed_operations_typed_server(self):
        backend = ready_backend()
        lease_metadata = self._claimed_batch(backend)
        backend.fail_tinker_batch(["fb1"], "train step finished without committing", lease_metadata)
        view = backend.operations.get("fb1")
        assert view["state"] == "FAILED" and view["error_category"] == "server"
        assert "without committing" in view["error"]

    def test_finalized_forward_backward_is_poison_evidence_for_the_next_optim(self):
        # The finalizer must PRESERVE poison semantics, not bypass them: the
        # failed forward_backward left possibly-partial gradients, so the
        # next optim_step is routed to a discard.
        backend = ready_backend()
        lease_metadata = self._claimed_batch(backend)
        backend.fail_tinker_batch(["fb1"], "abnormal train outcome", lease_metadata)
        backend.enqueue_operation("X", "opt2", 2, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert "forward_backward ordinal 1" in op["poison"]

    def test_already_terminal_operations_are_left_untouched(self):
        # A late finalization after a partial commit must never overwrite a
        # landed result.
        backend = ready_backend()
        lease_metadata = self._claimed_batch(backend)
        backend.commit_tinker_batch([reg_key(backend)], ["fb1"], {"fb1": [[-0.1, -0.2]]})
        backend.fail_tinker_batch(["fb1"], "late failure", lease_metadata)
        view = backend.operations.get("fb1")
        assert view["state"] == "SUCCEEDED" and view["result"]["logprobs"] == [[-0.1, -0.2]]

    def test_lease_releases_even_when_the_ledger_walk_raises(self):
        backend = ready_backend()
        lease_metadata = self._claimed_batch(backend)
        released = []
        backend.residency.release_batch = lambda lease: released.append(lease.dispatch_id)

        def boom(operation_id, error, category="server"):
            raise RuntimeError("ledger unavailable")

        backend.operations.fail = boom
        with pytest.raises(RuntimeError, match="ledger unavailable"):
            backend.fail_tinker_batch(["fb1"], "abnormal train outcome", lease_metadata)
        assert released == [lease_metadata["dispatch_id"]]

    def test_unknown_operation_ids_and_missing_lease_are_tolerated(self):
        # Finalizing is best-effort bookkeeping: a batch whose operations were
        # already fenced away (retirement) must not crash the driver loop.
        backend = ready_backend()
        backend.fail_tinker_batch(["ghost"], "abnormal train outcome", None)


def test_service_info_reports_the_v1_matrix():
    backend = ready_backend()
    info = backend.service_info()
    assert info["base_model"] == "Qwen/Qwen3-0.6B"
    assert info["lora_rank_max"] == 32 and info["n_adapters"] == 4
    assert info["occupied_slots"] == [0] and info["ready_adapters"] == ["X"]
    assert info["supported_loss_fns"] == ["cross_entropy", "importance_sampling", "ppo"]


def test_engine_aborts_go_through_the_inference_admin_port():
    # The backend's only engine-facing need rides the narrow admin port with
    # the full registration-scoped rid prefix (anti-ABA).
    backend = make_backend()
    aborted = []

    class FakeAdmin:
        async def abort_registration(self, rid_prefix):
            aborted.append(rid_prefix)

    backend.inference_admin = FakeAdmin()
    asyncio.run(backend.abort_adapter_requests("X", "reg-1"))
    assert aborted == ["X::reg-1::"]


def test_trainer_readiness_flag_flips_once_marked():
    # Liveness comes up with the HTTP server; readiness only when the driver
    # says the trainer exists (probes must not report ok on a dead trainer).
    backend = make_backend()
    assert backend.trainer_ready is False
    backend.mark_trainer_ready()
    assert backend.trainer_ready is True


def test_advertised_host_is_the_bind_host():
    # A loopback bind must never advertise the node IP: that URL would not
    # reach the socket.
    from miles.ray.multi_lora.http_server import AdapterRunControlServer

    assert AdapterRunControlServer(None, host="127.0.0.1").advertised_host == "127.0.0.1"


class TestGapTimeoutSurface:
    """Backend wiring of the ledger gap timeout: the flag reaches the ledger,
    the driver's control-claim heartbeat enforces it, and the stall is a
    typed, observable surface (operation_view + service_info)."""

    def stalled_backend(self, timeout=30.0):
        backend = ready_backend()
        backend.operations.gap_timeout = timeout
        clock = {"now": 1000.0}
        backend.operations._time = lambda: clock["now"]
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        backend.claim_data_operation(*reg_key(backend))
        backend.operations.complete("fb1", {})
        # Ordinal 2 was consumed client-side and never posted; 3 arrives.
        backend.enqueue_operation("X", "opt3", 3, "optim_step", {"adam_params": {"learning_rate": 1e-4}})
        assert backend.claim_ready_control_operations()["operations"] == []  # blocked, and arms the clock
        return backend, clock

    def test_flag_reaches_the_ledger_with_a_default(self):
        assert make_backend().operations.gap_timeout == 600.0
        args = SimpleNamespace(multi_lora_n_adapters=4, tinker_operation_gap_timeout=5.0)
        assert MultiLoraOperationBackend(args, "http://unused").operations.gap_timeout == 5.0

    def test_stall_is_typed_and_observable_before_expiry(self):
        backend, clock = self.stalled_backend()
        clock["now"] += 10
        info = backend.service_info()
        assert info["operation_gap_timeout"] == 30.0
        [stall] = info["gap_stalls"]
        assert stall["missing_ordinal"] == 2 and stall["blocked_operations"] == 1
        view = backend.operation_view("opt3")
        assert view["state"] == "QUEUED"
        assert view["waiting_on_ordinal"] == 2 and view["gap_stalled_for"] == pytest.approx(10.0)

    def test_control_claim_heartbeat_expires_the_stall(self):
        backend, clock = self.stalled_backend()
        clock["now"] += 31
        assert backend.claim_ready_control_operations()["operations"] == []  # the sweep fires here
        view = backend.operation_view("opt3")
        assert view["state"] == "FAILED" and view["error_category"] == "user"
        assert "missing ordinal 2" in view["error"]
        assert backend.service_info()["gap_stalls"] == []
        # Clean resubmit: the sealed hole is poison-neutral, so the new
        # optim_step STEPS fb1's intact window instead of discarding it.
        backend.enqueue_operation("X", "opt4", 4, "optim_step", {"adam_params": {"learning_rate": 1e-4}})
        [operation] = backend.claim_ready_control_operations()["operations"]
        assert operation["operation_id"] == "opt4" and "poison" not in operation


class TestClaimedTtlSurface:
    """Backend wiring of the claimed-op TTL: an orphaned CLAIMED head terminal-fails typed instead of blocking."""

    def orphaned_backend(self, ttl=60.0):
        backend = ready_backend()
        backend.operations.claimed_ttl = ttl
        clock = {"now": 1000.0}
        backend.operations._time = lambda: clock["now"]
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        # Claimed, then the claiming executor vanished (e.g. restart lost its in-memory runtimes).
        assert backend.claim_data_operation(*reg_key(backend)) is not None
        return backend, clock

    def test_flag_reaches_the_ledger_with_a_default(self):
        assert make_backend().operations.claimed_ttl == 1800.0
        args = SimpleNamespace(multi_lora_n_adapters=4, tinker_operation_claimed_ttl=5.0)
        assert MultiLoraOperationBackend(args, "http://unused").operations.claimed_ttl == 5.0

    def test_heartbeat_fails_the_orphan_typed_server_and_unblocks_the_queue(self):
        backend, clock = self.orphaned_backend()
        clock["now"] += 61
        backend.enqueue_operation("X", "opt2", 2, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert op["operation_id"] == "opt2"  # the swept orphan no longer blocks the queue head
        view = backend.operations.get("fb1")
        assert view["state"] == "FAILED" and view["error_category"] == "server"
        assert "'fb1'" in view["error"] and "61s" in view["error"] and "forward_backward" in view["error"]

    def test_sweep_routes_through_the_lease_releasing_batch_finalizer(self):
        backend, clock = self.orphaned_backend()
        calls = []
        original = backend.fail_tinker_batch

        def spy(operation_ids, error, lease_metadata=None):
            calls.append((operation_ids, lease_metadata))
            original(operation_ids, error, lease_metadata)

        backend.fail_tinker_batch = spy
        clock["now"] += 61
        assert backend.service_info()["operation_claimed_ttl"] == 60.0
        # No lease metadata exists for an orphaned claim; the finalizer's finally covers batches that carry one.
        assert calls == [(["fb1"], None)]

    def test_younger_claim_survives_the_sweep(self):
        backend, clock = self.orphaned_backend()
        clock["now"] += 59
        backend.service_info()
        assert backend.operations.get("fb1")["state"] == "CLAIMED"

    def test_late_completion_of_a_swept_operation_is_ignored_not_a_crash(self):
        backend, clock = self.orphaned_backend()
        clock["now"] += 61
        backend.service_info()  # sweeps fb1 to FAILED
        backend.complete_control_operations({"fb1": dict(ok=True, result={})})
        assert backend.operations.get("fb1")["state"] == "FAILED"
