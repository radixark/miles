"""Refactor-equivalence capture for the gradient-window state machine
(codex-rollout-fullparameter-design-0810 §3.4): scripted operation sequences
through the CURRENT TinkerBackend, asserting a field-by-field fingerprint of
the ledger views and the registry's step/dirty/lifecycle state after every
mutating call.

These are the two sacred carriers of the tinker backend (verified bit-for-bit
on H200): poison-window semantics and strict per-registration ordinal
execution. Any refactor that moves step/dirty ownership (e.g. into a
registration-keyed GradientWindowTracker) must keep every fingerprint below
byte-identical — the registry's ``record.step`` and pin-backed ``is_dirty``
remain valid observation points because the refactor keeps them as exact
Multi-LoRA lifecycle mirrors of the tracker state.
"""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

from miles.ray.tinker_backend.backend import TinkerBackend
from miles.ray.tinker_backend.config import AdapterRunConfig


def make_backend(max_adapters: int = 4) -> TinkerBackend:
    args = SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save="/tmp/tinker-test-save",
        lora_rank=32,
        lora_alpha=64,
        hf_checkpoint="Qwen/Qwen3-0.6B",
    )
    return TinkerBackend(args, "http://unused")


def ready(backend: TinkerBackend, name: str, **config) -> str:
    asyncio.run(backend.register(name, AdapterRunConfig(**config)))
    backend.registry.mark_ready([name])
    return backend.registry.find(name).registration_id


def fb_payload(n=1):
    return {
        "samples": [
            {"tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1], "loss_weights": [1.0, 1.0]}
            for _ in range(n)
        ],
        "loss": {"loss_fn": "cross_entropy"},
    }


def window_state(backend: TinkerBackend, name: str) -> dict:
    """The per-registration training-stream state: step clocks, dirty flag,
    and lifecycle. Field-by-field — a refactor must reproduce ALL of it."""
    record = backend.registry.records.get(name)
    if record is None:
        return {"missing": True}
    return dict(
        state=record.state.value,
        slot=record.slot,
        step=record.step,
        start_step=record.start_step,
        serving_version=record.serving_version,
        dirty=backend.registry.is_dirty(name),
    )


def op_state(backend: TinkerBackend, op_id: str) -> dict:
    """Ledger view minus the identity constants asserted once at enqueue."""
    view = backend.operations.get(op_id)
    return dict(
        state=view["state"],
        result=view["result"],
        error=view["error"],
        error_category=view["error_category"],
    )


class TestForwardBackwardWindow:
    def test_fb_commit_marks_dirty_and_forward_commit_does_not(self):
        backend = make_backend()
        rid = ready(backend, "A")

        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=False
        )

        assert backend.operations.claim_data_operation("A", rid)["operation_id"] == "fb1"
        backend.commit_tinker_batch([("A", rid)], ["fb1"], {"fb1": [[-0.1, -0.2]]})
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=True
        )
        assert op_state(backend, "fb1") == dict(
            state="SUCCEEDED",
            result={
                "logprobs": [[-0.1, -0.2]],
                "metrics": {"loss:sum": 0.30000000000000004, "unmasked_tokens:sum": 2.0},
            },
            error=None,
            error_category=None,
        )

        # forward: logprobs only, never dirty (the commit lists no accumulator).
        backend.enqueue_operation("A", "fwd2", 2, "forward", {"samples": fb_payload()["samples"]})
        backend.operations.claim_data_operation("A", rid)
        backend.commit_tinker_batch([], ["fwd2"], {"fwd2": [[-0.3, -0.4]]})
        assert op_state(backend, "fwd2") == dict(
            state="SUCCEEDED", result={"logprobs": [[-0.3, -0.4]]}, error=None, error_category=None
        )
        # dirty is still True from fb1, untouched by the forward.
        assert window_state(backend, "A")["dirty"] is True

        backend2 = make_backend()
        rid2 = ready(backend2, "B")
        backend2.enqueue_operation("B", "fwd1", 1, "forward", {"samples": fb_payload()["samples"]})
        backend2.operations.claim_data_operation("B", rid2)
        backend2.commit_tinker_batch([], ["fwd1"], {"fwd1": [[-0.3, -0.4]]})
        assert window_state(backend2, "B") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=False
        )


class TestPoisonWindow:
    def test_failed_chunk_poisons_the_window_field_by_field(self):
        """#2258 §5 end to end: fail one chunk, succeed another, then watch the
        pending optim_step claim carry poison, execute as a discard, and leave
        the next window clean."""
        backend = make_backend()
        rid = ready(backend, "A")

        # Window: fb1 FAILS, fb2 succeeds — partial gradients.
        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.operations.fail("fb1", "bad chunk", "user")
        backend.enqueue_operation("A", "fb2", 2, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.commit_tinker_batch([("A", rid)], ["fb2"], {"fb2": [[-0.1, -0.2]]})
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=True
        )

        backend.enqueue_operation("A", "opt3", 3, "optim_step")
        claimed = backend.claim_ready_control_operations()
        [op] = claimed["operations"]
        assert op["operation_id"] == "opt3"
        assert op["step"] == 0 and op["serving_version"] == 0
        # Binding truth rides the control batch's lease, not the claim.
        assert claimed["lease"]["bindings_by_operation"] == [["opt3", ["A", rid, 0]]]
        assert op["poison"] == (
            "a forward_backward in this gradient window failed (forward_backward ordinal 1 FAILED: bad chunk); "
            "the window's accumulated gradients were discarded — resubmit the batch and optim_step again"
        )

        # The trainer runs the discard on every rank and reports a user
        # failure that confirms the window was physically consumed.
        backend.complete_control_operations(
            {"opt3": dict(ok=False, error=op["poison"], category="user", gradient_window_consumed=True)}
        )
        assert op_state(backend, "opt3") == dict(
            state="FAILED", result=None, error=op["poison"], error_category="user"
        )
        # Step clock untouched, dirty cleared by the executed discard.
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=False
        )

        # The executed (poison-consuming) optim delimits: the next window is clean.
        backend.enqueue_operation("A", "fb4", 4, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.commit_tinker_batch([("A", rid)], ["fb4"], {"fb4": [[-0.1, -0.2]]})
        backend.enqueue_operation("A", "opt5", 5, "optim_step")
        [clean] = backend.claim_ready_control_operations()["operations"]
        assert clean["operation_id"] == "opt5" and "poison" not in clean
        backend.complete_control_operations({"opt5": dict(ok=True, result={"grad_norm": 0.5})})
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=1, start_step=0, serving_version=0, dirty=False
        )

    def test_cancelled_optim_is_not_a_window_delimiter(self):
        """An optim_step that never executed (cancelled while QUEUED) must not
        delimit: the poison from the failed chunk survives to the NEXT
        actually-executed optim_step."""
        backend = make_backend()
        rid = ready(backend, "A")
        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.operations.fail("fb1", "bad chunk", "user")

        backend.enqueue_operation("A", "opt2", 2, "optim_step")
        backend.operations.cancel("opt2")
        assert op_state(backend, "opt2") == dict(
            state="CANCELLED", result=None, error="cancelled by client", error_category="user"
        )

        backend.enqueue_operation("A", "opt3", 3, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert op["operation_id"] == "opt3"
        assert "forward_backward ordinal 1 FAILED" in op["poison"]

    def test_clean_optim_step_without_prior_fb_succeeds(self):
        """Current behavior allows a clean optim_step (no F/B in the window);
        no dirty prerequisite may ever be added."""
        backend = make_backend()
        ready(backend, "A")
        backend.enqueue_operation("A", "opt1", 1, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        assert "poison" not in op
        backend.complete_control_operations({"opt1": dict(ok=True, result={"grad_norm": 0.0})})
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=1, start_step=0, serving_version=0, dirty=False
        )

    def test_vetoed_step_clears_dirty_without_advancing_the_clock(self):
        backend = make_backend()
        rid = ready(backend, "A")
        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.commit_tinker_batch([("A", rid)], ["fb1"], {"fb1": [[-0.1, -0.2]]})
        backend.enqueue_operation("A", "opt2", 2, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations(
            {
                "opt2": dict(
                    ok=False,
                    error="non-finite gradients; step vetoed and gradients cleared",
                    category="server",
                    gradient_window_consumed=True,
                )
            }
        )
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=False
        )


class TestStepClockLifecycle:
    def test_num_step_bound_auto_retires_on_the_committed_step(self):
        backend = make_backend()
        rid = ready(backend, "A", num_step=1)
        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.commit_tinker_batch([("A", rid)], ["fb1"], {"fb1": [[-0.1, -0.2]]})
        backend.enqueue_operation("A", "opt2", 2, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations({"opt2": dict(ok=True, result={"grad_norm": 0.5})})
        assert window_state(backend, "A") == dict(
            state="RETIRING", slot=0, step=1, start_step=0, serving_version=0, dirty=False
        )

    def test_load_state_success_repositions_both_clocks(self):
        backend = make_backend()
        ready(backend, "A")
        backend.enqueue_operation("A", "load1", 1, "load_state", {"path": "/tmp/state"})
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations({"load1": dict(ok=True, result={"step": 42, "path": "/tmp/state"})})
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=42, start_step=42, serving_version=0, dirty=False
        )

    def test_dirty_gate_fails_state_moves_until_the_window_is_consumed(self):
        backend = make_backend()
        rid = ready(backend, "A")
        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid)
        backend.commit_tinker_batch([("A", rid)], ["fb1"], {"fb1": [[-0.1, -0.2]]})

        backend.enqueue_operation("A", "save2", 2, "save_state", {"tag": "t0"})
        assert backend.claim_ready_control_operations() == {"operations": [], "lease": None}
        assert op_state(backend, "save2") == dict(
            state="FAILED",
            result=None,
            error="adapter 'A' holds unstepped gradients; optim_step (or deregister) before save_state",
            error_category="user",
        )

        backend.enqueue_operation("A", "opt3", 3, "optim_step")
        [op] = backend.claim_ready_control_operations()["operations"]
        backend.complete_control_operations({"opt3": dict(ok=True, result={"grad_norm": 0.5})})
        backend.enqueue_operation("A", "save4", 4, "save_state", {"tag": "t0"})
        [save_op] = backend.claim_ready_control_operations()["operations"]
        assert save_op["operation_id"] == "save4"


class TestIndependentWindows:
    def test_two_registrations_never_share_step_or_dirty_state(self):
        backend = make_backend()
        rid_a = ready(backend, "A")
        rid_b = ready(backend, "B")

        # A's window poisons; B's succeeds and steps.
        backend.enqueue_operation("A", "a-fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("A", rid_a)
        backend.operations.fail("a-fb1", "bad chunk", "user")

        backend.enqueue_operation("B", "b-fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("B", rid_b)
        backend.commit_tinker_batch([("B", rid_b)], ["b-fb1"], {"b-fb1": [[-0.1, -0.2]]})

        backend.enqueue_operation("A", "a-opt2", 2, "optim_step")
        backend.enqueue_operation("B", "b-opt2", 2, "optim_step")
        claimed = {op["operation_id"]: op for op in backend.claim_ready_control_operations()["operations"]}
        assert set(claimed) == {"a-opt2", "b-opt2"}
        assert "forward_backward ordinal 1 FAILED" in claimed["a-opt2"]["poison"]
        assert "poison" not in claimed["b-opt2"]

        backend.complete_control_operations(
            {
                "a-opt2": dict(ok=False, error=claimed["a-opt2"]["poison"], category="user"),
                "b-opt2": dict(ok=True, result={"grad_norm": 0.5}),
            }
        )
        assert window_state(backend, "A") == dict(
            state="READY", slot=0, step=0, start_step=0, serving_version=0, dirty=False
        )
        assert window_state(backend, "B") == dict(
            state="READY", slot=1, step=1, start_step=0, serving_version=0, dirty=False
        )
