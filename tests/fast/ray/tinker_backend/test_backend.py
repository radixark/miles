"""TinkerBackend control plane: registration resolution, the v1 compatibility
preflight (boundary rejection, never GPU-side), control-operation claims with
authoritative clocks and dirty gates, and commit bookkeeping."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

from miles.ray.tinker_backend.backend import TinkerBackend
from miles.ray.tinker_backend.config import AdapterRunConfig
from miles.ray.tinker_backend.registry import AdapterState
from miles.utils.tinker_backend import make_rid, parse_adapter


def make_backend(max_adapters: int = 4) -> TinkerBackend:
    args = SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save="/tmp/tinker-test-save",
        lora_rank=32,
        lora_alpha=64,
        hf_checkpoint="Qwen/Qwen3-0.6B",
    )
    return TinkerBackend(args, "http://unused")


def register(backend, name="X", **overrides) -> dict:
    return asyncio.run(backend.register(name, AdapterRunConfig(**overrides)))


def ready_backend(num_step=None):
    backend = make_backend()
    register(backend, num_step=num_step)
    backend.registry.mark_ready(["X"])
    return backend


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

    def test_rid_roundtrip_preserves_names_with_underscores(self):
        for name in ["a", "adapter_a", "weird__name", "x_y_z"]:
            assert parse_adapter(make_rid(name, "reg1")) == name


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
        bad["samples"][0]["advantages"] = [1.0]  # response_length is 2
        with pytest.raises(ValueError, match="length response_length"):
            backend.enqueue_operation("X", "op1", 1, "forward_backward", bad)

    def test_adam_params_validated(self):
        backend = ready_backend()
        with pytest.raises(ValueError, match="unknown adam_params field"):
            backend.enqueue_operation("X", "op1", 1, "optim_step", {"adam_params": {"lr": 1e-4}})
        with pytest.raises(ValueError, match="must be a number"):
            backend.enqueue_operation("X", "op1", 1, "optim_step", {"adam_params": {"learning_rate": "fast"}})

    def test_unknown_kind_and_missing_path(self):
        backend = ready_backend()
        with pytest.raises(ValueError, match="unknown operation kind"):
            backend.enqueue_operation("X", "op1", 1, "publish_snapshot")
        with pytest.raises(ValueError, match="needs a 'path'"):
            backend.enqueue_operation("X", "op1", 1, "load_state", {})

    def test_valid_operations_enqueue(self):
        backend = ready_backend()
        view = backend.enqueue_operation("X", "op1", 1, "forward_backward", fb_payload())
        assert view["state"] == "QUEUED"
        assert backend.enqueue_operation("X", "op2", 2, "optim_step", {"adam_params": {"learning_rate": 3e-4}})


class TestControlClaims:
    def test_claim_requires_ready_and_serialization(self):
        backend = make_backend()
        register(backend)
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        assert backend.claim_ready_control_operations() == []  # PENDING, not READY
        backend.registry.mark_ready(["X"])
        [op] = backend.claim_ready_control_operations()
        assert op["operation_id"] == "opt1" and op["slot"] == 0

    def test_claim_carries_authoritative_clocks(self):
        backend = ready_backend()
        backend.registry.set_step("X", 7)
        backend.registry.record_weight_update(["X"])
        backend.enqueue_operation("X", "pub1", 1, "save_weights_for_sampler")
        [op] = backend.claim_ready_control_operations()
        assert op["step"] == 7 and op["serving_version"] == 1

    def test_dirty_slot_fails_state_moves_but_allows_publish(self):
        backend = ready_backend()
        backend.commit_tinker_batch(["X"], [])
        backend.enqueue_operation("X", "save1", 1, "save_state", {"tag": "t0"})
        assert backend.claim_ready_control_operations() == []
        view = backend.operations.get("save1")
        assert view["state"] == "FAILED" and "unstepped gradients" in view["error"]

        backend.enqueue_operation("X", "pub1", 2, "save_weights_for_sampler")
        [op] = backend.claim_ready_control_operations()
        assert op["operation_id"] == "pub1"  # publishing pre-step weights is fine

    def test_success_advances_step_and_releases_pin(self):
        backend = ready_backend(num_step=2)
        backend.commit_tinker_batch(["X"], [])
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        [op] = backend.claim_ready_control_operations()
        backend.complete_control_operations({op["operation_id"]: dict(ok=True, result={"grad_norm": 0.5})})
        record = backend.registry.find("X")
        assert record.step == 1 and not backend.registry.is_dirty("X")

    def test_veto_fails_without_advancing(self):
        backend = ready_backend()
        backend.commit_tinker_batch(["X"], [])
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        [op] = backend.claim_ready_control_operations()
        backend.complete_control_operations({op["operation_id"]: dict(ok=False, error="veto", category="server")})
        assert backend.registry.find("X").step == 0
        assert not backend.registry.is_dirty("X")

    def test_load_state_repositions_the_clock(self):
        backend = ready_backend()
        backend.enqueue_operation("X", "load1", 1, "load_state", {"path": "/tmp/state"})
        [op] = backend.claim_ready_control_operations()
        backend.complete_control_operations({op["operation_id"]: dict(ok=True, result={"step": 42})})
        record = backend.registry.find("X")
        assert record.step == 42 and record.start_step == 42


class TestCommitAndFence:
    def test_commit_completes_data_ops_with_row_ordered_logprobs(self):
        backend = ready_backend()
        reg_id = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", fb_payload())
        backend.operations.claim_data_operation("X", reg_id)
        backend.commit_tinker_batch(["X"], ["fb1"], {"fb1": [[-0.1, -0.2]]})
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


def test_service_info_reports_the_v1_matrix():
    backend = ready_backend()
    info = backend.service_info()
    assert info["base_model"] == "Qwen/Qwen3-0.6B"
    assert info["lora_rank_max"] == 32 and info["n_adapters"] == 4
    assert info["occupied_slots"] == [0] and info["ready_adapters"] == ["X"]
    assert info["supported_loss_fns"] == ["cross_entropy", "importance_sampling", "ppo"]
