"""SFT-only contract probes: the REAL, unmodified ``tinker==0.24.1`` SDK
drives the live HTTP stack through the teacher-forced cross-entropy path —
accumulation windows, prompt masking, checkpoint gating, rejected-Adam
recovery — plus the two verified pre-HTTP SDK failure modes and what the
server does (gap timeout) and cannot do (immediate cancel) about them.

Permanent adaptation of the codex-0817-sft-fix §3.2 adversarial suite; the
stack fixture (frontend -> real backend -> FakeDriver) comes from
test_sdk_contract. Skipped when the ``tinker`` wheel is not installed."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-a-cpu")

from concurrent.futures import TimeoutError as FutureTimeoutError  # noqa: E402

import pytest  # noqa: E402

tinker = pytest.importorskip("tinker")

from tests.fast.ray.tinker_frontend import test_sdk_contract as sdk_contract  # noqa: E402
from tinker import types  # noqa: E402

from miles.ray.multi_lora.operations import SealedGap  # noqa: E402

BASE = sdk_contract.BASE
make_datum = sdk_contract.make_datum
# Live-HTTP stack fixtures, reused by reference (module-scoped: this module
# gets its own frontend/backend/FakeDriver instance).
stack = sdk_contract.stack
service_client = sdk_contract.service_client


def _record_for(stack, client):
    session = client.model_id.split(":", 1)[0]
    [record] = [
        record
        for record in stack.backend.registry.records.values()
        if record.config.metadata.get("session_id") == session
    ]
    return record


async def _set_driver_paused(stack, paused):
    stack.driver.paused = paused


def sft_datum(prompt_tokens, completion_tokens):
    """The correct teacher-forced shape (codex-0817-sft-fix §2): position i
    predicts tokens[i+1], prompt-internal positions weight 0."""
    tokens = prompt_tokens + completion_tokens
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens),
        },
    )


class TestSftTrainingContract:
    def test_three_fb_accumulate_then_one_optim(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        fbs = [client.forward_backward([make_datum([10 + i, 20 + i, 30 + i])], "cross_entropy") for i in range(3)]
        optim = client.optim_step(types.AdamParams(learning_rate=2e-4))

        results = [future.result() for future in fbs]
        step = optim.result()

        assert [result.metrics["loss:sum"] for result in results] == pytest.approx([1.5, 1.5, 1.5])
        assert step.metrics["learning_rate"] == pytest.approx(2e-4)
        assert _record_for(stack, client).step == 1
        forward = client.forward([make_datum([1, 2, 3])], "cross_entropy").result()
        assert forward.loss_fn_outputs[0]["logprobs"].tolist() == pytest.approx([-0.51] * 3)

    def test_prompt_masked_sft_datum_separates_the_two_denominators(self, service_client):
        # The §7 regression: unmasked_tokens:sum counts ALL mask-active
        # positions (prompt included); loss_weight:sum is the SFT per-token
        # denominator (completion positions under 0/1 masking).
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        result = client.forward_backward([sft_datum([11, 12, 13, 14], [15, 16, 17])], "cross_entropy").result()
        assert result.metrics["unmasked_tokens:sum"] == pytest.approx(6.0)
        assert result.metrics["loss_weight:sum"] == pytest.approx(3.0)
        assert result.metrics["loss:sum"] / result.metrics["loss_weight:sum"] == pytest.approx(0.5)
        client.optim_step(types.AdamParams(learning_rate=0.0)).result()

    def test_zero_weight_prefix_and_fractional_ce_weights(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        datum = make_datum(
            [10, 11, 12, 13],
            targets=[999, 12, 888, 77],
            weights=[0.0, 0.5, 0.0, 2.0],
        )
        result = client.forward_backward([datum], "cross_entropy").result()

        # Zero-weight non-next-token targets are legal and normalized. The
        # loss remains the linear weighted token sum: -(-.5) * (0 + .5 + 0 + 2),
        # and the weighted-mean denominator is the weight sum itself.
        assert result.loss_fn_outputs[0]["logprobs"].tolist() == pytest.approx([-0.5] * 4)
        assert result.metrics["loss:sum"] == pytest.approx(1.25)
        assert result.metrics["unmasked_tokens:sum"] == pytest.approx(4.0)
        assert result.metrics["loss_weight:sum"] == pytest.approx(2.5)
        client.optim_step(types.AdamParams(learning_rate=0.0)).result()

    def test_dirty_save_rejection_preserves_gradients_for_later_step(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()

        with pytest.raises(tinker.RequestFailedError, match="unstepped gradients"):
            client.save_state("must-not-save-dirty").result()

        # The rejected save consumed its ordinal but did not clear the already
        # accumulated gradients: a later optimizer step consumes that window.
        result = client.optim_step(types.AdamParams(learning_rate=3e-4)).result()
        assert result.metrics["grad_norm"] == pytest.approx(0.125)
        assert _record_for(stack, client).step == 1
        assert client.save_state("clean-after-step").result().path.endswith("/weights/clean-after-step")

    def test_rejected_adam_params_do_not_drop_prior_gradients(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()

        with pytest.raises(tinker.RequestFailedError, match="learning_rate.*>= 0"):
            client.optim_step(types.AdamParams(learning_rate=-1.0)).result()

        assert _record_for(stack, client).step == 0
        result = client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        assert result.metrics["grad_norm"] == pytest.approx(0.125)
        assert _record_for(stack, client).step == 1

    def test_forward_is_no_grad_and_checkpointable(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        result = client.forward([make_datum([1, 2, 3])], "cross_entropy").result()
        assert result.metrics["loss:sum"] == pytest.approx(1.5)
        assert _record_for(stack, client).step == 0
        assert client.save_state("after-forward").result().path.endswith("/weights/after-forward")

    def test_result_timeout_is_non_destructive(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        stack.run(_set_driver_paused(stack, True))
        future = client.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
        try:
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.02)
        finally:
            stack.run(_set_driver_paused(stack, False))
        assert future.result(timeout=5).metrics["loss:sum"] == pytest.approx(1.5)
        client.optim_step(types.AdamParams(learning_rate=0.0)).result()


class TestPreHttpSdkFailureModes:
    def test_pre_http_serialization_hole_gap_times_out_typed_then_the_same_client_resubmits(
        self, stack, service_client
    ):
        """The verified 0.24.1 failure (codex-0817-sft-fix §4): NaN Adam params
        fail JSON serialization AFTER the SDK spent the seq — the request
        never reaches the server, and the next operation queues behind a hole
        no retry ever fills. The gap timeout converts the permanent stall into
        a typed failure naming the missing ordinal; the fence holds (the
        missing identity never executes, the step clock proves it ran nothing)
        and the SAME TrainingClient resubmits cleanly."""
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()

        async def set_gap_timeout(value):
            previous = stack.backend.operations.gap_timeout
            stack.backend.operations.gap_timeout = value
            return previous

        original = stack.run(set_gap_timeout(0.3))
        try:
            bad = client.optim_step(types.AdamParams(learning_rate=float("nan")))
            with pytest.raises(ValueError, match="Out of range float values|JSON compliant"):
                bad.result(timeout=2)  # dies client-side; ordinal 2 is spent, never posted

            later = client.optim_step(types.AdamParams(learning_rate=1e-4))
            with pytest.raises(tinker.RequestFailedError, match="missing ordinal 2"):
                later.result(timeout=30)
        finally:
            stack.run(set_gap_timeout(original))

        record = _record_for(stack, client)
        assert record.step == 0  # nothing executed for the sealed ordinal or the failed one

        # Clean resubmit on the SAME client: fb1's window was never poisoned
        # (the seal is neutral), so the new optim_step STEPS it.
        result = client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=30)
        assert result.metrics["grad_norm"] == pytest.approx(0.125)
        assert record.step == 1

        async def sealed_ordinals():
            queue = stack.backend.operations.queues[(record.name, record.registration_id)]
            return [ordinal for ordinal, holder in queue.by_ordinal.items() if isinstance(holder, SealedGap)]

        assert stack.run(sealed_ordinals()) == [2]

    def test_immediate_sdk_future_cancel_spends_turn_and_wedges_later_work(self, stack, service_client):
        """Characterize the unmodified 0.24.1 SDK cancellation contract
        (codex-0817-sft-fix §5): cancelling the underlying concurrent future
        before its coroutine enters ``_take_turn`` spends the request id
        without advancing the SDK turn counter. Later operations wait forever
        CLIENT-side; Miles receives no ordinal it could terminalize (the queue
        stays empty), so this is an upstream SDK gap — the deployment guidance
        (never ``.future().cancel()``; discard the client) is the mitigation,
        and the gap timeout covers only mixed cases where later submissions
        did reach the server."""
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        cancelled = []
        for _ in range(32):
            future = client.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
            cancelled.append(future.future().cancel())
        assert any(cancelled)

        later = client.optim_step(types.AdamParams(learning_rate=0.0))
        with pytest.raises(FutureTimeoutError):
            later.result(timeout=0.5)
        later.future().cancel()

        record = _record_for(stack, client)
        assert stack.backend.operations.queue_view(record.name, record.registration_id) == []
