"""Operation result metrics: backend-recomputed loss in the tinker SDK's
``name:reduction`` format, and the contract test proving the real SDK
combiner merges our chunked metrics exactly (D12)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import math

import pytest

from miles.ray.tinker_backend.backend import operation_result_metrics


def ce_payload(weights_by_sample, masks=None):
    samples = []
    for i, weights in enumerate(weights_by_sample):
        sample = {"tokens": [1] * (len(weights) + 2), "response_length": len(weights), "loss_weights": weights}
        if masks is not None:
            sample["loss_mask"] = masks[i]
        samples.append(sample)
    return {"samples": samples, "loss": {"loss_fn": "cross_entropy"}}


class TestMetricsValues:
    def test_cross_entropy_matches_hand_sum(self):
        payload = ce_payload([[0.5, 2.0], [1.0]])
        logprobs = [[-1.0, -2.0], [-3.0]]
        metrics = operation_result_metrics(payload, logprobs)
        assert metrics["loss:sum"] == pytest.approx(0.5 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0)
        assert metrics["unmasked_tokens:sum"] == 3.0

    def test_mask_gates_tokens(self):
        payload = ce_payload([[1.0, 1.0]], masks=[[1, 0]])
        metrics = operation_result_metrics(payload, [[-1.0, -9.0]])
        assert metrics["loss:sum"] == pytest.approx(1.0)
        assert metrics["unmasked_tokens:sum"] == 1.0

    def test_importance_sampling_and_ppo_clip(self):
        base = {
            "tokens": [1, 1, 1],
            "response_length": 2,
            "rollout_log_probs": [-1.0, -1.0],
            "advantages": [1.0, -2.0],
        }
        logprobs = [[-0.5, -1.5]]
        ratios = [math.exp(0.5), math.exp(-0.5)]

        metrics = operation_result_metrics({"samples": [base], "loss": {"loss_fn": "importance_sampling"}}, logprobs)
        assert metrics["loss:sum"] == pytest.approx(-(ratios[0] * 1.0) - (ratios[1] * -2.0))

        spec = {"loss_fn": "ppo", "loss_fn_config": {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}}
        metrics_ppo = operation_result_metrics({"samples": [base], "loss": spec}, logprobs)
        expected = -min(ratios[0] * 1.0, 1.1 * 1.0) - min(ratios[1] * -2.0, 0.9 * -2.0)
        assert metrics_ppo["loss:sum"] == pytest.approx(expected)
        assert metrics_ppo["loss:sum"] != pytest.approx(metrics["loss:sum"])

    def test_degenerate_ratio_cannot_overflow_the_recompute(self):
        # exp(1000) would raise OverflowError AFTER the GPU work landed,
        # leaving the operation without a terminal result; the recompute clamps.
        sample = {
            "tokens": [1, 1, 1],
            "response_length": 2,
            "rollout_log_probs": [-1000.0, -1.0],
            "advantages": [1.0, 1.0],
        }
        payload = {"samples": [sample], "loss": {"loss_fn": "importance_sampling"}}
        metrics = operation_result_metrics(payload, [[0.0, -1.0]])
        assert math.isfinite(metrics["loss:sum"])

    def test_sum_metrics_are_chunk_additive(self):
        whole = ce_payload([[0.5, 2.0], [1.0, 1.0, 1.0]])
        whole_logprobs = [[-1.0, -2.0], [-3.0, -4.0, -5.0]]
        chunks = [
            (ce_payload([[0.5, 2.0]]), [whole_logprobs[0]]),
            (ce_payload([[1.0, 1.0, 1.0]]), [whole_logprobs[1]]),
        ]
        whole_metrics = operation_result_metrics(whole, whole_logprobs)
        for key, value in whole_metrics.items():
            assert value == pytest.approx(sum(operation_result_metrics(p, lp)[key] for p, lp in chunks)), key


def test_sdk_combiner_merges_our_chunked_metrics():
    """The load-bearing contract: every key we emit uses a reduction the SDK
    combiner implements, and combining per-chunk outputs reproduces the
    whole-batch metrics (the client sees one merged result)."""
    helpers = pytest.importorskip("tinker.lib.chunked_fwdbwd_helpers")
    types = pytest.importorskip("tinker.types")

    whole = ce_payload([[0.5, 2.0], [1.0, 1.0, 1.0], [3.0]])
    whole_logprobs = [[-1.0, -2.0], [-3.0, -4.0, -5.0], [-0.25]]
    chunk_rows = [(0, 2), (2, 3)]

    def chunk_output(start, stop):
        payload = {"samples": whole["samples"][start:stop], "loss": whole["loss"]}
        metrics = operation_result_metrics(payload, whole_logprobs[start:stop])
        for key in metrics:
            assert key.split(":")[1] in helpers.REDUCE_MAP, f"SDK cannot reduce '{key}'"
        return types.ForwardBackwardOutput(
            loss_fn_output_type="scalar",
            metrics=metrics,
            loss_fn_outputs=[{} for _ in range(stop - start)],
        )

    combined = helpers.combine_fwd_bwd_output_results([chunk_output(*rows) for rows in chunk_rows])
    whole_metrics = operation_result_metrics(whole, whole_logprobs)
    assert combined.metrics["loss:sum"] == pytest.approx(whole_metrics["loss:sum"])
    assert combined.metrics["unmasked_tokens:sum"] == pytest.approx(whole_metrics["unmasked_tokens:sum"])
    assert len(combined.loss_fn_outputs) == 3
