import math

import pytest

from miles.ray.multi_lora.backend import operation_result_metrics


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
        assert metrics["loss_weight:sum"] == pytest.approx(1.0)

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
    assert combined.metrics["loss_weight:sum"] == pytest.approx(whole_metrics["loss_weight:sum"])
    assert len(combined.loss_fn_outputs) == 3


class TestLossWeightSum:
    def test_prompt_masked_sft_gets_the_completion_denominator(self):
        payload = ce_payload([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        metrics = operation_result_metrics(payload, [[-0.5] * 7])
        assert metrics["unmasked_tokens:sum"] == 7.0
        assert metrics["loss_weight:sum"] == pytest.approx(4.0)
        assert metrics["loss:sum"] / metrics["loss_weight:sum"] == pytest.approx(0.5)

    def test_fractional_weights_get_a_weighted_mean_denominator(self):
        metrics = operation_result_metrics(ce_payload([[0.0, 0.5, 0.0, 2.0]]), [[-0.5] * 4])
        assert metrics["loss:sum"] == pytest.approx(1.25)
        assert metrics["loss_weight:sum"] == pytest.approx(2.5)

    def test_all_zero_weight_chunk_still_reports_the_key(self):
        metrics = operation_result_metrics(ce_payload([[0.0, 0.0]]), [[-1.0, -1.0]])
        assert metrics["loss_weight:sum"] == 0.0

    def test_non_ce_losses_do_not_report_it(self):
        sample = {
            "tokens": [1, 1, 1],
            "response_length": 2,
            "rollout_log_probs": [-1.0, -1.0],
            "advantages": [1.0, 1.0],
        }
        payload = {"samples": [sample], "loss": {"loss_fn": "importance_sampling"}}
        assert "loss_weight:sum" not in operation_result_metrics(payload, [[-0.5, -1.5]])
