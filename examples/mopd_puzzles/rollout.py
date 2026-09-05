"""Puzzle accuracy and domain metrics for verifier training and distillation."""

import time
from collections import defaultdict

import numpy as np

from examples.mopd_puzzles.tasks import score
from miles.rollout.on_policy_distillation import reward_func as opd_reward_func
from miles.rollout.sglang_rollout import generate_rollout as sglang_generate_rollout


async def reward_func(args, sample, **kwargs):
    start = time.monotonic()
    reward = await opd_reward_func(args, sample, **kwargs)
    sample.metadata["opd_scoring_seconds"] = time.monotonic() - start
    # raw_reward is a logging field; the OPD postprocessor still supplies the
    # training rewards and fixed teacher scores independently.
    sample.metadata["raw_reward"] = score(sample.response, sample.label)
    return reward


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("Use examples.mopd_puzzles.evaluate for evaluation")
    output = sglang_generate_rollout(args, rollout_id, data_source, evaluation=False)
    domains = defaultdict(list)
    for group in output.samples:
        for sample in group:
            domains[sample.metadata["domain"]].append(sample)
    metrics = dict(output.metrics or {})
    for domain, samples in domains.items():
        prefix = f"rollout/domain/{domain}"
        metrics[f"{prefix}/samples"] = len(samples)
        metrics[f"{prefix}/accuracy"] = np.mean([score(s.response, s.label) for s in samples]).item()
        lengths = [s.response_length for s in samples]
        metrics[f"{prefix}/response_tokens"] = sum(lengths)
        metrics[f"{prefix}/median_response_length"] = np.median(lengths).item()
        metrics[f"{prefix}/p95_response_length"] = np.percentile(lengths, 95).item()
        times = [s.metadata["opd_scoring_seconds"] for s in samples if "opd_scoring_seconds" in s.metadata]
        if times:
            metrics[f"{prefix}/teacher_request_seconds_mean"] = np.mean(times).item()
            metrics[f"{prefix}/teacher_request_seconds_p95"] = np.percentile(times, 95).item()
    output.metrics = metrics
    return output
