"""What the rollout manager reports to run observers: each step's samples and the data buffer."""

from __future__ import annotations

from collections import defaultdict

from miles.ray.rollout.debug_data import samples_of
from miles.utils.observers import Observers, RolloutSample
from miles.utils.types import Sample


def report_rollout_samples(data, *, rollout_id: int, evaluation: bool) -> None:
    """``data`` is a step's rollout data, as ``samples_of`` reads it."""
    observers = Observers()
    if not observers.hears("rollout_samples"):
        return
    with observers.isolated("rollout_samples"):
        samples = samples_of(data, evaluation=evaluation)
        observers.rollout_samples(rollout_id, evaluation=evaluation, samples=_rollout_sample_records(samples))


def report_data_buffer(length: int) -> None:
    observers = Observers()
    if observers.active:
        observers.data_buffer(length)


def _rollout_sample_records(samples: list[Sample]) -> list[RolloutSample]:
    occurrences: defaultdict[int, int] = defaultdict(int)
    records = []
    for sample in samples:
        records.append(_rollout_sample_record(sample, sample_occurrence=occurrences[sample.index]))
        occurrences[sample.index] += 1
    return records


def _rollout_sample_record(sample: Sample, *, sample_occurrence: int) -> RolloutSample:
    prefill_lags = [lag for call in sample.weight_versions if (lag := call.prefill_lag) is not None]
    return RolloutSample(
        sample_index=sample.index,
        sample_occurrence=sample_occurrence,
        group_index=sample.group_index,
        status=sample.status.value,
        remove_sample=sample.remove_sample,
        reward=float(sample.reward) if isinstance(sample.reward, (int, float)) else None,
        response_length=sample.response_length,
        tokens=list(sample.tokens),
        loss_mask=sample.loss_mask,
        rollout_log_probs=sample.rollout_log_probs,
        weight_versions=[span.version for span in sample.all_weight_version_spans],
        turns=len(sample.weight_versions) or None,
        prefill_weight_version_min=sample.oldest_prefill_weight_version,
        prefill_lag=max(prefill_lags) if prefill_lags else None,
        tool_calls=_tool_call_count(sample),
        non_generation_time=sample.non_generation_time,
        spec_correct_drafts=sample.spec_info.spec_num_correct_drafts,
        spec_proposed_drafts=sample.spec_info.spec_num_proposed_drafts,
        prefix_cached_tokens=sample.prefix_cache_info.cached_tokens,
        prefix_prompt_tokens=sample.prefix_cache_info.total_prompt_tokens,
        prompt=sample.prompt,
        messages=(sample.metadata or {}).get("messages"),
    )


def _tool_call_count(sample: Sample) -> int | None:
    # a plain-string prompt (single-turn math) has no message structure to count
    if not isinstance(sample.prompt, list):
        return None
    return sum(1 for message in sample.prompt if isinstance(message, dict) and message.get("role") == "tool")
