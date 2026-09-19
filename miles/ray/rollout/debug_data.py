import json
import logging
import math
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path

import torch

from miles.rollout.filter_hub.base_types import iter_samples
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def trajectory_rows(samples: list[Sample]) -> list[dict]:
    """One row per sample that recorded a raw conversation
    (``metadata["messages"]``, attached by the session / multi_turn paths).

    ``sample_occurrence`` counts over the FULL sample list -- the numbering the
    dashboard uses everywhere -- not over the recorded subset, so a sample
    stays addressable even when an earlier leaf of the same index recorded no
    conversation."""
    rows = []
    occurrences: defaultdict[int, int] = defaultdict(int)
    for sample in samples:
        occurrence = occurrences[sample.index]
        occurrences[sample.index] += 1
        messages = sample.metadata.get("messages") if sample.metadata else None
        if messages is None:
            continue
        rows.append(
            dict(
                sample_index=sample.index,
                sample_occurrence=occurrence,
                group_index=sample.group_index,
                status=sample.status.value,
                reward=sample.reward if isinstance(sample.reward, (int, float)) else None,
                prompt=sample.prompt,
                messages=messages,
            )
        )
    return rows


_warned_no_polars = False


def save_dashboard_columns(samples: list[Sample], path: Path) -> None:
    """Point-read mirror of the per-token rollout columns (parquet, one row
    per sample) so the dashboard token view never full-loads the .pt."""
    global _warned_no_polars
    try:
        import polars as pl
    except ImportError:
        if not _warned_no_polars:
            logger.warning("polars not installed; skipping dashboard_columns dump (pip install miles[dashboard])")
            _warned_no_polars = True
        return
    schema = dict(
        sample_index=pl.Int32,
        response_length=pl.Int32,
        total_length=pl.Int32,
        tokens=pl.List(pl.Int32),
        loss_mask=pl.List(pl.Int8),
        rollout_log_probs=pl.List(pl.Float32),
    )
    frame = pl.DataFrame(
        [
            dict(
                sample_index=sample.index,
                response_length=sample.response_length,
                total_length=len(sample.tokens),
                tokens=list(sample.tokens),
                loss_mask=sample.loss_mask,
                rollout_log_probs=sample.rollout_log_probs,
            )
            for sample in samples
        ],
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp, row_group_size=8)
    tmp.replace(path)


def save_debug_trajectory_data(args, samples: list[Sample], rollout_id, evaluation: bool):
    if (path_template := args.save_debug_trajectory_data) is None:
        return
    rows = trajectory_rows(samples)
    if not rows:
        return  # no conversations: no file (the dashboard keys off its presence)
    path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
    logger.info(f"Save trajectory dump to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    for sample in samples:
        if sample.metadata:
            sample.metadata.pop("messages", None)  # the sidecar is their home; keep the .pt lean


def load_debug_rollout_data(args, rollout_id: int) -> tuple[list[Sample], dict]:
    data, metadata = _load_rollout_data_file(Path(args.load_debug_rollout_data.format(rollout_id=rollout_id)))
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data, metadata


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool, metadata: dict | None = None) -> None:
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        samples = [sample for info in data.values() for sample in info["samples"]] if evaluation else list(data)
        save_debug_trajectory_data(args, samples, rollout_id, evaluation)
        stem = ("eval_" if evaluation else "") + str(rollout_id)
        save_dashboard_columns(samples, path.parent.parent / "dashboard_columns" / f"rollout_{stem}.parquet")

        # TODO may improve the format
        dump_data = dict(samples=[sample.to_dict() for sample in samples])
        torch.save(dict(rollout_id=rollout_id, metadata=metadata or {}, **dump_data), path)


def summarize_rollout_candidates(args, groups, filter_outputs) -> dict:
    """Summarize finished group tasks, not just accepted training inputs.

    Invalid includes missing/nonfinite scores and strict-grader valid=False;
    format_invalid counts only COMPLETED responses, separately from truncation.
    """
    statuses, rewards, reasons = Counter(), Counter(), Counter()
    invalid = format_invalid = fully_completed = 0
    for group in groups:
        samples = list(iter_samples(group))
        fully_completed += bool(samples) and all(s.status == Sample.Status.COMPLETED for s in samples)
        for sample in samples:
            statuses[sample.status.value] += 1
            reward = sample.reward
            value = reward.get(args.reward_key) if isinstance(reward, dict) and args.reward_key else reward
            finite = isinstance(value, Real) and math.isfinite(value)
            rewards[str(value) if finite else "missing_or_invalid"] += 1
            invalid_grade = isinstance(reward, dict) and reward.get("valid") is False
            invalid += not finite or invalid_grade
            format_invalid += invalid_grade and sample.status == Sample.Status.COMPLETED
    for output in filter_outputs:
        if output is not None and not output.keep:
            reasons[output.reason or "unspecified"] += 1
    return dict(
        complete_groups=len(groups),
        fully_completed_groups=fully_completed,
        rejected_groups=sum(reasons.values()),
        unfiltered_groups=sum(output is None for output in filter_outputs),
        status_counts=dict(statuses),
        invalid_samples=invalid,
        format_invalid_samples=format_invalid,
        truncated_samples=statuses[Sample.Status.TRUNCATED.value],
        rewards=dict(rewards),
        filter_reason_counts=dict(reasons),
    )


def save_rollout_candidate_evidence(args, rollout_id, groups, filter_outputs, selected, summary, *, error):
    """Opt-in diagnostic sidecar; intentionally NOT the training/replay schema.

    Only lightweight text/grading evidence is serialized, never replay tensors.
    Nothing is retained after the collector returns and no Sample is mutated.
    """
    if (template := getattr(args, "save_debug_rollout_data", None)) is None:
        return None
    training_path = Path(template.format(rollout_id=rollout_id))
    outcome = "failure" if error is not None else "success"
    # Keep non-numeric filenames outside rollout_data/*.pt dashboard discovery.
    path = training_path.parent / "candidates" / f"{training_path.stem}.{outcome}.pt"
    selected_ids = {id(group) for group in selected}
    candidates = []
    for group, output in zip(groups, filter_outputs, strict=True):
        candidates.append(
            dict(
                selected_for_batch=id(group) in selected_ids,
                filter_keep=bool(output.keep) if output is not None else None,
                filter_reason=output.reason if output is not None else "not_processed",
                samples=[
                    dict(
                        index=sample.index,
                        group_index=sample.group_index,
                        rollout_id=sample.rollout_id,
                        prompt=sample.prompt,
                        response=sample.response,
                        response_length=sample.response_length,
                        label=sample.label,
                        status=sample.status.value,
                        reward=sample.reward,
                        remove_sample=sample.remove_sample,
                        effective_response_length=sample.effective_response_length,
                    )
                    for sample in iter_samples(group)
                ],
            )
        )
    payload = dict(
        kind="rollout_candidate_evidence",
        schema_version=1,
        diagnostic_only=True,
        rollout_id=rollout_id,
        outcome=outcome,
        error=error,
        summary=summary,
        candidate_groups=candidates,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class RolloutDataInjectionUtil:
    @classmethod
    def should_inject(cls, args, rollout_id: int) -> bool:
        if args.ci_inject_rollout_data_path is None:
            return False
        return rollout_id >= args.ci_inject_rollout_data_start_rollout_id

    @classmethod
    def load(cls, args, rollout_id: int) -> tuple[list[Sample], dict]:
        path = Path(args.ci_inject_rollout_data_path.format(rollout_id=rollout_id))
        assert path.is_file(), f"Recorded rollout data to inject is missing: {path}"
        logger.info(f"CI rollout-data injection: replacing generated data of rollout {rollout_id} with {path}")
        return _load_rollout_data_file(path)

    @classmethod
    def assert_matches_generated(
        cls, args, *, generated: list[Sample], injected: list[Sample], rollout_id: int
    ) -> None:
        min_match_ratio: float = args.ci_inject_rollout_data_min_match_ratio
        assert len(generated) == len(
            injected
        ), f"rollout {rollout_id}: sample count mismatch, generated {len(generated)} vs injected {len(injected)}"

        ratios: list[float] = []
        for index, (generated_sample, injected_sample) in enumerate(zip(generated, injected, strict=True)):
            assert cls._prompt_tokens(generated_sample) == cls._prompt_tokens(injected_sample), (
                f"rollout {rollout_id}: prompt tokens mismatch at sample {index}; "
                "injected recording does not pair with the generated batch"
            )
            ratios.append(cls._response_token_match_ratio(generated_sample, injected_sample))

        mean_ratio = sum(ratios) / len(ratios)
        logger.info(
            f"CI rollout-data injection match for rollout {rollout_id}: "
            f"mean response token match {mean_ratio:.4f} (min {min(ratios):.4f}, "
            f"threshold {min_match_ratio}) over {len(ratios)} samples"
        )
        assert mean_ratio > min_match_ratio, (
            f"rollout {rollout_id}: generated responses match the injected recording at only "
            f"{mean_ratio:.4f} (threshold {min_match_ratio}); the engine "
            "weights likely diverged from the baseline beyond ulp-level drift"
        )

    @classmethod
    def _response_token_match_ratio(cls, a: Sample, b: Sample) -> float:
        response_a = cls._response_tokens(a)
        response_b = cls._response_tokens(b)
        denominator = max(len(response_a), len(response_b))
        if denominator == 0:
            return 1.0
        matched = sum(token_a == token_b for token_a, token_b in zip(response_a, response_b, strict=False))
        return matched / denominator

    @classmethod
    def _prompt_tokens(cls, sample: Sample) -> list[int]:
        return sample.tokens[: len(sample.tokens) - sample.response_length]

    @classmethod
    def _response_tokens(cls, sample: Sample) -> list[int]:
        return sample.tokens[len(sample.tokens) - sample.response_length :]


def _load_rollout_data_file(path: Path) -> tuple[list[Sample], dict]:
    payload = torch.load(path, weights_only=False)
    if payload.get("kind") == "rollout_candidate_evidence":
        raise ValueError(f"Diagnostic candidate evidence is not training/replay input: {path}")
    data = [Sample.from_dict(sample) for sample in payload["samples"]]
    metadata = payload.get("metadata") or {}
    return data, metadata
